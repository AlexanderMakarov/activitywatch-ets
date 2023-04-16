import dataclasses
import collections
from typing import List, Dict, Tuple, Any

from ..config.config import LOG, AFK_RULE_PRIORITY, WATCHDOG_RULE_PRIORITY, TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS,\
                            BUCKET_DEBUG_RAW_RULE_RESULTS, BUCKET_DEBUG_FINAL_RULE_RESULTS, BUCKET_DEBUG_ACTIVITES
from .interval import Interval
from .input_entities import EventKeyHandler, Rule, Event
from .output_entities import RuleResult, Activity, AnalyzerResult
from ..helpers.helpers import seconds_to_int_timedelta, event_data_to_str


BUCKET_AFK_PREFIX = "aw-watcher-afk"
BUCKET_STOPWATCH_PREFIX = "aw-stopwatch"
DEFAULT_AFK_RULES = [EventKeyHandler('status', [
    Rule("afk", AFK_RULE_PRIORITY, skip=True),
    Rule("not-afk", 1, is_placeholder=True)
])]
DEFAULT_STOPWATCH_RULES = [EventKeyHandler('label', [
    Rule(".*", 0, subhandler=EventKeyHandler('running', [
        Rule("true", WATCHDOG_RULE_PRIORITY),
        # Even if it is the only event in interval it carries no activity.
        Rule("false", 0, skip=True)
    ]))
])]
ANALYZE_MODE_ACTIVITIES = "FOR_ACTIVITIES"
ANALYZE_MODE_DEBUG = "FOR_DEBUG"
ANALYZE_MODE_TUNER = "FOR_TUNER"
ANALYZE_MODES = [ANALYZE_MODE_ACTIVITIES, ANALYZE_MODE_DEBUG, ANALYZE_MODE_TUNER]


def increment_metric(metrics: Dict[str, Tuple[int, float]], metric_name: str, interval: Interval):
    """
    TODO replace with `Metrics` class. 
    """
    metric = metrics.get(metric_name, (0, 0))
    metrics[metric_name] = (metric[0] + 1, metric[1] + interval.get_duration())


def _intervals_duration(intervals: List[Interval]):
    # Note that 'last end minus first start' doesn't work due to possible gaps between intervals.
    return sum(x.get_duration() for x in intervals)


@dataclasses.dataclass
class RuleResultsWindow:
    """
    Subsidiary class to make sliding window on `Interval`-s linked list which accumulates `RuleResult`-s directly or
    supposingly belonging to one specific `Activity`.
    """
    rule_results: List[RuleResult]
    priority: int
    description: str
    duration: float

    def to_str(self, debug=False) -> str:
        """
        Converts to string.
        :param debug: Flag to make output more detailed.
        :return: String representation of the window.
        """
        prefix = f"{seconds_to_int_timedelta(self.duration)} with priority={self.priority}"
        if debug:
            events = dict((x.event.timestamp, x.event) for x in self.rule_results)
            events_str = "\n  ".join(
                f"{seconds_to_int_timedelta(x.duration.total_seconds())} {event_data_to_str(x)}"
                for x in events.values()
            )
            return f"{prefix}, description='{self.description}' and {len(events)} events:" + "\n  " + events_str
        else:
            return f"{prefix} and {len(self.rule_results)} rule results"

    def append(self, rule_result: RuleResult):
        """
        Appends new RuleResult to the window.
        :param rule_result: RuleResult to add.
        """
        self.rule_results.append(rule_result)
        self.duration += _intervals_duration(rule_result.intervals)
        # If new rule has more priority than all existing in window then update windows priority and description.
        if rule_result.rule.priority >= self.priority:
            self.priority = rule_result.rule.priority
            self.description = rule_result.description

    def to_activity(self) -> Activity:
        """
        Converts window to 'Activity'.
        :return: New 'Activity' from current window.
        """
        start_time = self.rule_results[0].intervals[0].start_time
        end_time = self.rule_results[-1].intervals[-1].end_time
        tmp = set(x.description for x in self.rule_results)
        description = ", ".join(sorted(tmp))
        return Activity(start_time, end_time, list(self.rule_results), description, self.duration)


def _is_new_activity(window: RuleResultsWindow, rule_result: RuleResult) -> bool:
    current_rule = rule_result.rule
    # Don't separate activities for rules with the same priority or same description.
    if current_rule.priority == window.priority or rule_result.description == window.description:
        return False
    # Separate all "independent" activities undoubtedly.
    if current_rule.priority >= AFK_RULE_PRIORITY:
        return rule_result.description != window.description
    return False


class ProblemReporter:
    """
    Class to encapsulate and report problem in analyzing intervals/events.
    """

    EVENT_WITHOUT_RULE = "EVENT_WITHOUT_RULE"
    MISSED_RULE = "MISSED_RULE"
    WEEK_RULE = "WEEK_RULE"
    TOO_SPECIFIC_RULE = "TOO_SPECIFIC_RULE"
    TOO_WIDE_RULE = "TOO_WIDE_RULE"

    SUPPORTED_PROBLEMS = [
        EVENT_WITHOUT_RULE,
        MISSED_RULE,
        WEEK_RULE,
        TOO_WIDE_RULE,
        TOO_SPECIFIC_RULE
    ]

    def __init__(self, rule: str, **kwargs) -> None:
        self.rule = rule
        self.kwargs: Dict[str, Any] = kwargs

    def report(self, disable_problems: List[str]):
        if self.rule in disable_problems:
            return
        elif self.rule == ProblemReporter.EVENT_WITHOUT_RULE:
            LOG.info("%s: Can't find handler for %s", self.rule, self.kwargs['event'])
        elif self.rule == ProblemReporter.MISSED_RULE:
            LOG.info("%s: Skipping %s because can't find rule for events in it:\n  %s", self.rule,
                     self.kwargs['cur_interval'], "\n  ".join(str(x) for x in self.kwargs['cur_interval'].events))
        elif self.rule == ProblemReporter.WEEK_RULE:
            cur_interval = self.kwargs['cur_interval']
            LOG.info("%s: Need to reveal rule for interval %s with %d events:\n  %s", self.rule,
                     cur_interval.to_str(only_time=True), len(cur_interval.events),
                     "\n  ".join(str(x) for x in cur_interval.events))
        elif self.rule == ProblemReporter.TOO_SPECIFIC_RULE:
            LOG.info("%s: On handling %s separated too small window %s", self.rule, self.kwargs['rule_result'],
                     self.kwargs['window'].to_str(True))
        elif self.rule == ProblemReporter.TOO_WIDE_RULE:
            LOG.info("%s: Got too long window %s after adding %s.", self.rule, self.kwargs['window'].to_str(False),
                     self.kwargs['rule_result'])
        else:
            LOG.warning("%s.report doesn't support '%s' with arguments: %s",
                        self.__class__.__name__, self.rule, self.kwargs)


def get_eventkeyhandlers_per_bucket_prefix(custom_rules: Dict[str, List[EventKeyHandler]])\
        -> Dict[str, List[EventKeyHandler]]:
    """
    Builds dictionary of EventKeyHandler-s per ActivityWatch bucket name prefix aka "rules" from
    default rules and custom ones.
    :param custom_rules: Custom dictionary of EventKeyHandler-s per ActivityWatch bucket name prefix.
    :return: Full set of rules.
    """
    bucket_prefix_to_ruleshandler: Dict[str, List[EventKeyHandler]] = dict(custom_rules)
    if BUCKET_AFK_PREFIX not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler[BUCKET_AFK_PREFIX] = DEFAULT_AFK_RULES
    if BUCKET_STOPWATCH_PREFIX not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler[BUCKET_STOPWATCH_PREFIX] = DEFAULT_STOPWATCH_RULES
    return bucket_prefix_to_ruleshandler


def find_handler_for_event(event: Event, eventkeyhandlers_per_bucket_prefix: Dict[str, List[EventKeyHandler]])\
        -> EventKeyHandler:
    """
    Finds handler for the given event.
    :param event: Event to find handler for.
    :param eventkeyhandlers_per_bucket_prefix: Map of rules to use.
    :return: Handler for the given event.
    """
    for bucket_prefix, bucket_handlers in eventkeyhandlers_per_bucket_prefix.items():
        if event.bucket_id.startswith(bucket_prefix):
            for bucket_handler in bucket_handlers:
                if bucket_handler.key in event.data:
                    return bucket_handler
    return None


def _find_out_rule_for_interval(interval: Interval, metrics: Dict[str, Tuple[int, float]],
        eventkeyhandlers_per_bucket_prefix: Dict[str, List[EventKeyHandler]], ignore_hints: List[str]) -> RuleResult:
    rule_result: RuleResult = None
    # Iterate all events to find out one with higher priority.
    for event in interval.events:
        # Search `EventKeyHandler` by 2 criteria:
        # 1) event bucket ID starts with handler 'bucket_id',
        # 2) handler key exists in event data.
        handler: EventKeyHandler = find_handler_for_event(event, eventkeyhandlers_per_bucket_prefix)
        if not handler:
            ProblemReporter(ProblemReporter.EVENT_WITHOUT_RULE, event=event).report(ignore_hints)
            increment_metric(metrics, 'intervals with unknows events', interval)
            continue  # It is OK, some events (inside interval) may don't have handlers intentionally.
        # Find rule by event data. Note that it may be sorted out by priority afterwards.
        rule, descriptions = handler.get_rule(event)
        if rule:
            # Keep only rule with the highest priority.
            if rule_result is None or rule.priority > rule_result.rule.priority:
                rule_result = RuleResult(
                    rule, event, "->".join(descriptions), [interval], []
                )
    return rule_result


def _window_to_activity(window: RuleResultsWindow, rule_result: RuleResult, activities: List[Activity],
                        is_build_debug_buckets: bool, activity_debug_events: List[Event], round_to: float,
                        ignore_hints: List[str]):
    if window.duration < round_to:
        ProblemReporter(ProblemReporter.TOO_SPECIFIC_RULE, rule_result=rule_result, window=window)\
            .report(ignore_hints)
        # Here may be a reason in "too specific" rules which leads to too small activities.
        # But current algorithm doesn't allow to keep few sliding windows in parallel
        # while person actually may switch too often between different activities.
        # TODO increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
    activity = window.to_activity()
    activities.append(activity)
    if is_build_debug_buckets:
        activity_debug_events.append(
            Event(BUCKET_DEBUG_ACTIVITES, activity.start_time, activity.duration, {
                'description': rule_result.description,
                'rule_results_count': len(activity.rule_results),
            }))

def analyze_intervals(interval: Interval, round_to: float, custom_rules: Dict[str, List[EventKeyHandler]],
        ignore_hints: List[str], analyze_mode: str = ANALYZE_MODE_ACTIVITIES) -> AnalyzerResult:
    """
    Analyzes linked list of 'Interval'-s to convert them into list of 'Activity'-es and provide explanation about
    how well it was done.
    :param interval: Linked list of 'Interval'-s to analyze.
    :param round_to: Both minimal summary interval length to show and step to align reporting intervals to.
    :param rules: User-specific/crafted map of `EventKeyHandler`-s to event buckets (by name prefix).
    :param ignore_hints: List of problems to disable in logs.
    :param analyze_mode: Mode to analyze intervals linked list. See constants starting with `ANALYZE_MODE`.
    :return: Tuple of:
    1 - List of assembled `Activity`-es.
    2 - `Counter` of intervals-by-rule description-s to sum of their durations.
        Like a `Activity`-es built by naive "equal" strategy, i.e. report with too many and too small activities.
    3 - Map of metrics to estimate report quality/coverage and improve rules.
        Key is name of metric, value is tuple [number_of_intervals, sum_of_durations].
    """
    if analyze_mode not in ANALYZE_MODES:
        raise ValueError(f"Analyze mode '{analyze_mode}' is not supported. Are supported only {ANALYZE_MODES}.")
    is_build_debug_buckets = analyze_mode in (ANALYZE_MODE_DEBUG, ANALYZE_MODE_TUNER)
    # Assemble full set of EventKeyHandler-s from predifined ones and custom.
    bucket_prefix_to_ruleshandler: Dict[str, List[EventKeyHandler]] = \
        get_eventkeyhandlers_per_bucket_prefix(custom_rules)
    # Prepare to loop through intervals with searching rules, building report and metrics.
    # Go to the first interval.
    cur_interval: Interval = interval.iterate_prev()
    # Make containers for outputs.
    rules_counter = collections.Counter()
    activities: List[Activity] = []
    metrics = {  # Put default metrics. Next it will be appended by per-rule metrics.
        'total intervals': (0, 0.0),
        'not afk intervals': (0, 0.0),
        'intervals with unknows events': (0, 0.0),
        'intervals without rules': (0, 0.0),
        'intervals merged to next rule': (0, 0.0),
        'intervals with rule to skip': (0, 0.0),
        'intervals need to reveal rule for': (0, 0.0),
    }
    raw_rule_result_debug_events = []
    final_rule_result_debug_events = []
    activity_debug_events = []
    # Prepare dummy `Interval` to use first interval on the very first iteration below.
    cur_interval = Interval(cur_interval.start_time, cur_interval.end_time, None, cur_interval)
    # Iterate all intervals, iterate rules for all events in each and choose highest by prioirty event
    # to choose dominant rule for interval which will become part of activity.
    deferred_intervals: List[Interval] = []  # `Interval`-s deffered as "append to next independent rule".
    window: RuleResultsWindow = None
    while cur_interval.next:
        cur_interval = cur_interval.next
        rule_result = _find_out_rule_for_interval(cur_interval, metrics, bucket_prefix_to_ruleshandler, ignore_hints)
        # Gather some important metrics at start.
        is_not_afk = any(x.bucket_id.startswith(BUCKET_AFK_PREFIX) for x in cur_interval.events)
        if is_not_afk:
            increment_metric(metrics, 'not afk intervals', cur_interval)
        # NOTE: order is very important below.
        # Decide whether to count this interval or not and update metrics.
        if rule_result is None:
            ProblemReporter(ProblemReporter.MISSED_RULE, cur_interval=cur_interval).report(ignore_hints)
            increment_metric(metrics, 'intervals without rules', cur_interval)
            continue
        # Update per-rule-name metric. It should include all rules (i.e. "skip", "placeholder", etc.).
        increment_metric(metrics, str(rule_result.rule), cur_interval)
        # Fill up BUCKET_DEBUG_RULE_RESULTS before handling deferred intervals
        # to represent logic of choosing rule by events in interval.
        if is_build_debug_buckets:
            raw_rule_result_debug_events.append(
                Event(BUCKET_DEBUG_RAW_RULE_RESULTS, cur_interval.start_time, cur_interval.get_duration(), {
                    'description': rule_result.description,
                    'rule': str(rule_result.rule),
                    'events_cnt': len(cur_interval.events),
                }))
        duration = _intervals_duration(rule_result.intervals)
        # Append deferred intervals if there are such.
        if deferred_intervals is not None:
            rule_result.intervals += deferred_intervals
            increment_metric(metrics, 'intervals merged to next rule', cur_interval)
            deferred_intervals = []
        # Check if rule says skip interval from the report.
        if rule_result.rule.skip:
            LOG.debug("Skipping %f sec %d interval(s) because of %s priority is highest for %s.",
                      round(duration, 1), len(rule_result.intervals), rule_result.rule, rule_result.event)
            increment_metric(metrics, 'intervals with rule to skip', cur_interval)
            continue
        # Check if rule is a placeholder and provide all information about interval to write appropriate rule for it.
        if rule_result.rule.is_placeholder:
            ProblemReporter(ProblemReporter.WEEK_RULE, cur_interval=cur_interval).report(ignore_hints)
            increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
        # Update 'total_intervals' metric.
        increment_metric(metrics, 'total intervals', cur_interval)
        # Defer interval if need. Note that `activity_counter` shouldn't be touched by this rule.
        if rule_result.rule.merge_next:
            deferred_intervals.append(cur_interval)
            continue
        # Update 'rules counter'.
        rules_counter[rule_result.description] += duration
        if is_build_debug_buckets:
            final_rule_result_debug_events.append(
                Event(BUCKET_DEBUG_FINAL_RULE_RESULTS, rule_result.intervals[0].start_time,
                      rules_counter[rule_result.description], {
                    'description': rule_result.description,
                    'rule': str(rule_result.rule),
                    'intervals_count': len(rule_result.intervals),
                }))
        # Decide if current window is completed, may be converted to `Activity` and next window started.
        is_start_new_window = True  # By default start new window.
        if window is not None:
            # Decide if current `RuleResult` is separate activity from previous ones and need to create `Activity` from
            # items accumulated in `activity_window` so far.
            if _is_new_activity(window, rule_result):
                _window_to_activity(window, rule_result, activities, is_build_debug_buckets, activity_debug_events,
                                    round_to, ignore_hints)
                window = None
            else:
                window.append(rule_result)
                if window.duration > TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS:
                    ProblemReporter(ProblemReporter.TOO_WIDE_RULE, rule_result=rule_result, window=window)\
                        .report(ignore_hints)
                    increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
                is_start_new_window = False
        if is_start_new_window:
            window = RuleResultsWindow([rule_result], rule_result.rule.priority, rule_result.description, duration)
    # Handle last window.
    if window is not None:
        _window_to_activity(window, rule_result, activities, is_build_debug_buckets, activity_debug_events,
                            round_to, ignore_hints)
    return AnalyzerResult(activities, rules_counter, metrics,
                          raw_rule_result_debug_events, final_rule_result_debug_events, activity_debug_events)
