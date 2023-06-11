import dataclasses
import collections
from operator import attrgetter
from typing import List, Dict, Set, Tuple, Any

from ..config.config import LOG, AFK_RULE_PRIORITY, WATCHDOG_RULE_PRIORITY, TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS,\
                            BUCKET_DEBUG_RAW_RULE_RESULTS, BUCKET_DEBUG_FINAL_RULE_RESULTS, BUCKET_DEBUG_ACTIVITES
from .interval import Interval, intervals_duration
from .input_entities import Event, Rule2
from .metrics import Metrics
from .output_entities import RuleResult, Activity, AnalyzerResult
from ..helpers.helpers import seconds_to_int_timedelta, event_data_to_str


BUCKET_AFK_PREFIX = "aw-watcher-afk"
BUCKET_STOPWATCH_PREFIX = "aw-stopwatch"
DEFAULT_RULES = [
    Rule2(BUCKET_AFK_PREFIX + ".*", 1, to_string="afk").with_subrules(key='status', subrules=[
        Rule2("afk", AFK_RULE_PRIORITY).skip(),
        Rule2("not-afk", 1).placeholder()
    ]),
    Rule2(BUCKET_STOPWATCH_PREFIX + ".*", 0, to_string="stopwatch").with_subrules(key='label', subrules=[
        Rule2(".*", 0).with_subrules(key='running', subrules=[
            Rule2("true", WATCHDOG_RULE_PRIORITY),
            # Even if it is the only event in interval it carries no activity.
            Rule2("false", 0).skip(),
        ])
    ]),
]


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
        self.duration += intervals_duration(rule_result.intervals)
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
    # TODO improve separation by activities.
    # Separate all "independent" activities undoubtedly.
    if current_rule.priority >= AFK_RULE_PRIORITY:
        return rule_result.description != window.description
    return False


class ProblemReporter:
    """
    Container to encapsulate logs in case of some edge situations during analysis.
    """

    @staticmethod
    def report_event_without_rule(rule, event):
        LOG.info("%s: Can't find handler for %s", rule, event)

    @staticmethod
    def report_missed_rule(cur_interval):
        LOG.info("Skipping %s because can't find rule for events in it:\n  %s",
                 cur_interval, "\n  ".join(str(x) for x in cur_interval.events))

    @staticmethod
    def report_placeholder_rule(rule, cur_interval):
        LOG.info("%s: Need to reveal rule for interval %s with %d events:\n  %s", rule,
                 cur_interval.to_str(only_time=True), len(cur_interval.events),
                 "\n  ".join(str(x) for x in cur_interval.events))

    @staticmethod
    def report_too_small_window(rule_result: RuleResult, window):
        LOG.info("%s: On handling %s got too small window %s", rule_result.rule, rule_result, window.to_str(True))

    @staticmethod
    def report_too_long_window(window, rule_result: RuleResult):
        LOG.info("%s: Got too long window %s after adding %s.", rule_result.rule, window.to_str(False), rule_result)

    SUPPORTED_ITEMS = {
        'unknown events occurencies': report_event_without_rule,
        'too small windows': report_too_small_window,
        'intervals without rules': report_missed_rule,
        'intervals need to reveal rule for': report_placeholder_rule,
        'intervals matched by too wide rules': report_too_long_window,
    }


def find_rule_for_event(event: Event, rules: List[Rule2]) -> Tuple[Rule2, str]:
    """
    Finds rule for the given event.
    :param event: Event to find handler for.
    :param rules: List of rules to search in.
    :return: Tuple with rule which more precisely matches event in the graph
    and description of matched part if rule was found.
    """
    for rule in rules:
        matched_rule, description = rule.find_rule_for_event(event)
        if matched_rule:
            return matched_rule, description
    return None, None


def find_rule_for_interval(interval: Interval, metrics: Metrics, rules: List[Rule2]) -> List[RuleResult]:
    """
    Searches rules describing all events in given interval.
    :param interval: `Interval` search rules for events of.
    :param metrics: `Metrics` instance to track metrics.
    :param rules: Graph of rules to search in.
    :return: List of `RuleResult`-s describing events in given interval.
    """
    rule_results = []
    # Iterate all events to find out one with higher priority.
    for event in interval.events:
        rule, description = find_rule_for_event(event, rules)
        if not rule:
            metrics.increment_and_call_handler('unknown events occurencies', interval, rule, event)
            continue  # It is OK, some events (inside interval) may don't have handlers intentionally.
        rule_results.append(RuleResult(rule, event, description, [interval]))
    return rule_results


def _window_to_activity(window: RuleResultsWindow, rule_result: RuleResult, activities: List[Activity],
                        is_build_debug_buckets: bool, activity_debug_events: List[Event], round_to: float,
                        metrics: Metrics):
    if window.duration < round_to:
        metrics.increment_and_call_handler('too small windows', None, rule_result, window)
        # Here may be a reason in "too specific" rules which leads to too small activities.
        # But current algorithm doesn't support few sliding windows in parallel
        # while person actually may switch too often between different activities.
        # TODO support few sliding windows.
    activity = window.to_activity()
    activities.append(activity)
    if is_build_debug_buckets:
        debug_event = Event(BUCKET_DEBUG_ACTIVITES, activity.start_time, activity.duration,
                            {
                                'description': rule_result.description,
                                'rule_results_count': len(activity.rule_results),
                            })
        activity_debug_events.append(debug_event)

def analyze_intervals(interval: Interval, round_to: float, custom_rules: List[Rule2],
                      is_build_debug_buckets: bool = False, ignore_hints: Set[str]= None) -> AnalyzerResult:
    """
    Analyzes linked list of 'Interval'-s to convert them into list of 'Activity'-es and provide explanation about
    how well it was done.
    :param interval: Linked list of 'Interval'-s to analyze.
    :param round_to: Both minimal summary interval length to show and step to align reporting intervals to.
    :param custom_rules: User-specific/crafted map of `Rule2`-s.
    :param is_build_debug_buckets: Flag to build debug buckets.
    :param ignore_hints: List of problems to disable in logs.
    :return: Tuple of:
    1 - List of assembled `Activity`-es.
    2 - `Counter` of intervals-by-rule description-s to sum of their durations.
        Like a `Activity`-es built by naive "equal" strategy, i.e. report with too many and too small activities.
    3 - Map of metrics to estimate report quality/coverage and improve rules.
        Key is name of metric, value is tuple [number_of_intervals, sum_of_durations].
    """
    # Assemble full set of rules from predifined ones and custom.
    rules = custom_rules + DEFAULT_RULES
    # Prepare to loop through intervals with searching rules, building report and metrics.
    # Go to the first interval.
    cur_interval: Interval = interval.iterate_prev()
    # Make containers for outputs.
    rules_counter = collections.Counter()
    activities: List[Activity] = []
    metrics = Metrics(ProblemReporter.SUPPORTED_ITEMS, ignore_hints)
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
        rule_results: List[RuleResult] = find_rule_for_interval(cur_interval, metrics, rules)
        # Gather some important metrics at start.
        is_not_afk = any(x for x in cur_interval.events \
                         if x.bucket_id.startswith(BUCKET_AFK_PREFIX) and x.data['status'] != 'afk')
        if is_not_afk:
            metrics.increment('not afk intervals', cur_interval)
        # NOTE: order is very important below.
        # Decide whether to count this interval or not and update metrics.
        if not rule_results:
            metrics.increment_and_call_handler('intervals without rules', cur_interval, cur_interval)
            continue
        # Find the only rule representing interval - by priority
        rule_result = max((x for x in rule_results), key=attrgetter('rule.priority'))
        # Update per-rule-name metric. It should include all rules (i.e. "skip", "placeholder", etc.).
        metrics.increment(str(rule_result.rule), cur_interval)
        # Fill up BUCKET_DEBUG_RULE_RESULTS before handling deferred intervals
        # to represent logic of choosing rule by events in interval.
        if is_build_debug_buckets:
            debug_event = Event(BUCKET_DEBUG_RAW_RULE_RESULTS, cur_interval.start_time, cur_interval.get_duration(),
                                {
                                    'description': rule_result.description,
                                    'rule': str(rule_result.rule),
                                    'events_cnt': len(cur_interval.events),
                                })
            raw_rule_result_debug_events.append(debug_event)
        duration = intervals_duration(rule_result.intervals)
        # Append deferred intervals if there are such.
        if deferred_intervals:
            rule_result.intervals += deferred_intervals
            metrics.increment('intervals merged to next rule', cur_interval)
            deferred_intervals = []
        # Check if rule says skip interval from the report.
        if rule_result.rule.is_skip:
            LOG.debug("Skipping %f sec %d interval(s) because of %s priority is highest for %s.",
                      round(duration, 1), len(rule_result.intervals), rule_result.rule, rule_result.event)
            metrics.increment('intervals with rule to skip', cur_interval)
            continue
        # Check if rule is a placeholder and provide all information about interval to write appropriate rule for it.
        if rule_result.rule.is_placeholder:
            metrics.increment_and_call_handler('intervals need to reveal rule for', cur_interval,
                                               rule_result.rule, cur_interval)
        # Update 'intervals to build activities from' metric.
        metrics.increment('intervals to build activities from', cur_interval)
        # Defer interval if need. Note that `activity_counter` shouldn't be touched by this rule.
        if rule_result.rule.is_merge_next:
            deferred_intervals.append(cur_interval)
            continue
        # Update 'rules counter'.
        rules_counter[rule_result.description] += duration
        if is_build_debug_buckets:
            debug_event = Event(BUCKET_DEBUG_FINAL_RULE_RESULTS, rule_result.intervals[0].start_time,
                                rules_counter[rule_result.description], {
                                    'description': rule_result.description,
                                    'rule': str(rule_result.rule),
                                    'intervals_count': len(rule_result.intervals),
                                })
            final_rule_result_debug_events.append(debug_event)
        # Decide if current window is completed, may be converted to `Activity` and next window started.
        is_start_new_window = True  # By default start new window.
        if window is not None:
            # Decide if current `RuleResult` is separate activity from previous ones and need to create `Activity` from
            # items accumulated in `activity_window` so far.
            if _is_new_activity(window, rule_result):
                _window_to_activity(window, rule_result, activities, is_build_debug_buckets, activity_debug_events,
                                    round_to, metrics)
                window = None
            else:
                window.append(rule_result)
                if window.duration > TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS:
                    metrics.increment_and_call_handler('intervals matched by too wide rules', cur_interval,
                                                       window, rule_result)
                is_start_new_window = False
        if is_start_new_window:
            window = RuleResultsWindow([rule_result], rule_result.rule.priority, rule_result.description, duration)
    # Handle case when last interval in "merge next".
    if deferred_intervals:
        metrics.increment('intervals merged to next rule', cur_interval)
        if window is not None:
            window.append(rule_result)
        else:
            window = RuleResultsWindow([rule_result], rule_result.rule.priority, rule_result.description, duration)
    # Handle last window.
    if window is not None:
        _window_to_activity(window, rule_result, activities, is_build_debug_buckets, activity_debug_events,
                            round_to, metrics)
    return AnalyzerResult(
        activities,
        rules_counter,
        metrics,
        raw_rule_result_debug_events,
        final_rule_result_debug_events,
        activity_debug_events
    )
