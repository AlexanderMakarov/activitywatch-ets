import dataclasses
import collections
from typing import List, Dict, Tuple

from ..config.config import LOG, AFK_RULE_PRIORITY, WATCHDOG_RULE_PRIORITY
from .interval import Interval
from .input_entities import EventKeyHandler, Rule
from .output_entities import RuleResult, Activity
from ..helpers.helpers import seconds_to_int_timedelta, event_data_to_str


def _increment_metric(metrics: Dict[str, Tuple[int, float]], metric_name: str, interval: Interval):
    metric = metrics.get(metric_name, (0, 0))
    metrics[metric_name] = (metric[0] + 1, metric[1] + interval.get_duration())


def _intervals_duration(intervals: List[Interval]):
    # Note that 'last end minus first start' doesn't work due to possible gaps between intervals.
    return sum(x.get_duration() for x in intervals)


def _find_out_rule_for_interval(
        interval: Interval,
        metrics: Dict[str, Tuple[int, float]],
        bucket_prefix_to_ruleshandler: Dict[str, List[EventKeyHandler]]
    ) -> RuleResult:
    rule_result: RuleResult = None
    # Iterate all events to 
    for event in interval.events:
        # Search `EventKeyHandler` by 2 criteria:
        # 1) event bucket ID starts with handler 'bucket_id',
        # 2) handler key exists in event data.
        handler: EventKeyHandler = None
        for bucket_prefix, bucket_handlers in bucket_prefix_to_ruleshandler.items():
            if event.bucket_id.startswith(bucket_prefix):
                for bucket_handler in bucket_handlers:
                    if bucket_handler.key in event.data:
                        handler = bucket_handler
                        break
        if not handler:
            LOG.info(f"Can't find handler for {event}")
            _increment_metric(metrics, 'events without handlers', interval)
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
        repr = f"{seconds_to_int_timedelta(self.duration)} with priority={self.priority}"
        if debug:
            events = dict((x.event.timestamp, x.event) for x in self.rule_results)
            events_str = "\n  ".join(
                f"{seconds_to_int_timedelta(x.duration.total_seconds())} {event_data_to_str(x)}"
                for x in events.values()
            )
            return f"{repr}, description='{self.description}' and {len(events)} events:" + "\n  " + events_str
        else:
            return f"{repr} and {len(self.rule_results)} rule results"

    def append(self, rule_result: RuleResult):
        self.rule_results.append(rule_result)
        self.duration += _intervals_duration(rule_result.intervals)
        # If new rule has more priority than all existing in window then update windows priority and description.
        if rule_result.rule.priority >= self.priority:
            self.priority = rule_result.rule.priority
            self.description = rule_result.description

    def to_activity(self) -> Activity:
        start_time = self.rule_results[0].intervals[0].start_time
        end_time = self.rule_results[-1].intervals[-1].end_time
        tmp = set(x.description for x in self.rule_results)
        description = ", ".join(sorted(tmp))
        return Activity(start_time, end_time, list(self.rule_results), description, self.duration)


def _is_new_activity(window: RuleResultsWindow, rule_result: RuleResult, round_to: float) -> bool:
    current_rule = rule_result.rule
    # Don't separate activities for rules with the same priority or same description.
    if current_rule.priority == window.priority and rule_result.description != window.description:
        return False
    # Separate all "indipendent" activities as soon as they appear.
    if current_rule.priority >= AFK_RULE_PRIORITY:
        return rule_result.description != window.description
    # TODO Apply `round_to` with distributing small buckets into wider ones.
    return False


def analyze_intervals(interval: Interval, round_to: float, custom_rules: Dict[str, List[EventKeyHandler]]
        ) -> Tuple[List[Activity], collections.Counter, Dict[str, Tuple[int, float]]]:
    """
    :param interval: Linked list of intervals to analyze.
    :param round_to: Both minimal summary interval length to show and step to align reporting intervals to.
    :param rules: User-specific map of event bucket name prefix to list of `EventKeyHandler`-s to handle data intervals.
    :return: Tuple of:
    1 - List of assembled `Activity`-es.
    2 - `Counter` of interval description-s to theirs durations. Aka `Activity`-es built by naive "equal" strategy.
    3 - Map of metrics to estimate report quality/coverage.
    """
    # Assemble full set of EventKeyHandler-s from predifined ones and custom.
    bucket_prefix_to_ruleshandler: Dict[str, List[EventKeyHandler]] = dict(custom_rules)
    if "aw-watcher-afk" not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler["aw-watcher-afk"] = [EventKeyHandler('status', [
            Rule("afk", AFK_RULE_PRIORITY, skip=True),
            Rule("not-afk", 1, is_placeholder=True)
        ])]
    if "aw-stopwatch" not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler["aw-stopwatch"] = [EventKeyHandler('label', [
            Rule(".*", 0, subhandler=EventKeyHandler('running', [
                Rule("true", WATCHDOG_RULE_PRIORITY),
                Rule("false", 0, skip=True)  # Even if it is the only event in interval it carries no activity.
            ]))
        ])]
    # Prepare to loop through intervals with searching rules, building report and metrics.
    # Go to the first interval.
    cur_interval: Interval = interval.iterate_prev()
    # Make containers for outputs.
    rules_counter = collections.Counter()
    activities: List[Activity] = []
    metrics = {  # Put default metrics. Next it will be appended by per-rule metrics.
        'total intervals': (0, 0),
        'events without handlers': (0, 0),
        'intervals without rules': (0, 0),
        'intervals merged to next rule': (0, 0),
        'intervals with rule to skip': (0, 0),
        'intervals need to reveal rule for': (0, 0),
    }
    # Prepare dummy `Interval`` to use first interval on the very first iteration below.
    cur_interval = Interval(cur_interval.start_time, cur_interval.end_time, None, cur_interval)
    # Iterate all intervals, find rules for all events in it and choose highest by prioirty to handle interval to
    # build slices, activity_counter, metrics.
    deferred_intervals: List[Interval] = []  # `Interval`-s deffered as "append to next independent rule".
    window: RuleResultsWindow = None
    while cur_interval.next:
        cur_interval = cur_interval.next
        rule_result = _find_out_rule_for_interval(cur_interval, metrics, bucket_prefix_to_ruleshandler)
        # Decide whether to count this interval or not and update metrics.
        if rule_result is None:
            LOG.info(f"Skipping {cur_interval} because it doesn't contain events matching any rule."
                     + " Events:\n  " + "\n  ".join(str(x) for x in cur_interval.events))
            _increment_metric(metrics, 'intervals without rules', cur_interval)
            continue
        # Update per-rule-name metric. It should include all rules (i.e. "skip", "placeholder", etc.).
        _increment_metric(metrics, str(rule_result.rule), cur_interval)
        duration = _intervals_duration(rule_result.intervals)
        # Append deferred intervals if there are such.
        if deferred_intervals is not None:
            rule_result.intervals += deferred_intervals
            _increment_metric(metrics, 'intervals merged to next rule', cur_interval)
            deferred_intervals = []
        # Check if rule says skip interval from the report.
        if rule_result.rule.skip:
            LOG.debug(f"Skipping {duration:.1f} sec {len(rule_result.intervals)}"
                      f" intervals(s) because of {rule_result.rule} priority is highest"
                      f" for {rule_result.event}.")
            _increment_metric(metrics, 'intervals with rule to skip', cur_interval)
            continue
        # Check if rule is a placeholder and provide all information about interval to write appropriate rule for it. 
        if rule_result.rule.is_placeholder:
            LOG.info("Need to reveal rule for interval %s with %d events:\n  %s"
                     % (cur_interval.to_str(only_time=True), len(cur_interval.events),
                        "\n  ".join(str(x) for x in cur_interval.events))
                    )
            _increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
        # is_fresh = cur_interval.prev is None or cur_interval.prev.end_time != cur_interval.start_time
        # Update 'total_intervals' metric.
        _increment_metric(metrics, 'total intervals', cur_interval)
        # Defer interval if need. Note that `activity_counter` shouldn't be touched by this rule.
        if rule_result.rule.merge_next:
            deferred_intervals.append(cur_interval)
            continue
        # Update 'rules counter'.
        rules_counter[rule_result.description] += duration
        # -----------
        is_start_new_window = True  # By default start new window.
        if window is not None:
            # Decide if current `RuleResult` is separate activity from previous ones and need to create `Activity` from
            # items accumulated in `activity_window` so far.
            if _is_new_activity(window, rule_result, round_to):
                if window.duration < round_to:
                    LOG.info(f"On handling {rule_result} separated too small window {window.to_str(True)}")
                    # TODO _increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
                activities.append(window.to_activity())
            else:
                window.append(rule_result)
                # if window.duration > TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS:
                #     LOG.info(f"Too long window {window.to_str(False)} after checking {rule_result}.")
                #     # TODO _increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
                is_start_new_window = False
        if is_start_new_window:
            window = RuleResultsWindow([rule_result], rule_result.rule.priority, rule_result.description, duration)
    return activities, rules_counter, metrics
