import collections
import dataclasses
import datetime
from operator import attrgetter
from typing import Dict, List, Optional, Set, Tuple

import intervaltree

from ..config.config import (AFK_RULE_PRIORITY, CURRENT_TIMEZONE,
                             DEBUG_BUCKET_PREFIX, LOG, MIN_DURATION_SEC,
                             TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS,
                             WATCHDOG_RULE_PRIORITY)
from ..helpers.helpers import event_data_to_str, seconds_to_int_timedelta
from .input_entities import ActivityBoundaries, Event, Rule2
from .interval import Interval, intervals_duration
from .metrics import Metrics
from .output_entities import Activity, AnalyzerResult, RuleResult
from .strategies import ActivitiesByStrategy, ActivityByStrategy

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
        return Activity(start_time, end_time, list(self.rule_results), description)


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
        debug_event = Event(DEBUG_BUCKET_PREFIX + "3_activities", activity.start_time, activity.duration,
                            {
                                'description': rule_result.description,
                                'events_count': len(activity.events),
                            })
        activity_debug_events.append(debug_event)

def analyze_intervals(interval: Interval, round_to: float, custom_rules: List[Rule2],
                      is_build_debug_buckets: bool = False) -> AnalyzerResult:
    """
    Analyzes linked list of 'Interval'-s to convert them into list of 'Activity'-es and provide explanation about
    how well it was done.
    :param interval: Linked list of 'Interval'-s to analyze.
    :param round_to: Both minimal summary interval length to show and step to align reporting intervals to.
    :param custom_rules: User-specific/crafted map of `Rule2`-s.
    :param is_build_debug_buckets: Flag to build debug buckets.
    :return: Tuple of:
    1 - List of assembled `Activity`-es.
    2 - `Counter` of intervals-by-rule description-s to sum of their durations.
        Like a `Activity`-es built by naive "equal" strategy, i.e. report with too many and too small activities.
    3 - Map of metrics to estimate report quality/coverage and improve rules.
        Key is name of metric, value is tuple [number_of_intervals, sum_of_durations].
    """

    # Options to loop through events:
    # 1. Try to search activities in each bucket separately. Next merge.
    #    - If edges don't match then need somehow adjust to cover gaps or cut activities.
    #     => Use priorities between buckets/rules to cut.
    #    - How to solve "few activities from bucketA during activity from bucketB"?
    #     => Make biggest activity if it fits in frame, otherwise use priorities between buckets/rules.
    #    - Need to specify order of buckets handling and specify priority of merging.
    #     => Order of handling in the STRATEGIES, priorities may be found by 'out*' parameters - depending on way of merging.
    # 2. Analyze all at once:
    #    - It is similar to option with Intervals list -> analyze with rules.
    #    - Hard to specify priorities. Quite high by priority events from different bucket may interrupt each other for the same activity.
    #    - Hard to find name for activity.
    BUCKET_DEBUG_RAW_RULE_RESULTS = DEBUG_BUCKET_PREFIX + "1_raw_rule_results"
    BUCKET_DEBUG_FINAL_RULE_RESULTS = DEBUG_BUCKET_PREFIX + "2_final_rule_results"
    BUCKET_DEBUG_ACTIVITIES = DEBUG_BUCKET_PREFIX + "3_activities"

    # Assemble full set of rules from predifined ones and custom.
    rules = custom_rules + DEFAULT_RULES
    # Prepare to loop through intervals with searching rules, building report and metrics.
    # Go to the first interval.
    cur_interval: Interval = interval.iterate_prev()
    # Make containers for outputs.
    rules_counter = collections.Counter()
    activities: List[Activity] = []
    metrics = Metrics(ProblemReporter.SUPPORTED_ITEMS)
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
        {
            BUCKET_DEBUG_RAW_RULE_RESULTS: raw_rule_result_debug_events,
            BUCKET_DEBUG_FINAL_RULE_RESULTS: final_rule_result_debug_events,
            BUCKET_DEBUG_ACTIVITIES: activity_debug_events,
        }
    )

# TODO remove above.


def _cut_activity_start(activity: ActivityByStrategy, point: datetime.datetime) -> ActivityByStrategy:
    """
    Cuts start from activity (suggested and 'max_start_time' if need) and returns only tail after specified point.
    """
    events = [x for x in activity.events if (x.timestamp + x.duration) > point]  # Keep "border" event.
    return ActivityByStrategy(
        suggested_start_time=point,
        suggested_end_time=activity.suggested_end_time,
        max_start_time=max(point, activity.max_start_time),
        min_end_time=activity.min_end_time,
        duration=sum(x.duration.seconds for x in events),
        events=events,
        grouping_data=activity.grouping_data,
        strategy=activity.strategy,
    )


def _cut_activity_end(activity: ActivityByStrategy, point: datetime.datetime) -> ActivityByStrategy:
    """
    Cuts end from activity (suggested and 'min_end_time' if need) and returns only head before specified point.
    """
    events = [x for x in activity.events if x.timestamp < point]  # Keep "border" event.
    return ActivityByStrategy(
        suggested_start_time=activity.suggested_start_time,
        suggested_end_time=point,
        max_start_time=activity.max_start_time,
        min_end_time=min(point, activity.min_end_time),
        duration=sum(x.duration.seconds for x in events),
        events=events,
        grouping_data=activity.grouping_data,
        strategy=activity.strategy,
    )


def _split_activity(activity: ActivityByStrategy, start_point: datetime.datetime, end_point: datetime.datetime)\
        -> List[ActivityByStrategy]:
    """
    Cuts some middle part from the activity to get 2 new 'ActivityByStrategy'-s:
    - from the start of initial activity to the 'start_point',
    - from the 'end_point' to the end of initial activity.
    """
    return [_cut_activity_end(activity, start_point), _cut_activity_start(activity, end_point)]


def _exclude_tree_intervals(activities: List[ActivityByStrategy], tree: intervaltree.IntervalTree, metrics: Metrics,
                            name_of_tree: str) -> List[ActivityByStrategy]:
    """
    Cuts different parts of activities which overlap with intervals in the specified tree.
    Note that for metrics are used bounds of activities, not inner event intervals sum.
    :param activities: List of activities to cut tree intervals from.
    :param tree: Tree with intervals to cut activities by.
    :param metrics: Metrics instance.
    :param name_of_tree: Name of the tree for metrics.
    :return: List of chopped activities.
    """
    result: List[ActivityByStrategy] = []

    # Iterate until there are "not checked" activities in the queue.
    activities_queue: List[ActivityByStrategy] = list(activities)
    while len(activities_queue) > 0:
        activity = activities_queue.pop()
        # Search all intervals, there is no "first_overlap" method. Completes in O(m + k*log n) time, where:
        # n = size of the tree
        # m = number of matches
        # k = size of the search range ?
        intervals: List[intervaltree.Interval] = tree.overlap(activity.suggested_start_time,
                                                              activity.suggested_end_time)
        # If current activity doesn't overlap with tree then put it into the result and proceed with the queue.
        if len(intervals) == 0:
            result.append(activity)
            continue
        # Handle only first (and random). Because in result activity would be either skipped or modified.
        # In "modified" case need to search overlaps again anyway.
        interval = intervals.pop()
        if interval.begin <= activity.suggested_start_time:
            # Interval starts before activity.
            # Check activity is covered by interval completely.
            if interval.end >= activity.suggested_end_time:
                metrics.incr('activities removed by ' + name_of_tree,
                             (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
                continue
            # Check (second time actually) interval overlaps activity.
            if interval.end < activity.suggested_end_time:
                prev_start_time = activity.suggested_start_time
                tmp = _cut_activity_start(activity, interval.end)
                activities_queue.append(tmp)
                metrics.incr('activities with head cut by ' + name_of_tree,
                             (tmp.suggested_start_time - prev_start_time).total_seconds())
                continue
            else:
                raise NotImplementedError(f'Inner error with {interval} actually not overlapping with'
                                          f' {activity}.')
        elif interval.end >= activity.suggested_end_time:
            # Interval ends after activity.
            # Check (second time actually) activity end is overlapped by interval.
            if interval.begin > activity.suggested_start_time:
                prev_end_time = activity.suggested_end_time
                tmp = _cut_activity_end(activity, interval.begin)
                activities_queue.append(tmp)  # Due to interval was random, activity need to check one more time.
                metrics.incr('activities with tail cut by ' + name_of_tree,
                             (prev_end_time - tmp.suggested_end_time).total_seconds())
            else:
                raise NotImplementedError(f'Inner error with {interval} actually not overlapping with'
                                          f' {activity}.')
        else:
            # Interval is placed in the middle of the current activity.
            split_activities = _split_activity(activity, interval.begin, interval.end)
            # Due to interval was random, both parts of the initial activity should be checked again.
            activities_queue.extend(split_activities)
            # Update metric without duration because very often activity is cut few times and resulting
            # cumulative duration is bigger than initial activity duration.
            metrics.incr('activities with middle cut by ' + name_of_tree)
            continue
    # Sort result because parts of long activities may be placed into it randomly.
    return sorted(result, key=lambda x: x.suggested_start_time)


def _include_tree_intervals(activities: List[ActivityByStrategy], boundaries: ActivityBoundaries,
                            tree: intervaltree.IntervalTree, metrics: Metrics, name_of_tree: str)\
                                -> List[ActivityByStrategy]:
    """
    Cuts different parts of activities which doesn't overlap with intervals in the specified tree.
    Note that for metrics are used bounds of activities, not inner event intervals sum.
    :param activities: List of activities to fit into the tree.
    :param boundaries: Boundaries of activity.
    :param tree: Tree with intervals to cut activities by.
    :param metrics: Metrics instance.
    :param name_of_tree: Name of the tree for metrics.
    :return: List of modified activities.
    """
    result: List[ActivityByStrategy] = []
    for activity in activities:
        # Search all intervals, there is no "first_overlap" method. Completes in O(m + k*log n) time, where:
        # n = size of the tree
        # m = number of matches
        # k = size of the search range ?
        overlapping_intervals: List[intervaltree.Interval] = tree.overlap(
            activity.suggested_start_time,
            activity.suggested_end_time
        )

        # Check that activity is not overlapping at all.
        if not overlapping_intervals:
            metrics.incr(f'activities removed because are out of {name_of_tree}',
                         (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
            continue

        # Build continuous overlapping segments to chop activity by them. Sort segments.
        segments = []
        for interval in sorted(overlapping_intervals, key=lambda x: x.begin):
            # If segments list is empty or current interval doesn't overlap with the last segment.
            if not segments or segments[-1][1] < interval.begin:
                segments.append([interval.begin, interval.end])
            else:
                # Merge overlapping segments.
                segments[-1][1] = max(segments[-1][1], interval.end)

        # Determine if the activity's start and end times are covered.
        start_covered = any(seg[0] <= activity.suggested_start_time <= seg[1] for seg in segments)
        end_covered = any(seg[0] <= activity.suggested_end_time <= seg[1] for seg in segments)
        gaps_in_middle = len(segments) > 1

        # Check that activty is fully covered. The only possibility to keep ActivityBoundaries.STRICT.
        if start_covered and end_covered and not gaps_in_middle:
            metrics.incr(f'activities completely covered by {name_of_tree}',
                         (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
            result.append(activity)
            continue

        # Check remained ActivityBoundaries cases.
        if boundaries == ActivityBoundaries.START:
            if not start_covered:
                metrics.incr(f'activities removed with {boundaries} started before {name_of_tree}',
                             (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
                continue
            if segments[0][1] < activity.max_start_time:
                metrics.incr(f'activities removed with {boundaries} end out of {name_of_tree}',
                             (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
                continue
            prev_end_time = activity.suggested_end_time
            activity = _cut_activity_end(activity, segments[0][1])
            result.append(activity)
            metrics.incr(f'activities with tail cut by {name_of_tree}',
                         (prev_end_time - activity.suggested_end_time).total_seconds())
        elif boundaries == ActivityBoundaries.END:
            if not end_covered:
                metrics.incr(f'activities removed with {boundaries} ended before {name_of_tree}',
                             (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
                continue
            if segments[-1][0] > activity.min_end_time:
                metrics.incr(f'activities removed with {boundaries} start out of {name_of_tree}',
                             (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
                continue
            prev_start_time = activity.suggested_start_time
            activity = _cut_activity_start(activity, segments[-1][0])
            result.append(activity)
            metrics.incr(f'activities with start cut by {name_of_tree}',
                         (activity.suggested_start_time - prev_start_time).total_seconds())
            continue
        elif boundaries == ActivityBoundaries.DIM:
            for segment in segments:
                # Chop some segment from the activity. Doesn't check min and max time points of activity.
                tmp = activity
                is_chopped = False
                prev_duration = tmp.suggested_end_time - tmp.suggested_start_time
                if segment[0] > tmp.suggested_start_time:
                    tmp = _cut_activity_start(tmp, segments[-1][0])
                    is_chopped = True
                if segment[1] < tmp.suggested_end_time:
                    tmp = _cut_activity_end(tmp, segments[0][1])
                    is_chopped = True
                result.append(tmp)
                if is_chopped:
                    # Update metric without duration because very often activity is cut few times and resulting
                    # cumulative duration is bigger than initial activity duration.
                    metrics.incr(f'activities with {boundaries} cut by {name_of_tree}')
                else:
                    metrics.incr(f'activities completely covered by {name_of_tree}', prev_duration.total_seconds())
                activity = tmp
        else:
            metrics.incr(f'activities with {boundaries} removed by {name_of_tree}',
                         (activity.suggested_end_time - activity.suggested_start_time).total_seconds())
    return result

    # Iterate until there are "not checked" activities in the queue.
    # activities_queue: List[ActivityByStrategy] = list(activities)
    # while len(activities_queue) > 0:
    #     activity = activities_queue.pop()
    #     # Search all intervals, there is no "first_overlap" method. Completes in O(m + k*log n) time, where:
    #     # n = size of the tree
    #     # m = number of matches
    #     # k = size of the search range ?
    #     intervals: List[intervaltree.Interval] = tree.overlap(activity.suggested_start_time,
    #                                                           activity.suggested_end_time)
    #     # If current activity doesn't overlap with tree then remove it at all.
    #     if len(intervals) == 0:
    #         metrics.incr(f'activities removed because are out of {name_of_tree}', activity.duration)
    #         continue
    #     # Need to check that the whole activity interval is covered by intervals in the tree.
    #     # If some parts are not covered then depending on boundaries need to cut some parts of it.
        
    #     todo
    #     # Handle only first random interval.
    #     interval = intervals.pop()
    #     if interval.begin <= activity.suggested_start_time:
    #         # Interval starts before activity.
    #         # Check activity is covered by interval completely.
    #         if interval.end >= activity.suggested_end_time:
    #             metrics.incr('activities removed by ' + name_of_tree, activity.duration)
    #             continue
    #         # Check activity can't be chopped at start.
    #         if boundaries is not ActivityBoundaries.END:
    #             metrics.incr('activities removed because impossible to cut on start by ' + name_of_tree,
    #                          activity.duration)
    #             continue
    #         # Check (second time actually) interval overlaps activity.
    #         if interval.end < activity.suggested_end_time:  # TODO: check how much
    #             prev_duration = activity.duration
    #             tmp = _cut_activity_start(activity, interval.end)
    #             activities_queue.append(tmp)
    #             metrics.incr('activities cut on start by ' + name_of_tree, prev_duration - tmp.duration)
    #             continue
    #         else:
    #             raise NotImplementedError(f'Inner error with {interval} actually not overlapping with'
    #                                       f' {activity}.')
    #     elif interval.end >= activity.suggested_end_time:
    #         # Interval ends after activity.
    #         # Check activity can be chopped at the end.
    #         if boundaries is not ActivityBoundaries.START:
    #             metrics.incr('activities removed because impossible to cut on end by ' + name_of_tree,
    #                          activity.duration)
    #             continue
    #         # Check (second time actually) activity end is overlapped by interval.
    #         if interval.begin > activity.suggested_start_time:  # TODO: check how much
    #             prev_duration = activity.duration
    #             tmp = _cut_activity_end(activity, interval.begin)
    #             activities_queue.append(tmp)  # Due to interval was random, activity need to check one more time.
    #             metrics.incr('activities cut on end by '+ name_of_tree, prev_duration - tmp.duration)
    #         else:
    #             raise NotImplementedError(f'Inner error with {interval} actually not overlapping with'
    #                                       f' {activity}.')
    #     else:
    #         # Interval is placed in the middle of the current activity.
    #         prev_duration = activity.duration
    #         if may_cut_point[1] and may_cut_point[0]:  # TODO: maybe remove this?
    #             split_activities = _split_activity(activity, interval.begin, interval.end)
    #             # Due to interval was random, both parts of the initial activity should be checked again.
    #             activities_queue.extend(split_activities)
    #             # Note that duration is measured by events.
    #             metrics.incr('activities with cut out middle by ' + name_of_tree,
    #                             prev_duration - (interval.end - interval.begin).total_seconds())
    #             continue
    #         elif boundaries is ActivityBoundaries.START:
    #             tmp = _cut_activity_end(activity, interval.begin)  # TODO: check how much
    #             activities_queue.append(tmp)
    #             metrics.incr('activities cut on end by ' + name_of_tree, prev_duration - tmp.duration)
    #             continue
    #         elif boundaries is ActivityBoundaries.END:  # TODO: check how much
    #             tmp = _cut_activity_start(activity, interval.end)
    #             activities_queue.append(tmp)
    #             metrics.incr('activities cut on start by ' + name_of_tree, prev_duration - tmp.duration)
    #             continue
    #         else:
    #             raise NotImplementedError('Inner error when impossible to cut both start and end of activity'
    #                                       f' {activity} by {interval}')
    # Sort result because parts of long activities may be placed into it randomly.
    # return sorted(result, key=lambda x: x.suggested_start_time)


def _add_debug_event(debug_dict: Dict[str, List[Event]], bucket_id: str, timestamp: datetime.datetime,
                     duration: datetime.timedelta, description: str, events_count: int):
    data = {'desc': description, 'events_count': str(events_count)}  # Note that ActivityWatch UI shows only strings.
    debug_dict.setdefault(bucket_id, []).append(Event(bucket_id, timestamp, duration, data))


def _add_debug_events_to_not_overlap(activites: List[ActivityByStrategy], debug_dict: Dict,
                                     bucket_name: str, debug_buckets_cnt: int, metrics: Metrics) -> int:
    """
    Adds debug events to the given dictionary in few buckets where events are not overlapping.
    Uses "suggested" time boundaries, not "minimal" ones.
    :param activites: Activities to add debug events from.
    :param debug_dict: Dictionary to add debug events to.
    :param bucket_name: Part of name for debug buckets.
    :param metrics: Metrics instance to report progress.
    :param debug_buckets_cnt: Counter of debug buckets.
    :return: Updated counter of debug buckets.
    """
    activites = sorted(activites, key=lambda x: x.suggested_start_time)
    groups: List[List[ActivityByStrategy]] = []
    # Seaparate activities by groups.
    for activity in activites:
        found_group = False
        for group in groups:
            is_overlapping = False
            # Iterate all groups each time to find place for the new activity.
            # Need to pack events as dense as possible - activities may overlap.
            for existing_activity in group:
                # Check activities overlap.
                if activity.suggested_end_time >= existing_activity.suggested_start_time \
                        and activity.suggested_start_time <= existing_activity.suggested_end_time:
                    is_overlapping = True
                    break
            if not is_overlapping:
                group.append(activity)
                found_group = True
                break
        if not found_group:
            groups.append([activity])
            metrics.incr(f'debug event groups for {bucket_name}.* strategy')
    # Fill buckets of events from groups.
    for group in groups:
        debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}{debug_buckets_cnt:03}_{bucket_name}"
        debug_buckets_cnt += 1
        for activity in group:
            _add_debug_event(
                debug_dict,
                debug_bucket_prefix,
                activity.suggested_start_time,
                # For debug events need to use end-start time, not duration by events.
                activity.suggested_end_time - activity.suggested_start_time,
                str(activity.grouping_data),
                len(activity.events),
            )
            metrics.incr('debug events in ' + debug_bucket_prefix, activity.duration)
    return debug_buckets_cnt


def _find_basic_activity_interval(candidates_tree: intervaltree.IntervalTree, current_start_point: datetime.datetime,
                                  metrics: Metrics) -> Optional[intervaltree.Interval]:
    """
    Tries to find good "basic activity" in the given candidates tree. Intervals in this tree are:
    - Chopped to don't overlap with "result activities" added so far.
    - Random.
    :returns: Best possible "basic activity" or `None` if there are no more candidates.
    """
    min_duraiton_timedelta = datetime.timedelta(seconds=MIN_DURATION_SEC)
    # Get accumulator of all activities started in interval [current_start_point, min_duraiton].
    # Note that candidates_tree in most cases contains intervals chopped by AFK and already existing
    # results activities.
    # But if there is a big gap, then here may be intervals started on the previous RA and lasting still.
    candidates_for_ba: Set[intervaltree.Interval] = candidates_tree.overlap(
        current_start_point, current_start_point + min_duraiton_timedelta
    )
    if not candidates_for_ba:
        # If nothing at right then stop iterating - no more actvities remained, we are done.
        return None
    # TODO enhance with max_start_time and min_end_time properties of ActivityByStrategy.
    # Try find BA as:
    # - started on the `current_start_point`,
    # - with `out_activity_boundaries` "whole" or "start",
    # - length is equal or more than `MIN_DURATION_SEC` (choose minimal).
    # For this first sort them by the start, next by length (shorter - first).
    candidates_for_ba = sorted(candidates_for_ba, key=lambda x: (x.begin, x.end - x.begin))
    ba_interval = next((x for x in candidates_for_ba
                        if x.begin == current_start_point
                        and x.length() >= min_duraiton_timedelta
                        and x.data.strategy.out_activity_boundaries in ['whole', 'start']),
                        None)
    # If there are no such activity then just make BA from the longest activity.
    # Note that all "remained" activities are cut by existing "result" activities and shouldn't overlap with them.
    if ba_interval is None:
        ba_interval = max(candidates_for_ba, key=intervaltree.Interval.length)
        LOG.info("Can't find 'perfect' basic activity after %s and longer than %s. Using as basic longest - %s.",
                    current_start_point.astimezone(CURRENT_TIMEZONE).strftime("%H:%M:%S"), min_duraiton_timedelta,
                    ba_interval.data)
        metrics.incr("basic activities assembled from remainings", ba_interval.length().total_seconds())
    else:
        LOG.info("Found 'perfect' basic activity after %s - %s.",
                 current_start_point.astimezone(CURRENT_TIMEZONE).strftime("%H:%M:%S"), ba_interval.data)
        metrics.incr("basic activities from solid interval", ba_interval.length().total_seconds())
    return ba_interval


def _build_result_activity(ba_interval: intervaltree.Interval, candidates_tree: intervaltree.IntervalTree,
                           metrics: Metrics) -> Activity:
    # Find all overlapping activities.
    overlapping_intervals = candidates_tree.overlap(ba_interval.begin, ba_interval.end)  # Includes BA.
    LOG.info("Basic activity is overlapped by %d 'candidate' activities.", len(overlapping_intervals))
    ra_events = ba_interval.data.events
    overlapping_activities: List[ActivityByStrategy] = [ba_interval.data]
    for interval in overlapping_intervals:
        if interval == ba_interval:
            continue
        activity = interval.data
        # If BA overlaps activity then just concatenate data from it into BA.
        if ba_interval.overlaps(interval):
            ra_events.extend(activity.events)
            overlapping_activities.append(activity)
            candidates_tree.remove(interval)
            metrics.incr('activities absorbed by basic activity completely', interval.length().total_seconds())
            continue
        boundaries = activity.strategy.out_activity_boundaries
        # Check BA overlaps the start of the activity.
        if ba_interval.contains_point(interval.begin):
            if boundaries == ActivityBoundaries.START:
                # If activity is `out_activity_boundaries=start` then concatenate data from it into BA.
                ra_events.extend(activity.events)
                overlapping_activities.append(activity)
                candidates_tree.remove(interval)
                metrics.incr('activities absorbed by basic activity at the start',
                                interval.length().total_seconds())
            elif boundaries == ActivityBoundaries.DIM:
                # If activity is `out_activity_boundaries=whole` then split activity,
                # and concatenate last part with BA. First part is not needed anyway.
                split_activity = _cut_activity_end(activity, ba_interval.end)
                ra_events.extend(split_activity.events)
                overlapping_activities.append(split_activity)
                candidates_tree.remove(interval)
                # Note that tail of the activity may be used for the next basic/candidate activity.
                candidates_tree.addi(ba_interval.end, split_activity.suggested_end_time, split_activity)
                metrics.incr('activities enhancing basic activity by the start', split_activity.duration)
            else:
                # If activity is `out_activity_boundaries=end` then skip it.
                metrics.incr('activities unable to enhance basic activity by the start',
                                interval.length().total_seconds())
            continue
        # Check BA overlaps the end of activity.
        if ba_interval.contains_point(interval.end):
            if boundaries == ActivityBoundaries.END:
                # If activity is `out_activity_boundaries=end` then concatenate data from it into BA.
                ra_events.extend(activity.events)
                overlapping_activities.append(activity)
                candidates_tree.remove(interval)
                metrics.incr('activities absorbed by basic activity at the end', interval.length().total_seconds())
            elif boundaries == ActivityBoundaries.DIM:
                # If activity is `out_activity_boundaries=whole` then split activity,
                # and concatenate first part with BA.
                split_activity = _cut_activity_start(activity, ba_interval.begin)
                ra_events.extend(split_activity.events)
                overlapping_activities.append(split_activity)
                candidates_tree.remove(interval)
                metrics.incr('activities enhancing basic activity by the end', split_activity.duration)
            else:
                # If activity is `out_activity_boundaries=start` then skip it.
                metrics.incr('activities unable to enhance basic activity by the end',
                                interval.length().total_seconds())
            continue
        # If BA itself is placed inside actvity then concatenate middle of it
        # and only if it is `out_activity_boundaries=whole`.
        if boundaries == ActivityBoundaries.DIM:
            tmp = _cut_activity_start(activity, ba_interval.begin)
            tmp = _cut_activity_end(tmp, ba_interval.end)
            ra_events.extend(tmp.events)
            overlapping_activities.append(tmp)
            candidates_tree.remove(interval)
            # Note that tail of the activity may be used for the next basic/candidate activity.
            candidates_tree.addi(ba_interval.end, tmp.suggested_end_time, tmp)
            metrics.incr('activities enhancing basic activity by the middle', ba_interval.length().total_seconds())
            continue
        # Skip all other cases.
        raise NotImplementedError(f'Inner error when {activity} marked as overlapping with basic activity'
                                    f'{ba_interval} but in reality it is not.')
    ra_duration = ba_interval.length().seconds
    # Find right description for the resulting activity.
    dominant_names = []
    other_names = []
    # TODO set name properly.
    for activity in overlapping_activities:
        if activity.strategy.out_activity_name == "alone":
            dominant_names.append(str(activity.grouping_data))
        else:
            other_names.append(str(activity.grouping_data))
    description = "; ".join(dominant_names)
    if len(description) < 1:
        metrics.incr("result activities without distinct name", ra_duration)
        description = "; ".join(other_names)
    else:
        metrics.incr(f"result activities with {len(dominant_names)} distinct names", ra_duration)

    # Build raw RA, i.e. with "not sure end". Fill it with all the events from the "enhancing" activities.
    metrics.incr("result activities", ra_duration)
    return Activity(
        ba_interval.begin,
        ba_interval.end,
        ra_events,
        description,
    )


def analyze_activities_per_strategy(activities_by_strategy: List[ActivitiesByStrategy],
                                    is_add_debug_buckets: bool = True) -> AnalyzerResult:
    # TODO don't decide activity for each interval - it means loosing information about next inetervals.
    # TODO need to analyze neighbors.
    # 1) If there was Zoom meeting 15 minutes and no meetings before and after then it probably was a meeting.
    #   If it matches Outlook event then it is a name for the meeting. "Window" should show Zoom or Slack.
    #   AFK is probably "afk" here due to idle.
    # 2) If a lot of IDEA activity during some interval then it is an active coding.
    #   It may be named by JIRA ticket aroung it. "Window" should show Zoom or Slack.
    #   AFK should be "not-afk" here due to active movements.
    # 3) If "window" shows "browser" and "browser" switches tabs then it is active work in browser.
    #   If it is the same tab in browser then it is probably some web app.
    # TODO need to make links between rules of different buckets.
    # Extra ideas:
    # - We may make clusters with fields of 'data'. Like the same "jira_id" for JIRA or 'project' for IDEA.
    # Strategies:
    # 1. Find "long" events and make "windows" basing on them. Even intersecting.
    # 2. Find many nearby events of the "same application". Make "windows" basing on them. Even intersecting.
    # 3. Separate buckets on "strategies":
    #   - Trustworthy, event=activity, sequential - like watchdog.
    #   - Reliable whole event, event=activity, sequential, need approve - like Outlook. Need approve from window (slack, zoom).
    #   - Reliable whole event, few events=activity, may mix activities, need name of activity - like VS code, window
    #   - Reliable only event start, may mix activities - like IDEA. Need approve from other buckets.
    #   - Reliable only event end, may mix activities - like JIRA. Need approve from window or browser.
    #   - Dependant, sequential - like AFK. Very bad source of activities.
    # Looks like ML problem....
    
    # Make activities in "sure that activity" order.
    # In this way "remained gaps" will shape activities for which data is unclear.
    metrics = Metrics({})
    debug_dict: Dict[str, List[Event]] = {}
    debug_buckets_cnt = 1

    # 1. Find AFK strategy. It is required for `out_only_not_afk` handling.
    LOG.info('Searching not-AFK intervals.')
    not_afk_tree = intervaltree.IntervalTree()
    for strategy_result in activities_by_strategy:
        bucket_prefix = strategy_result.strategy.bucket_prefix
        if not bucket_prefix.startswith(BUCKET_AFK_PREFIX):
            continue
        metrics.incr('AFK strategies')
        if strategy_result.strategy.in_activities_may_overlap:
            LOG.warning("Unsupported setup for %s* strategy - in_activities_may_overlap=True."
                        "Skipping any AFK-related logic populated by this strategy.", bucket_prefix)
            continue
        # Add to not_afk_tree only not-AFK activities. Expect that they are not intersect.
        for activity in strategy_result.activities:
            status = activity.grouping_data.values[0]  # grouping_data is WindowKey.
            if status == 'not-afk':
                not_afk_tree.addi(activity.suggested_start_time, activity.suggested_end_time, activity)
                metrics.incr('not-afk intervals', activity.duration)
            else:
                metrics.incr('afk intervals', activity.duration)

    # 2. Cut activities from `out_only_not_afk=True` strategies.
    # Note that some "should be in not-afk" events may be produced as started before AFK watcher started
    # (for example IDEA events when computer in hibernate mode).
    LOG.info('Chopping activities by AFK intervals.')
    for strategy_result in activities_by_strategy:
        if not strategy_result.strategy.out_only_not_afk:
            continue
        metrics.incr('strategies to cut by AFK')
        strategy_result.activities = _include_tree_intervals(
            strategy_result.activities, strategy_result.strategy.out_activity_boundaries, not_afk_tree, metrics, 'not-AFK'
        )

    # 3. Add into result activities from `out_self_sufficient=True` strategies. Check for overlappings.
    LOG.info('Adding "out_self_sufficient" strategies activities.')
    debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}{debug_buckets_cnt:03}_self_sufficient"
    result_tree = intervaltree.IntervalTree()
    for strategy_result in activities_by_strategy:
        if not strategy_result.strategy.out_self_sufficient:
            continue
        metrics.incr('self sufficient strategies')
        for activity in strategy_result.activities:
            overlap: List[intervaltree.Interval] = result_tree.overlap(activity.start_time, activity.end_time)
            if overlap:
                raise ValueError(f"Overlapping activities from {strategy_result.strategy}: "
                                 f"{activity} is overlapping with " + ", ".join(x.data for x in overlap))
            result_tree.addi(activity.start_time, activity.end_time, activity)
            LOG.info("Found 'self-sufficient' activity: %s", activity)
            metrics.incr('self sufficient activities', activity.duration)
            if is_add_debug_buckets:
                _add_debug_event(
                    debug_dict,
                    debug_bucket_prefix,
                    activity.start_time,
                    activity.end_time - activity.start_time,  # Use end-start time, not duration by events.
                    activity.description,
                    len(activity.events),
                )
    # Check that at least something was added to the result tree. Otherwise no debug events were added.
    if len(result_tree) > 0:
        debug_buckets_cnt += 1

    # Take all remained intervals and using `out_activity_boundaries` value make activites with logic:
    #   - Make tree of remained intervals by cutting out all activities overlapping `result_tree` intervals.
    #   - Iterate this tree to find activities with at least MIN_DURATION_SEC duration and try to make them minimal.
    #     In details at start and at the middle of the gap:
    #       - choose shortest activity with `out_activity_boundaries` "whole" or "start"
    #         and duration >= MIN_DURATION_SEC, name it "basic activity" (BA)
    #       - find overlapping activities (OA-s)
    #           - if OA-s is smaller than BA then merge it into BA as is;
    #           - if OA is bigger than BA then check if `out_activity_boundaries` value allows to cut it
    #               - if `out_activity_boundaries` value allows then cut and merge overlapping part into BA;
    #               - if `out_activity_boundaries` value is against then cut until BA end
    #       - Sort longest actvities by `out_activity_name` and make name TODO
    #     At the end of the gap (when length of all activities is <= MIN_DURATION_SEC):
    #       - take first longest activity with `out_activity_boundaries` "whole" or "start", name it BA
    #       - find OA-s and merge with the same logic
    #       - Sort longest actvities by `out_activity_name` and make name TODO

    # 4. Make tree from remained activities. Chop them by existing `result` activities if there are such.
    candidates_tree = intervaltree.IntervalTree()
    if len(result_tree) > 0:
        LOG.info('Building "candidate" activities by chopping remaining activities by activities added so far.')
        for strategy_result in activities_by_strategy:
            strategy = strategy_result.strategy
            bucket_prefix = strategy.bucket_prefix
            # Skip already handled/contributed strategies.
            if bucket_prefix.startswith(BUCKET_AFK_PREFIX) or strategy.out_self_sufficient:
                continue
            # Cut activities from strategy by result tree. Do it strictly, i.e. boundaries=whole.
            remained_activities = _exclude_tree_intervals(
                strategy_result.activities, result_tree, metrics, 'activities built from self sufficient strategies'
            )
            # Iterate remained activities and put them into common candidates tree and into per-strategy tree if need.
            # Note that activities here are not in order!
            for activity in remained_activities:
                candidates_tree.addi(activity.suggested_start_time, activity.suggested_end_time, activity)
            # Add debug events and buckets if need.
            if is_add_debug_buckets:
                debug_buckets_cnt = _add_debug_events_to_not_overlap(
                    remained_activities, debug_dict, strategy.bucket_prefix, debug_buckets_cnt, metrics
                )
    else:
        LOG.info('Skipping chopping of remained activities because there were no "out_self_sufficient"'
                 ' strategies activities found this day.')
        for strategy_result in activities_by_strategy:
            for activity in strategy_result.activities:
                candidates_tree.addi(activity.suggested_start_time, activity.suggested_end_time, activity)

    # 5. Iterate remained activities to fill `result` remained gaps.
    LOG.info('Assemble activities-by-strategies into result activities.')
    current_start_point: datetime.datetime = candidates_tree.begin()  # Start from leftest/oldest activity.
    debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}999_activities"  # 999 - to place it last in UI.
    while current_start_point and len(result_tree) < 100:  # Limit number of iterations to avoid infinite loops on bugs or wrong input.
        metrics.incr("iterations to assemble remaining activities")
        # Find "basic activity" to base "result" activity on interval of it.
        ba_interval = _find_basic_activity_interval(candidates_tree, current_start_point, metrics)
        if ba_interval is None:
            break  # No more activities.
        # Find all overlapping activities and make new `result` activity (RA).
        ra = _build_result_activity(ba_interval, candidates_tree, metrics)
        # Cut RA by already existing result activities.
        result_tree_overlapped_with_ra_end: Set[intervaltree.Interval] = result_tree.at(ra.end_time)
        # If we had interval in `result_tree` when added RA then we need to search next gap.
        # Note that `result_tree_overlapped_with_ra_end` may contain few intervals not in order..
        for existing_interval in result_tree_overlapped_with_ra_end:
            if existing_interval.begin < ra.end_time:
                ra.end_time = existing_interval.begin  # TODO remove events from RA and correct name.
                metrics.incr('result activities shrinked because it overlaps by end with alredy existing',
                             (ra.end_time - existing_interval.begin).total_seconds())
        # Add RA into the result tree.
        result_tree.addi(ra.start_time, ra.end_time, ra)
        # Add RA to debug bucket if need.
        if is_add_debug_buckets:
            # Use the only debug bucket here because events should be consequtive.
            _add_debug_event(
                debug_dict,
                debug_bucket_prefix,
                ra.start_time,
                ra.end_time - ra.start_time,  # Use end-start time, not duration by events.
                ra.description,
                len(ra.events),
            )
        # Configure next iteration.
        current_start_point = ra.end_time
    return AnalyzerResult(
        sorted([x.data for x in result_tree], key=lambda x: x.start_time),
        None,
        metrics,
        debug_dict,
    )
