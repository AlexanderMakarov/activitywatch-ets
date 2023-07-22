import dataclasses
import collections
import datetime
from operator import attrgetter
from typing import List, Dict, Set, Tuple, Any
import intervaltree

from .strategies import ActivitiesByStrategy

from ..config.config import LOG, AFK_RULE_PRIORITY, WATCHDOG_RULE_PRIORITY, TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS,\
                            DEBUG_BUCKET_PREFIX, MIN_DURATION_SEC
from .interval import Interval, intervals_duration
from .input_entities import Event, Rule2, Strategy
from .metrics import Metrics
from .output_entities import RuleResult, Activity, AnalyzerResult
from ..helpers.helpers import seconds_to_int_timedelta, event_data_to_str, event_to_str


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
        debug_event = Event(DEBUG_BUCKET_PREFIX + "3_activities", activity.start_time, activity.duration,
                            {
                                'description': rule_result.description,
                                'events_count': len(activity.events),
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
        {
            BUCKET_DEBUG_RAW_RULE_RESULTS: raw_rule_result_debug_events,
            BUCKET_DEBUG_FINAL_RULE_RESULTS: final_rule_result_debug_events,
            BUCKET_DEBUG_ACTIVITIES: activity_debug_events,
        }
    )


def _cut_activity_start(activity: Activity, point: datetime.datetime) -> Activity:
    """
    Cuts start from activity and returns only tail after specified point.
    """
    events = [x for x in activity.events if (x.timestamp + x.duration) > point]  # Keep "border" event.
    return Activity(
        start_time=point,
        end_time=activity.end_time,
        events=events,
        description=activity.description,
        duration=sum(x.duration.seconds for x in events)
    )


def _cut_activity_end(activity: Activity, point: datetime.datetime) -> Activity:
    """
    Cuts end from activity and returns only head before specified point.
    """
    events = [x for x in activity.events if x.timestamp < point]  # Keep "border" event.
    return Activity(
        start_time=activity.start_time,
        end_time=point,
        events=events,
        description=activity.description,
        duration=sum(x.duration.seconds for x in events)
    )


def _split_activity(activity: Activity, start_point: datetime.datetime, end_point: datetime.datetime)\
        -> List[Activity]:
    """
    Cuts activity on 2 parts and returns 2 new activities:
    - from the start of initial activity to start_point,
    - from end_point to the end of initial activity.
    """
    return [_cut_activity_end(activity, start_point), _cut_activity_start(activity, end_point)]


def _exclude_tree_intervals(activities: List[Activity], boundaries: str, tree: intervaltree.IntervalTree,
                            metrics: Metrics, name_of_tree: str) -> List[Activity]:
    """
    Cuts different parts of activities which overlap with intervals in the specified tree.
    :param activities: List of activities to cut tree intervals from.
    :param boundaries: `Strategy` boundaries.
    :param tree: Tree with intervals to cut activities by.
    :param metrics: Metrics instance.
    :param name_of_tree: Name of the tree for metrics.
    :return: List of chopped activities.
    """
    may_cut_start = boundaries != 'start'
    may_cut_end = boundaries != 'end'
    result: List[Activity] = []
    # Iterate all activities to cut tree intervals from them.
    for activity in activities:
        modified_activities = [activity]
        intervals: List[intervaltree.Interval] = sorted(tree[activity.start_time:activity.end_time])
        # Iterate all intervals, and chop activities by them.
        # Note that after each iteration activities change and may stop to be intersecting.
        for i, interval in enumerate(intervals):
            activities_after_interval_handling = []
            # Note that after 1st interval handling number of activities to handle may be 2 and so on.
            # So even with not first interval and may_cut_end=True we can't stop because first interval
            # may split activity on 2 and second activity is still may be handled.
            # if i > 0 and may_cut_end and len(modified_activities) < 2:
            #     # If we cut end of activity then nothing to process anymore.
            #     break
            # Check how remained intervals overlaps with chopped activities and chop more, if needed.
            for current_activity in modified_activities:
                if interval.begin <= current_activity.start_time:
                    # Interval starts before activity.
                    # Check interval covers activity completely.
                    if interval.end >= current_activity.end_time:
                        metrics.incr('activities removed by ' + name_of_tree, current_activity.duration)
                        continue
                    # Check activity may be chopped at start.
                    if not may_cut_start:
                        metrics.incr('activities removed because impossible to cut on start by ' + name_of_tree,
                                     current_activity.duration)
                        continue
                    # Check interval overlaps activity.
                    if interval.end < current_activity.end_time:
                        prev_duration = current_activity.duration
                        tmp = _cut_activity_start(current_activity, interval.end)
                        activities_after_interval_handling.append(tmp)
                        metrics.incr('activities cut on start by ' + name_of_tree, prev_duration - tmp.duration)
                    continue
                elif interval.end >= current_activity.end_time:
                    # Interval ends after activity.
                    # Check activity may be chopped at end.
                    if not may_cut_end:
                        metrics.incr('activities removed because impossible to cut on end by ' + name_of_tree,
                                     current_activity.duration)
                        continue
                    # Check interval overlaps activity.
                    if interval.begin > current_activity.start_time:
                        prev_duration = current_activity.duration
                        tmp = _cut_activity_end(current_activity, interval.begin)
                        activities_after_interval_handling.append(tmp)
                        metrics.incr('activities cut on end by '+ name_of_tree, prev_duration - tmp.duration)
                    continue
                else:
                    # Interval is placed in the middle of the current activity.
                    prev_duration = current_activity.duration
                    if may_cut_end and may_cut_start:
                        split_activities = _split_activity(current_activity, interval.begin, interval.end)
                        # Note that left part of the activity won't be touched anymore. Put stright into result.
                        result.append(split_activities[0])
                        activities_after_interval_handling.append(split_activities[1])
                        # Note that duration is measured by events.
                        metrics.incr('activities with cut out middle by ' + name_of_tree,
                                     prev_duration - (interval.end - interval.begin).total_seconds())
                    elif may_cut_end:
                        tmp = _cut_activity_end(current_activity, interval.begin)
                        activities_after_interval_handling.append(tmp)
                        metrics.incr('activities cut on end by ' + name_of_tree, prev_duration - tmp.duration)
                    elif may_cut_start:
                        tmp = _cut_activity_start(current_activity, interval.end)
                        activities_after_interval_handling.append(tmp)
                        metrics.incr('activities cut on start by ' + name_of_tree, prev_duration - tmp.duration)
                    else:
                        raise NotImplementedError('Inner error when impossible cut both start and end of activity'
                                                  f' {current_activity} by {interval}')
            # Replace list of activities to handle for new interval.
            modified_activities = activities_after_interval_handling
        # Append activities to the result.
        result.extend(modified_activities)
    return result


def _add_debug_event(debug_dict: Dict[str, List[Event]], bucket_id: str, timestamp: datetime.datetime,
                     duration: datetime.timedelta, description: str, events_count: int, strategy_desc: str = None):
    data = {'desc': description, 'events_count': events_count}
    if strategy_desc:
        data['strategy_desc'] = strategy_desc
    debug_dict.setdefault(bucket_id, []).append(Event(bucket_id, timestamp, duration, data))


def _add_debug_events_to_not_overlap(activites: List[Activity], debug_dict: Dict,
                                     bucket_name: str, debug_buckets_cnt: int, metrics: Metrics) -> int:
    """
    Adds debug events to the given dictionary in few buckets where events are not overlapping.
    :param activites: Activities to add debug events from.
    :param debug_dict: Dictionary to add debug events to.
    :param bucket_name: Part of name for debug buckets.
    :param metrics: Metrics instance to report progress.
    :param debug_buckets_cnt: Counter of debug buckets.
    :return: Updated counter of debug buckets.
    """
    activites = sorted(activites, key=lambda x: x.start_time)
    groups: List[List[Activity]] = []
    # Seaparate activities by groups.
    for activity in activites:
        found_group = False
        for group in groups:
            is_overlapping = False
            # Iterate all groups each time to find place for the new activity.
            # Need to pack events as dense as possible - activities may overlap.
            for existing_activity in group:
                # Check activities overlap.
                if activity.end_time >= existing_activity.start_time \
                        and activity.start_time <= existing_activity.end_time:
                    is_overlapping = True
                    break
            if not is_overlapping:
                group.append(activity)
                found_group = True
                break
        if not found_group:
            groups.append([activity])
            metrics.incr('debug event groups for ' + bucket_name)
    # Fill buckets of events from groups.
    for group in groups:
        debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}{debug_buckets_cnt:03}_{bucket_name}"
        debug_buckets_cnt += 1
        for activity in group:
            _add_debug_event(
                debug_dict,
                debug_bucket_prefix,
                activity.start_time,
                activity.end_time - activity.start_time,  # Use end-start time, not duration by events.
                activity.description,
                len(activity.events),
            )
            metrics.incr('debug events in ' + debug_bucket_prefix, activity.duration)
    return debug_buckets_cnt


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
    result_tree = intervaltree.IntervalTree()  # Note that it started to be populated only on step 3.
    metrics = Metrics({})
    debug_dict: Dict[str, List[Event]] = {}
    debug_buckets_cnt = 1

    # 1. Find AFK strategy. It is required for `out_only_not_afk` handling.
    LOG.info('Searching AFK intervals.')
    afk_tree = intervaltree.IntervalTree()
    for strategy_result in activities_by_strategy:
        bucket_prefix = strategy_result.strategy.bucket_prefix
        if not bucket_prefix.startswith(BUCKET_AFK_PREFIX):
            continue
        metrics.incr('AFK strategies')
        if strategy_result.strategy.in_activities_may_overlap:
            LOG.warning("Unsupported setup for %s* strategy - in_activities_may_overlap=True."
                        "Skipping any AFK-related logic populated by this strategy.", bucket_prefix)
            continue
        # Add to afk_tree only AFK activities. Expect that they are not intersect. TODO support.
        for activity in strategy_result.activities:
            if activity.events[0].data['status'] == 'afk':
                afk_tree.addi(activity.start_time, activity.end_time, activity)
                metrics.incr('afk intervals', activity.duration)

    # 2. Cut activities from `out_only_not_afk=True` strategies.
    LOG.info('Chopping activities by AFK intervals.')
    for strategy_result in activities_by_strategy:
        if not strategy_result.strategy.out_only_not_afk:
            continue
        metrics.incr('strategies to cut by AFK')
        strategy_result.activities = _exclude_tree_intervals(
            strategy_result.activities, strategy_result.strategy.out_activity_boundaries, afk_tree, metrics, 'AFK'
        )

    # 3. Add into result activities from `out_self_sufficient=True` strategies. Check for overlappings.
    LOG.info('Adding "out_self_sufficient" strategies activities.')
    debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}{debug_buckets_cnt:03}_self_sufficient"
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

    # 4. Make tree from remained activities chopped by existing `result` activities.

    # TODO fix:
    # + aw-watcher-window below makes 2.5 days of activities. And it generates total mess.
    # - IDEA activities aren't chopped by AFK.
    # + "activities split on 2 by AFK" is a negative duration.
    # - resulting activities are overlapping on few seconds
    # - wrong activities in result - too much "Can't find basic activity".
    # TODO:
    # - update merger.py to populate "strict_start_time" and "strict_end_time" for `out_activity_boundaries` behavior.
    # - use `out_activity_name` to sanitize activity name.
    # + multiple debug buckets for "window" strategy (for all with in_activities_may_overlap=true) to avoid overlaps.

    LOG.info('Building "candidate" activities by chopping remaining activities by activities added so far.')
    candidates_tree = intervaltree.IntervalTree()
    for strategy_result in activities_by_strategy:
        # Skip already contributed strategies.
        strategy = strategy_result.strategy
        bucket_prefix = strategy.bucket_prefix
        if bucket_prefix.startswith(BUCKET_AFK_PREFIX) or strategy.out_self_sufficient:
            continue
        # Cut activities from strategy by result tree. Do it strictly, i.e. boundaries=whole.
        remained_activities = _exclude_tree_intervals(
            strategy_result.activities, 'whole', result_tree, metrics,
            'activities built from self sufficient strategies'
        )
        # Iterate remained activities and put them into common candidates tree and into per-strategy tree if need.
        # Note that activities here are not in order!
        for activity in remained_activities:
            candidates_tree.addi(activity.start_time, activity.end_time, activity)
        # Add debug events and buckets if need.
        if is_add_debug_buckets:
            debug_buckets_cnt = _add_debug_events_to_not_overlap(
                remained_activities, debug_dict, strategy.bucket_prefix, debug_buckets_cnt, metrics
            )

    # 5. Iterate remained activities to fill `result` remained gaps.
    LOG.info('Determine activities from remained and chopped activities.')
    min_duraiton_timedelta = datetime.timedelta(seconds=MIN_DURATION_SEC)
    current_start_point: datetime.datetime = candidates_tree.begin()  # Start from leftest/oldest activity.
    debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}999_activities"
    while current_start_point and len(result_tree) < 100:  # Limit number of iterations to avoid infinite loops on bugs or wrong input.
        metrics.incr("iterations to assemble remaining activities")
        # Get accumulator of all activities started in interval [last_point, min_duraiton].
        candidates_for_ba: List[intervaltree.Interval] = candidates_tree[
            current_start_point:current_start_point + min_duraiton_timedelta
        ]
        if not candidates_for_ba:
            # If nothing at right then stop iterating - no more actvities remained, we are done.
            break
        # Try find BA as:
        # - started on `current_start_point`,
        # - with `out_activity_boundaries` "whole" or "start",
        # - length is equal or more than `MIN_DURATION_SEC` (choose minimal).
        candidates_for_ba = sorted(candidates_for_ba, key=lambda x: (x.begin, x.end - x.begin))
        ba_interval = next((x for x in candidates_for_ba
                            if x.begin == current_start_point
                            and x.length() >= min_duraiton_timedelta
                            and x.data.strategy.out_activity_boundaries in ['whole', 'start']),
                           None)
        # If there are no such activity then just make BA from the longest activity.
        # Note that all "remained" activities are cut by existing "result" activities and shouldn't span too long.
        if ba_interval is None:
            ba_interval = max(candidates_for_ba, key=intervaltree.Interval.length)
            LOG.info("Can't find basic activity after %s and more than %s. Using as basic longest - %s",
                     current_start_point, min_duraiton_timedelta, ba_interval.data)
            metrics.incr("basic activities assembled from remainings", ba_interval.length().total_seconds())
        # Find all overlapping activities and make new `result` activity (RA).
        overlapping_intervals = candidates_tree.overlap(ba_interval.begin, ba_interval.end)
        metrics.incr(f'basic activities overlapped by {len(overlapping_intervals)} intervals')
        ra_events = ba_interval.data.events
        ra_description_parts: Set = set((ba_interval.data.description,))
        for interval in overlapping_intervals:
            activity = interval.data
            # If BA overlaps activity then just concatenate data from it into BA.
            if ba_interval.overlaps(interval):
                ra_events.extend(activity.events)
                ra_description_parts.add(activity.description)
                metrics.incr('basic activity completely overlaps intervals', interval.length().total_seconds())
                continue
            boundaries = activity.strategy.out_activity_boundaries
            # Check BA overlaps only start of activity.
            if ba_interval.contains_point(interval.begin):
                if boundaries == "start":
                    # If activity is `out_activity_boundaries=start` then concatenate data from it into BA.
                    ra_events.extend(activity.events)
                    ra_description_parts.add(activity.description)
                    metrics.incr('basic activity covers start of intervals', interval.length().total_seconds())
                elif boundaries == "whole":
                    # If activity is `out_activity_boundaries=whole` then split activity,
                    # and concatenate last part with BA. First part is not needed anyway.
                    split_activity = _cut_activity_end(activity, ba_interval.end)
                    ra_events.extend(split_activity.events)
                    ra_description_parts.add(split_activity.description)
                    metrics.incr('basic activity covers start part of intervals', split_activity.duration)
                else:
                    # If activity is `out_activity_boundaries=end` then skip it.
                    metrics.incr('basic activity skips end part of interval', interval.length().total_seconds())
                continue
            # Check BA overlaps only end of activity.
            if ba_interval.contains_point(interval.end):
                if boundaries == "end":
                    # If activity is `out_activity_boundaries=end` then concatenate data from it into BA.
                    ra_events.extend(activity.events)
                    ra_description_parts.add(activity.description)
                    metrics.incr('basic activity covers end of intervals', interval.length().total_seconds())
                elif boundaries == "whole":
                    # If activity is `out_activity_boundaries=whole` then split activity,
                    # and concatenate first part with BA.
                    split_activity = _cut_activity_start(activity, ba_interval.begin)
                    ra_events.extend(split_activity.events)
                    ra_description_parts.add(split_activity.description)
                    metrics.incr('basic activity covers end part of intervals', split_activity.duration)
                else:
                    # If activity is `out_activity_boundaries=start` then skip it.
                    metrics.incr('basic activity skips start part of interval', interval.length().total_seconds())
                continue
            # If BA itself is placed inside actvity then concatenate middle of it
            # and only if it is `out_activity_boundaries=whole`.
            if boundaries == "whole":
                tmp = _cut_activity_start(activity, ba_interval.begin)
                tmp = _cut_activity_end(tmp, ba_interval.end)
                ra_events.extend(tmp.events)
                ra_description_parts.add(tmp.description)
                metrics.incr('basic activity covers middle part of interval', ba_interval.length().total_seconds())
                continue
            # Skip all other cases.
            metrics.incr('skipped overlapping activity')
        # Build RA and add into results tree.
        ra = Activity(
            ba_interval.begin,
            ba_interval.end,
            ra_events,
            ", ".join(ra_description_parts),
            ba_interval.length().seconds
        )
        result_tree_overlapped_with_ra_end = result_tree.at(ra.end_time)
        result_tree.addi(ra.start_time, ra.end_time, ra)
        # Add to debug bucket if need.
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
        # Need to jump to the next gap. Note that current RA may end up on existing activity in the `result_tree`.
        # If we had interval in `result_tree` when added RA then we need to search next gap.
        # Assume that `result_tree_overlapped_with_ra_end` contains the only interval.
        while result_tree_overlapped_with_ra_end:
            current_start_point = result_tree_overlapped_with_ra_end[0].begin
            # Find interval which starts on the then of the current. Do this until nothing found.
            result_tree_overlapped_with_ra_end = result_tree.at(current_start_point)
    return AnalyzerResult(
        sorted([x.data for x in result_tree], key=lambda x: x.start_time),
        None,
        metrics,
        debug_dict,
    )
