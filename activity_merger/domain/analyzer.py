import collections
import datetime
from typing import Dict, List, Optional, Set, Tuple

import intervaltree

from activity_merger.helpers.helpers import datetime_to_time_str

from ..config.config import (
    DEBUG_BUCKET_PREFIX,
    LIMIT_OF_RESULTING_ACTIVITIES,
    LOG,
    MAX_ACTIVITY_DURATION_SEC,
    MIN_ACTIVITY_DURATION_SEC,
)
from .input_entities import Event, IntervalBoundaries, Strategy
from .metrics import Metrics
from .output_entities import Activity, AnalyzerResult
from .strategies import BUCKET_AFK_PREFIX, ActivitiesByStrategy, ActivityByStrategy, calculate_activity_density


def _cut_activity_start(activity: ActivityByStrategy, point: datetime.datetime) -> ActivityByStrategy:
    """
    Cuts start from activity (suggested and 'max_start_time' if need) and returns only tail after specified point.
    """
    events = [x for x in activity.events if (x.timestamp + x.duration) > point]  # Keep "border" event.
    density_zero = activity.density == 0  # If density was zero before cutoff then it will remain same.
    return ActivityByStrategy(
        suggested_start_time=point,
        suggested_end_time=activity.suggested_end_time,
        max_start_time=max(point, activity.max_start_time),
        min_end_time=activity.min_end_time,
        events=events,
        density=0 if density_zero else calculate_activity_density(events, point, activity.suggested_end_time),
        grouping_data=activity.grouping_data,
        strategy=activity.strategy,
    )


def _cut_activity_end(activity: ActivityByStrategy, point: datetime.datetime) -> ActivityByStrategy:
    """
    Cuts end from activity (suggested and 'min_end_time' if need) and returns only head before specified point.
    """
    events = [x for x in activity.events if x.timestamp < point]  # Keep "border" event.
    density_zero = activity.density == 0  # If density was zero before cutoff then it will remain same.
    return ActivityByStrategy(
        suggested_start_time=activity.suggested_start_time,
        suggested_end_time=point,
        max_start_time=activity.max_start_time,
        min_end_time=min(point, activity.min_end_time),
        events=events,
        density=0 if density_zero else calculate_activity_density(events, activity.suggested_start_time, point),
        grouping_data=activity.grouping_data,
        strategy=activity.strategy,
    )


def _split_activity(
    activity: ActivityByStrategy, start_point: datetime.datetime, end_point: datetime.datetime
) -> List[ActivityByStrategy]:
    """
    Cuts some middle part from the activity to get 2 new 'ActivityByStrategy'-s:
    - from the start of initial activity to the 'start_point',
    - from the 'end_point' to the end of initial activity.
    """
    return [_cut_activity_end(activity, start_point), _cut_activity_start(activity, end_point)]


def _exclude_tree_intervals(
    activities: List[ActivityByStrategy], tree: intervaltree.IntervalTree, metrics: Metrics, name_of_tree: str
) -> List[ActivityByStrategy]:
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
        intervals: List[intervaltree.Interval] = tree.overlap(
            activity.suggested_start_time, activity.suggested_end_time
        )
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
                metrics.incr(
                    "activities removed by " + name_of_tree,
                    (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
                )
                continue
            # Check (second time actually) interval overlaps activity.
            if interval.end < activity.suggested_end_time:
                prev_start_time = activity.suggested_start_time
                tmp = _cut_activity_start(activity, interval.end)
                activities_queue.append(tmp)
                metrics.incr(
                    "activities with head cut by " + name_of_tree,
                    (tmp.suggested_start_time - prev_start_time).total_seconds(),
                )
                continue
            else:
                raise NotImplementedError(f"Inner error with {interval} actually not overlapping with" f" {activity}.")
        elif interval.end >= activity.suggested_end_time:
            # Interval ends after activity.
            # Check (second time actually) activity end is overlapped by interval.
            if interval.begin > activity.suggested_start_time:
                prev_end_time = activity.suggested_end_time
                tmp = _cut_activity_end(activity, interval.begin)
                activities_queue.append(tmp)  # Due to interval was random, activity need to check one more time.
                metrics.incr(
                    "activities with tail cut by " + name_of_tree,
                    (prev_end_time - tmp.suggested_end_time).total_seconds(),
                )
            else:
                raise NotImplementedError(f"Inner error with {interval} actually not overlapping with" f" {activity}.")
        else:
            # Interval is placed in the middle of the current activity.
            split_activities = _split_activity(activity, interval.begin, interval.end)
            # Due to interval was random, both parts of the initial activity should be checked again.
            activities_queue.extend(split_activities)
            # Update metric without duration because very often activity is cut few times and resulting
            # cumulative duration is bigger than initial activity duration.
            metrics.incr("activities with middle cut by " + name_of_tree)
            continue
    # Sort result because parts of long activities may be placed into it randomly.
    return sorted(result, key=lambda x: x.suggested_start_time)


def _include_tree_intervals(
    activities: List[ActivityByStrategy],
    boundaries: IntervalBoundaries,
    tree: intervaltree.IntervalTree,
    metrics: Metrics,
    name_of_tree: str,
) -> List[ActivityByStrategy]:
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
            activity.suggested_start_time, activity.suggested_end_time
        )

        # Check that activity is not overlapping at all.
        if not overlapping_intervals:
            metrics.incr(
                f"activities removed because are out of {name_of_tree}",
                (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
            )
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
            metrics.incr(
                f"activities completely covered by {name_of_tree}",
                (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
            )
            result.append(activity)
            continue
        else:
            metrics.incr(
                f"activities not completely covered by {name_of_tree} and need to be chopped",
                (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
            )

        # Check remained ActivityBoundaries cases.
        if boundaries == IntervalBoundaries.START:
            if not start_covered:
                metrics.incr(
                    f"activities removed with {boundaries} started before {name_of_tree}",
                    (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
                )
                continue
            if segments[0][1] < activity.max_start_time:
                metrics.incr(
                    f"activities removed with {boundaries} end out of {name_of_tree}",
                    (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
                )
                continue
            prev_end_time = activity.suggested_end_time
            activity = _cut_activity_end(activity, segments[0][1])
            result.append(activity)
            metrics.incr(
                f"activities with tail cut by {name_of_tree}",
                (prev_end_time - activity.suggested_end_time).total_seconds(),
            )
        elif boundaries == IntervalBoundaries.END:
            if not end_covered:
                metrics.incr(
                    f"activities removed with {boundaries} ended before {name_of_tree}",
                    (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
                )
                continue
            if segments[-1][0] > activity.min_end_time:
                metrics.incr(
                    f"activities removed with {boundaries} start out of {name_of_tree}",
                    (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
                )
                continue
            prev_start_time = activity.suggested_start_time
            activity = _cut_activity_start(activity, segments[-1][0])
            result.append(activity)
            metrics.incr(
                f"activities with start cut by {name_of_tree}",
                (activity.suggested_start_time - prev_start_time).total_seconds(),
            )
            continue
        elif boundaries == IntervalBoundaries.DIM:
            for segment in segments:
                # Chop some segment from the activity. Doesn't check min and max time points of activity.
                prev_duration = activity.suggested_end_time - activity.suggested_start_time
                is_chopped = False
                # Check activity start is covered by segment.
                if segment[0] > activity.suggested_start_time:
                    activity = _cut_activity_start(activity, segment[0])  # Get rid of head not covered by segment.
                    is_chopped = True
                # Check end of activity is covered by segment.
                if segment[1] < activity.suggested_end_time:
                    tmp = _cut_activity_end(activity, segment[1])  # Cut head coverent by segment to put into result.
                    result.append(tmp)
                    is_chopped = True
                else:
                    result.append(activity)  # Put the whole activity into result.
                # Check activity was chopped at all.
                if is_chopped:
                    # Update metric without duration because very often activity is cut few times and resulting
                    # cumulative duration is bigger than initial activity duration.
                    metrics.incr(f"activities with {boundaries} cut by {name_of_tree}")
                else:
                    metrics.incr(f"activities completely covered by {name_of_tree}", prev_duration.total_seconds())
        else:
            metrics.incr(
                f"activities with {boundaries} removed by {name_of_tree}",
                (activity.suggested_end_time - activity.suggested_start_time).total_seconds(),
            )
    return result


class DebugBucketsHandler:
    """
    Manages debug buckets.
    """

    def __init__(self):
        self.events: Dict[str, List[Event]] = {}
        self.cnt = 0  # Will be set to 1 in `build_new_bucket_id`.

    def build_new_bucket_id(self, bucket_name: str) -> str:
        result = f"{DEBUG_BUCKET_PREFIX}{self.cnt:03}_{bucket_name}"
        self.cnt += 1
        return result

    def add_event(
        self,
        bucket_id: str,
        timestamp: datetime.datetime,
        duration: datetime.timedelta,
        description: str,
        density: float,
        events_count: int,
    ):
        # Note that ActivityWatch UI shows only strings.
        if density:
            data = {"desc": description, "events_count": str(events_count), "density": f"{density:.2f}"}
        else:
            data = {"desc": description, "events_count": str(events_count)}
        self.events.setdefault(bucket_id, []).append(Event(bucket_id, timestamp, duration, data))

    def add_debug_events_to_not_overlap(
        self, activites: List[ActivityByStrategy], bucket_suffix: str, metrics: Metrics
    ) -> None:
        """
        Adds debug events in few buckets where events are not overlapping.
        Uses "suggested" time boundaries, not "minimal" ones.
        :param activites: Activities to add debug events from.
        :param bucket_suffix: Last part of name for debug buckets.
        :param metrics: Metrics instance to report progress.
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
                    if (
                        activity.suggested_end_time >= existing_activity.suggested_start_time
                        and activity.suggested_start_time <= existing_activity.suggested_end_time
                    ):
                        is_overlapping = True
                        break
                if not is_overlapping:
                    group.append(activity)
                    found_group = True
                    break
            if not found_group:
                groups.append([activity])
                metrics.incr(f"debug event groups for {bucket_suffix}.* strategy")
        # Fill buckets of events from groups.
        for group in groups:
            debug_bucket_prefix = self.build_new_bucket_id(bucket_suffix)
            for activity in group:
                self.add_event(
                    bucket_id=debug_bucket_prefix,
                    timestamp=activity.suggested_start_time,
                    # For debug events need to use end-start time, not duration by events.
                    duration=activity.suggested_end_time - activity.suggested_start_time,
                    # Build description in easiest way.
                    description=", ".join(f"{k}={v}" for k, v in activity.grouping_data.get_kv_pairs()),
                    density=activity.density,
                    events_count=len(activity.events),
                )
                metrics.incr("debug events in " + debug_bucket_prefix, activity.duration())


def _build_result_activity(
    ba_interval: intervaltree.Interval,
    candidates_tree: intervaltree.IntervalTree,
    is_only_good_strategies_for_description: bool,
    metrics: Metrics,
) -> Activity:
    # Find all overlapping activities.
    overlapping_intervals = candidates_tree.overlap(ba_interval.begin, ba_interval.end)  # Includes BA.
    LOG.info("Basic activity is overlapped by %d 'candidate' activities.", len(overlapping_intervals))
    ra_events = ba_interval.data.events
    overlapping_activities: List[ActivityByStrategy] = [ba_interval.data]
    for interval in overlapping_intervals:
        if interval == ba_interval:
            candidates_tree.remove(interval)  # Make sure it won't appear on other "result" activity.
            continue
        activity = interval.data
        boundaries = activity.strategy.in_trustable_boundaries
        # 1. If BA overlaps activity completely then just concatenate data from it into BA.
        if ba_interval.contains_interval(interval):
            ra_events.extend(activity.events)
            overlapping_activities.append(activity)
            candidates_tree.remove(interval)  # Make sure it won't appear on other "result" activity.
            metrics.incr("activities placed completely inside basic activities", interval.length().total_seconds())
            continue
        elif boundaries == IntervalBoundaries.STRICT:
            # We can't use part of STRICT activity so just report attempt and skip.
            LOG.warning(
                "Activity with %s boundaries appeared in the border of resulting activity: %s",
                IntervalBoundaries.STRICT,
                activity,
            )
            metrics.incr(
                f"activities with {IntervalBoundaries.STRICT} boundaries skipped completely from basic activities",
                interval.length().total_seconds(),
            )
            continue
        # 2. Handle case when BA overlaps the start of the activity.
        if ba_interval.contains_point(interval.begin):
            if boundaries == IntervalBoundaries.START:
                # If activity is `in_trustable_boundaries=start` then concatenate data from it into BA.
                ra_events.extend(activity.events)
                overlapping_activities.append(activity)
                candidates_tree.remove(interval)  # Make sure it won't appear on other "result" activity.
                metrics.incr("activities absorbed by basic activity at the start", interval.length().total_seconds())
            elif boundaries == IntervalBoundaries.DIM:
                # Split activity and concatenate last part with BA. First part is not needed anyway.
                split_activity = _cut_activity_end(activity, ba_interval.end)
                ra_events.extend(split_activity.events)
                overlapping_activities.append(split_activity)
                metrics.incr("activities enhancing basic activity by the start", split_activity.duration())
            else:
                # If activity is `in_trustable_boundaries=end` then skip it.
                metrics.incr(
                    "activities unable to enhance basic activity by the start", interval.length().total_seconds()
                )
            continue
        # 3. Handle case when BA overlaps the end of activity.
        if ba_interval.contains_point(interval.end):
            if boundaries == IntervalBoundaries.END:
                # If activity is `in_trustable_boundaries=end` then concatenate data from it into BA.
                ra_events.extend(activity.events)
                overlapping_activities.append(activity)
                candidates_tree.remove(interval)  # Make sure it won't appear on other "result" activity.
                metrics.incr("activities absorbed by basic activity at the end", interval.length().total_seconds())
            elif boundaries == IntervalBoundaries.DIM:
                # If activity is `in_trustable_boundaries=whole` then split activity,
                # and concatenate first part with BA.
                split_activity = _cut_activity_start(activity, ba_interval.begin)
                ra_events.extend(split_activity.events)
                overlapping_activities.append(split_activity)
                metrics.incr("activities enhancing basic activity by the end", split_activity.duration())
            else:
                # If activity is `in_trustable_boundaries=start` then skip it.
                metrics.incr(
                    "activities unable to enhance basic activity by the end", interval.length().total_seconds()
                )
            continue
        # 4 Handle case when BA itself is placed inside actvity.
        if boundaries == IntervalBoundaries.DIM:
            tmp = _cut_activity_start(activity, ba_interval.begin)
            tmp = _cut_activity_end(tmp, ba_interval.end)
            ra_events.extend(tmp.events)
            overlapping_activities.append(tmp)
            metrics.incr("activities enhancing basic activity by the middle", ba_interval.length().total_seconds())
            continue
        else:
            # Otherwise BA is in the middle of activity with START or END boundaries.
            continue

    # Build raw RA, i.e. with "not sure end". Fill it with all the events from the "enhancing" activities.
    ra_duration = ba_interval.length().seconds
    name = _build_activity_name(
        overlapping_activities, metrics, ra_duration, not is_only_good_strategies_for_description
    )
    metrics.incr("result activities", ra_duration)
    return Activity(
        ba_interval.begin,
        ba_interval.end,
        ra_events,
        name,
    )


def _build_activity_name(
    activities: List[ActivityByStrategy], metrics: Metrics, duration_sec: float, is_use_all_strategies: bool
) -> str:
    """
    Builds name of the resulting activity from list of `ActivityByStrategy`-s.
    In details all activities are grouped by strategy, sorted by "out_produces_good_activity_name"
    to get "good" names first or the only (depending on the settings), next processed to call
    "out_activity_name_sentence_builder" with key-value pairs received from related `GroupingDescriptor`-s.
    Handles "out_activity_name_sentence_builder" absence with default key-value pairs stringifier and
    with adding nothing if "out_activity_name_sentence_builder" returns empty value.
    :param activities: list of `ActivityByStrategy` making resulting activity.
    :param metrics: Metrics object to report details into.
    :param duration_sec: Duration of the resulting activity.
    :param is_use_all_strategies: Flag to build activity description from all strategies, not only from "good" ones.
    :return: Resulting activity description.
    """
    # Group activities by Strategy. Keep map of strategies to name. Name is a key in both dictionaries.
    strategy_to_name: Dict[str, Strategy] = {}
    grouped_activities: Dict[str, List[ActivityByStrategy]] = collections.defaultdict(list)
    for activity in activities:
        strategy_name = activity.strategy.name
        grouped_activities[strategy_name].append(activity)
        strategy_to_name[strategy_name] = activity.strategy
    # Sort groups to get "out_produces_good_activity_name" strategy and longest activities first.
    sorted_strategies = sorted(
        strategy_to_name.values(),
        key=lambda x: (
            x.out_produces_good_activity_name,
            sum(a.duration() * a.density for a in grouped_activities[x.name]),
        ),
        reverse=True,
    )
    sorted_grouped_activities = [grouped_activities[x.name] for x in sorted_strategies]

    # Determine whether need to include all strategies in the result.
    include_all = is_use_all_strategies or not sorted_strategies[0].out_produces_good_activity_name

    # Process each group of activities to build sentence.
    resulting_names: List[str] = []
    for grouped_activity_list in sorted_grouped_activities:
        common_kv_pairs = collections.defaultdict(set)

        # Collect the common key-value pairs from grouping_data
        activity: ActivityByStrategy
        for activity in grouped_activity_list:
            for pair in activity.grouping_data.get_kv_pairs():
                common_kv_pairs[pair[0]].add(pair[1])

        # Convert all values to comma-separated string.
        common_kv_list = [(k, ", ".join(sorted(v))) for k, v in common_kv_pairs.items()]

        # Use the strategy's out_activity_name_builder to get the name, or use a default logic.
        strategy = strategy_to_name[grouped_activity_list[0].strategy.name]
        name_builder = (
            strategy.out_activity_name_sentence_builder
            if strategy.out_activity_name_sentence_builder
            else lambda kv_pairs: ", ".join([f"{k}: {v}" for k, v in kv_pairs])
        )

        # Append to resulting names if the strategy produces a good activity name
        generated_name = name_builder(common_kv_list)
        if generated_name:
            resulting_names.append(generated_name)
        if not include_all and not strategy.out_produces_good_activity_name:
            metrics.incr("result activities with good name", duration_sec)
            break
    metrics.incr(f"result activities with name combined from {len(sorted_grouped_activities)} strategies", duration_sec)

    return " ".join(resulting_names)


def _add_ra(ra: Activity, is_add_debug_buckets: bool, result_tree: intervaltree.IntervalTree, debug_buckets_handler: DebugBucketsHandler):
    result_tree.addi(ra.start_time, ra.end_time, ra)
    LOG.info("Added into 'result tree' activity: %s", ra)
    # Add RA to debug bucket if need.
    debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}999_activities"  # 999 - to place it last in UI.
    if is_add_debug_buckets:
        # Use the only debug bucket here because events should be consequtive.
        debug_buckets_handler.add_event(
            bucket_id=debug_bucket_prefix,
            timestamp=ra.start_time,
            duration=ra.end_time - ra.start_time,  # Use end-start time, not duration by events.
            description=ra.description,
            density=0,  # Don't set density for "resulting" activities.
            events_count=len(ra.events),
        )


class AnalyzerStep:
    """
    Interface of "analyzer" step. Expected to be used for to make chain to analyse "activities by strategy"
    and transform them into resulting activities.
    """

    def get_description(self) -> str:
        """
        Returns human-friendly step description.
        """
        raise NotImplementedError("Not implemented 'get_description'.")

    def check_context(self, context: Dict[str, any]) -> None:  # TODO consider to use contextvars package.
        """
        Checks that all required items are present in the context.
        Raises an exception if the context is invalid. Expected to be executed before `run` method.
        By-default checks nothing.
        """

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        """
        Executes this step on the given context. All results are placed back to the context.
        By-default does nothing.
        Should return True if passed.
        """
        return False


class MakeResultTreeFromSelfSufficientActivitiesStep(AnalyzerStep):
    """
    Makes "result_tree" `IntervalTree` from "out_self_sufficient" activities.
    """

    def __init__(self, is_add_debug_buckets: bool = False):
        super(MakeResultTreeFromSelfSufficientActivitiesStep, self).__init__()
        self.is_add_debug_buckets = is_add_debug_buckets

    def get_description(self) -> str:
        return "Making 'result_tree' from self sufficient activities."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "activities_by_strategy" in context, "Need in 'activities_by_strategy' property"
        if self.is_add_debug_buckets:
            if "debug_buckets_handler" not in context:
                context["debug_buckets_handler"] = DebugBucketsHandler()

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        result_tree = intervaltree.IntervalTree()
        strategy_result: ActivitiesByStrategy
        for strategy_result in context["activities_by_strategy"]:
            if not strategy_result.strategy.out_self_sufficient:
                continue
            metrics.incr("self sufficient strategies")
            activity: ActivityByStrategy
            for activity in strategy_result.activities:
                overlap: List[intervaltree.Interval] = result_tree.overlap(
                    activity.suggested_start_time, activity.suggested_end_time
                )
                if overlap:
                    raise ValueError(
                        f"Overlapping activities from {strategy_result.strategy}: "
                        f"{activity} is overlapping with " + ", ".join(x.data for x in overlap)
                    )
                duration = activity.duration()
                name = _build_activity_name([activity], metrics, duration, True)
                metrics.incr("result activities", duration)
                ra = Activity(
                    activity.suggested_start_time,
                    activity.suggested_end_time,
                    activity.events,
                    name,
                )
                _add_ra(ra, self.is_add_debug_buckets, result_tree, debug_buckets_handler)
                metrics.incr("self sufficient activities", duration)
        context["result_tree"] = result_tree
        return True


class ChopActivitiesByResultTreeStep(AnalyzerStep):
    """
    Cuts activites by "result_tree" `IntervalTree` intervals.
    """

    def __init__(self, is_skip_afk: bool = False, is_skip_self_sufficient_strategies: bool = True):
        super(ChopActivitiesByResultTreeStep, self).__init__()
        self.is_skip_afk = is_skip_afk
        self.is_skip_self_sufficient_strategies = is_skip_self_sufficient_strategies

    def get_description(self) -> str:
        return "Chopping 'candidates_tree' from all not self sufficient activities."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "activities_by_strategy" in context, "Need in 'activities_by_strategy' property"
        assert "result_tree" in context, "Need in 'result_tree' property"

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        result_tree: intervaltree.IntervalTree = context.get("result_tree")
        if len(result_tree) > 0:
            strategy_result: ActivitiesByStrategy
            for strategy_result in context["activities_by_strategy"]:
                bucket_prefix = strategy_result.strategy.bucket_prefix
                # Skip AFK activities and self sufficient if need.
                if (self.is_skip_afk and bucket_prefix.startswith(BUCKET_AFK_PREFIX)) or (
                    self.is_skip_self_sufficient_strategies and strategy_result.strategy.out_self_sufficient
                ):
                    continue
                # Cut activities from strategy by result tree. Do it strictly, i.e. boundaries=whole.
                strategy_result.activities = _exclude_tree_intervals(
                    strategy_result.activities, result_tree, metrics, "result_tree"
                )
        return True


class MakeCandidatesTreeStep(AnalyzerStep):
    """
    Makes "candidates_tree" `IntervalTree`.
    """

    def __init__(
        self, is_add_debug_buckets: bool = False, is_add_afk: bool = False, is_add_self_sufficient: bool = False
    ):
        super(MakeCandidatesTreeStep, self).__init__()
        self.is_add_afk = is_add_afk
        self.is_add_self_sufficient = is_add_self_sufficient
        self.is_add_debug_buckets = is_add_debug_buckets

    def get_description(self) -> str:
        return "Building 'candidates_tree' activities."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "activities_by_strategy" in context, "Need in 'activities_by_strategy' property"
        if self.is_add_debug_buckets:
            if "debug_buckets_handler" not in context:
                context["debug_buckets_handler"] = DebugBucketsHandler()

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        candidates_tree = intervaltree.IntervalTree()
        strategy_result: ActivitiesByStrategy
        for strategy_result in context["activities_by_strategy"]:
            # Skip some strategies if need.
            if (not self.is_add_afk and strategy_result.strategy.bucket_prefix.startswith(BUCKET_AFK_PREFIX)) or (
                not self.is_add_self_sufficient and strategy_result.strategy.out_self_sufficient
            ):
                continue
            for activity in strategy_result.activities:
                candidates_tree.addi(activity.suggested_start_time, activity.suggested_end_time, activity)
            if self.is_add_debug_buckets:
                debug_buckets_handler.add_debug_events_to_not_overlap(
                    activites=strategy_result.activities,
                    bucket_suffix=strategy_result.strategy.bucket_prefix + "_candidate",
                    metrics=metrics,
                )
        context["candidates_tree"] = candidates_tree
        return True


class MergeCandidatesTreeIntoResultTreeStep(AnalyzerStep):
    """
    Makes "candidates_tree" `IntervalTree`.
    """

    def __init__(self, is_add_debug_buckets: bool = False, is_only_good_strategies_for_description: bool = True):
        super(MergeCandidatesTreeIntoResultTreeStep, self).__init__()
        self.is_only_good_strategies_for_description = is_only_good_strategies_for_description
        self.is_add_debug_buckets = is_add_debug_buckets

    def get_description(self) -> str:
        return "Merging 'candidates_tree' into 'result_tree'."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "result_tree" in context, "Need in 'result_tree' property"
        assert "candidates_tree" in context, "Need in 'candidates_tree' property"
        if self.is_add_debug_buckets:
            if "debug_buckets_handler" not in context:
                context["debug_buckets_handler"] = DebugBucketsHandler()

    def _check_and_report_basic_activity_interval(
        self,
        ba_interval: intervaltree.Interval,
        found_among: int,
        start_point: datetime.datetime,
        metric_name: str,
        metrics: Metrics,
    ) -> intervaltree.Interval:
        assert ba_interval.end > ba_interval.begin, f"Inner error with choosing as basic: {ba_interval.data}"
        metrics.incr(metric_name, (ba_interval.end - ba_interval.begin).total_seconds())
        LOG.info(
            "Sorted out 'basic' activity at %s..%s from %d candidates: %s",
            datetime_to_time_str(start_point),
            datetime_to_time_str(ba_interval.end),
            found_among,
            ba_interval.data,
        )
        return ba_interval

    def _find_basic_activity_interval(
        self,
        candidates_tree: intervaltree.IntervalTree,
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
        metrics: Metrics,
    ) -> Optional[intervaltree.Interval]:
        # We need to find "basic activity" as close as possible to the `start_point`.
        # Activities may both start later than `start_point` and start before `start_point`.
        # BA should better start on `start_point` but in case if no such activities this time or it is some
        # START/END boundaries activity then it cannot be BA.
        # So filter out what would be the appropriate:
        # - Best case when BA starts exactly on `start_point` and ends on `end_point` or
        #   later then `start_point` + max_duration_seconds.
        # - If BA starts earlier than `start_point` it should end later then `start_point` + MIN_ACTIVITY_DURATION_SEC.
        # - If no such activities left then BA starts later.
        candidates: Set[intervaltree.Interval] = candidates_tree.overlap(start_point, end_point)
        if not candidates:
            return None

        candidate_scores: List[Tuple[int, intervaltree.Interval]] = []
        for candidate in candidates:
            score = 0
            # Set scores for start and end points.
            if candidate.begin == start_point:
                score += 100
            if candidate.end == end_point:
                score += 100
            # Set scores for the duration on the intereseted interval.
            overlap_sec = candidate.overlap_size(start_point, end_point).total_seconds()
            if overlap_sec < 0:
                score -= 1000
            if MIN_ACTIVITY_DURATION_SEC <= overlap_sec <= max_duration_seconds:
                score += 200
            # Set scores for the boundaries.
            boundaries: IntervalBoundaries = candidate.data.strategy.in_trustable_boundaries
            if boundaries == IntervalBoundaries.STRICT:
                score += 100
            elif boundaries == IntervalBoundaries.DIM:
                score += 50
            # Set scores for the density.
            score += candidate.data.density * 100
            # Store score in the list.
            candidate_scores.append((score, candidate))
        # Take candidate with highest score.
        result = sorted(candidate_scores, key=lambda x: x[0], reverse=True)[0]
        score = result[0]
        # Name candidate score.
        score_description = "low"
        if score == 650:
            score_description = "highest"
        elif score > 450:
            score_description = "high"
        elif score > 200:
            score_description = "good"
        return self._check_and_report_basic_activity_interval(
            result[1], len(candidates), start_point, f"basic activities with {score_description} score", metrics
        )

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        result_tree: intervaltree.IntervalTree = context["result_tree"]
        candidates_tree: intervaltree.IntervalTree = context["candidates_tree"]
        debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}999_activities"  # 999 - to place it last in UI.

        # Loop through candidates tree and try to fill up gaps in result tree with intervals from here.
        # Note that very often results tree will be empty and need to make up all activities from candidates tree.

        # Choose initial "current" points:
        # TODO fix bug below - wrong coverage by activities of day
        current_start_point: datetime.datetime
        current_end_point: datetime.datetime
        if not result_tree:
            current_start_point = candidates_tree.begin()
            current_end_point = candidates_tree.end()
        else:
            current_start_point = min(candidates_tree.begin(), result_tree.begin())
            current_end_point = None
            for ra_interval in sorted(result_tree):
                if ra_interval.begin > current_start_point:
                    # I.e. start point is placed before and we may set end point by begin of result_tree.
                    current_end_point = ra_interval.begin
                    break
                else:
                    # I.e. current interval begins before start point. Then check end of interval.
                    if ra_interval.end >= candidates_tree.end():
                        # I.e. candidates tree starts somewhere on this result_tree interval.
                        # In this case we may choose start point at the end of this interval.
                        current_start_point = ra_interval.end
                    else:
                        # If candidates_tree is not started yet then go next.
                        continue
            # If end point was not set at all then just see up to end of the candidates_tree.
            if current_end_point is None:
                current_end_point = candidates_tree.end()

        # Start loop of filling result_tree with activities from candidates_tree.
        # Update current_start_point and current_end_point on each iteration.
        while current_start_point and len(result_tree) < LIMIT_OF_RESULTING_ACTIVITIES:
            metrics.incr("iterations to assemble remaining activities")
            # Find "basic activity" to base "result" activity on interval of it.
            ba_interval = self._find_basic_activity_interval(
                candidates_tree,
                current_start_point,
                current_end_point,
                MAX_ACTIVITY_DURATION_SEC,
                metrics,
            )
            if ba_interval is None:
                break  # No more activities are possible.
            # Find all overlapping activities and make new `result` activity (RA).
            ra = _build_result_activity(
                ba_interval, candidates_tree, self.is_only_good_strategies_for_description, metrics
            )
            # Check RA doesn't overlaps with existing result activities at the end.
            result_tree_overlapped_with_ra_end: Set[intervaltree.Interval] = result_tree.at(ra.end_time)
            # If we had interval in `result_tree` when added RA then we need to search next gap.
            # Note that `result_tree_overlapped_with_ra_end` may contain few intervals not in order.
            for existing_interval in result_tree_overlapped_with_ra_end:
                if existing_interval.begin < ra.end_time:
                    # Chop end of resulting activity if it overlaps with already existing iterval in `result_tree`.
                    ra.end_time = existing_interval.begin
                    ra.events = [x for x in ra.events if x.timestamp < ra.end_time]
                    metrics.incr(
                        "result activities shrinked because it overlaps by end with alredy existing",
                        (ra.end_time - existing_interval.begin).total_seconds(),
                    )
            # Add RA into the result tree.
            result_tree.addi(ra.start_time, ra.end_time, ra)
            LOG.info("Added into 'result tree' activity: %s", ra)
            # Add RA to debug bucket if need.
            if self.is_add_debug_buckets:
                # Use the only debug bucket here because events should be consequtive.
                debug_buckets_handler.add_event(
                    bucket_id=debug_bucket_prefix,
                    timestamp=ra.start_time,
                    duration=ra.end_time - ra.start_time,  # Use end-start time, not duration by events.
                    description=ra.description,
                    density=0,  # Don't set density for "resulting" activities.
                    events_count=len(ra.events),
                )
            # Configure next iteration.
            current_start_point = ra.end_time
        context["analyzer_result"] = AnalyzerResult(
            sorted([x.data for x in result_tree], key=lambda x: x.start_time),
            None,
            metrics,
            debug_buckets_handler.events if debug_buckets_handler else None,
        )
        return True


"""
20230930 sum of issues which may happen:
- Preparation to Java exam (10:03-10:30) is not separated from other events.
- Few seconds events are counted as activity (like ignore all shorter than 1 minute).
- (?) IDEA activities with both "project" and "filename" fields are counted as separate activities.
- If there was Zoom meeting 15 minutes and no meetings before and after then it probably was a meeting.
    If it matches Outlook event then it is a name for the meeting. "Window" should show Zoom or Slack.
- Cancelled "Pricing scrum meeting" was separated into activity in wrong way.
- IDEA-generated description may be too chatty. Need to cut it somehow.
"""


def merge_activities(
    activities_by_strategy: List[ActivitiesByStrategy],
    steps: List[AnalyzerStep],
    ignore_substrings: List[str],
) -> AnalyzerResult:
    """
    Merges activities into `AnalyzerResult` with the given steps.
    """

    context = {"activities_by_strategy": activities_by_strategy}
    for step in steps:
        metrics = Metrics({})
        step.check_context(context)
        step_description = step.get_description()
        LOG.info("STEP START: %s", step_description)
        time = datetime.datetime.now()
        step_result = step.run(context, metrics)
        time = datetime.datetime.now() - time
        if step_result:
            metrics_strings = list(metrics.to_strings(ignore_with_substrings=ignore_substrings))
            if metrics_strings:
                LOG.info("STEP FINISH: %s\n%s\n", time, "\n".join(metrics_strings))
            else:
                LOG.info("STEP FINISH: %s", time)
        else:
            metrics_strings = list(metrics.to_strings())
            LOG.error("STEP FAILED: %s\n%s", time, "\n".join(metrics_strings))
            return None
    return context["analyzer_result"]
