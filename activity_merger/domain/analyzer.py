import collections
import datetime
from typing import Dict, List, Optional, Set

import intervaltree

from ..config.config import CURRENT_TIMEZONE, DEBUG_BUCKET_PREFIX, LIMIT_OF_RESULTING_ACTIVITIES, LOG, MIN_DURATION_SEC
from .input_entities import IntervalBoundaries, Event, Strategy
from .metrics import Metrics
from .output_entities import Activity, AnalyzerResult
from .strategies import BUCKET_AFK_PREFIX, ActivitiesByStrategy, ActivityByStrategy


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
        events=events,
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
        events_count: int,
    ):
        # Note that ActivityWatch UI shows only strings.
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
                    events_count=len(activity.events),
                )
                metrics.incr("debug events in " + debug_bucket_prefix, activity.duration())


def _find_basic_activity_interval(
    candidates_tree: intervaltree.IntervalTree, current_start_point: datetime.datetime, metrics: Metrics
) -> Optional[intervaltree.Interval]:
    """
    Tries to find good "basic activity" in the given candidates tree. Intervals in this tree expected to be:
    - Chopped to don't overlap with "result activities" added so far.
    - Chopped by AFK activities (probably small).
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
    # - with `in_trustable_boundaries` "whole" or "start",
    # - length is equal or more than `MIN_DURATION_SEC` (choose minimal).
    # For this first sort them by the start, next by length (shorter - first).
    candidates_for_ba = sorted(candidates_for_ba, key=lambda x: (x.begin, x.end - x.begin))
    ba_interval = next(
        (
            x
            for x in candidates_for_ba
            if x.begin == current_start_point
            and x.length() >= min_duraiton_timedelta
            and x.data.strategy.in_trustable_boundaries in [IntervalBoundaries.STRICT, IntervalBoundaries.START]
        ),
        None,
    )
    # If there are no such activity then just make BA from the longest activity.
    # Note that all "remained" activities are cut by existing "result" activities and shouldn't overlap with them.
    if ba_interval is None:
        ba_interval = max(candidates_for_ba, key=intervaltree.Interval.length)
        LOG.info(
            "Can't find 'perfect' basic activity after %s and longer than %s. Using as basic longest - %s",
            current_start_point.astimezone(CURRENT_TIMEZONE).strftime("%H:%M:%S"),
            min_duraiton_timedelta,
            ba_interval.data,
        )
        metrics.incr("basic activities assembled from remainings", ba_interval.length().total_seconds())
    else:
        LOG.info(
            "Found 'perfect' basic activity after %s - %s.",
            current_start_point.astimezone(CURRENT_TIMEZONE).strftime("%H:%M:%S"),
            ba_interval.data,
        )
        metrics.incr("basic activities from solid interval", ba_interval.length().total_seconds())
    return ba_interval


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
            continue
        activity = interval.data
        # If BA overlaps activity then just concatenate data from it into BA.
        if ba_interval.overlaps(interval):
            ra_events.extend(activity.events)
            overlapping_activities.append(activity)
            candidates_tree.remove(interval)
            metrics.incr("activities placed completely inside basic activities", interval.length().total_seconds())
            continue
        boundaries = activity.strategy.in_trustable_boundaries
        # Check BA overlaps the start of the activity.
        if ba_interval.contains_point(interval.begin):
            if boundaries == IntervalBoundaries.START:
                # If activity is `in_trustable_boundaries=start` then concatenate data from it into BA.
                ra_events.extend(activity.events)
                overlapping_activities.append(activity)
                candidates_tree.remove(interval)
                metrics.incr("activities absorbed by basic activity at the start", interval.length().total_seconds())
            elif boundaries == IntervalBoundaries.DIM:
                # If activity is `in_trustable_boundaries=whole` then split activity,
                # and concatenate last part with BA. First part is not needed anyway.
                split_activity = _cut_activity_end(activity, ba_interval.end)
                ra_events.extend(split_activity.events)
                overlapping_activities.append(split_activity)
                candidates_tree.remove(interval)
                # Note that tail of the activity may be used for the next basic/candidate activity.
                candidates_tree.addi(ba_interval.end, split_activity.suggested_end_time, split_activity)
                metrics.incr("activities enhancing basic activity by the start", split_activity.duration())
            else:
                # If activity is `in_trustable_boundaries=end` then skip it.
                metrics.incr(
                    "activities unable to enhance basic activity by the start", interval.length().total_seconds()
                )
            continue
        # Check BA overlaps the end of activity.
        if ba_interval.contains_point(interval.end):
            if boundaries == IntervalBoundaries.END:
                # If activity is `in_trustable_boundaries=end` then concatenate data from it into BA.
                ra_events.extend(activity.events)
                overlapping_activities.append(activity)
                candidates_tree.remove(interval)
                metrics.incr("activities absorbed by basic activity at the end", interval.length().total_seconds())
            elif boundaries == IntervalBoundaries.DIM:
                # If activity is `in_trustable_boundaries=whole` then split activity,
                # and concatenate first part with BA.
                split_activity = _cut_activity_start(activity, ba_interval.begin)
                ra_events.extend(split_activity.events)
                overlapping_activities.append(split_activity)
                candidates_tree.remove(interval)
                metrics.incr("activities enhancing basic activity by the end", split_activity.duration())
            else:
                # If activity is `in_trustable_boundaries=start` then skip it.
                metrics.incr(
                    "activities unable to enhance basic activity by the end", interval.length().total_seconds()
                )
            continue
        # If BA itself is placed inside actvity then concatenate middle of it
        # and only if it is `in_trustable_boundaries=whole`.
        if boundaries == IntervalBoundaries.DIM:
            tmp = _cut_activity_start(activity, ba_interval.begin)
            tmp = _cut_activity_end(tmp, ba_interval.end)
            ra_events.extend(tmp.events)
            overlapping_activities.append(tmp)
            candidates_tree.remove(interval)
            # Note that tail of the activity may be used for the next basic/candidate activity.
            candidates_tree.addi(ba_interval.end, tmp.suggested_end_time, tmp)
            metrics.incr("activities enhancing basic activity by the middle", ba_interval.length().total_seconds())
            continue
        # Skip all other cases.
        raise NotImplementedError(
            f"Inner error when {activity} marked as overlapping with basic activity"
            f"{ba_interval} but in reality it is not."
        )

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
    grouped_activities: Dict[str, List[ActivitiesByStrategy]] = collections.defaultdict(list)
    for activity in activities:
        strategy_name = activity.strategy.name
        grouped_activities[strategy_name].append(activity)
        strategy_to_name[strategy_name] = activity.strategy
    # Sort groups to get "out_produces_good_activity_name" strategy activities first.
    sorted_strategies = sorted(strategy_to_name.values(), key=lambda x: x.out_produces_good_activity_name, reverse=True)
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

    def get_description(self) -> str:
        return "Making 'result_tree' from self sufficient activities."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "activities_by_strategy" in context, "Need in 'activities_by_strategy' property"

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
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
                result_tree.addi(activity.start_time, activity.end_time, activity)
                LOG.info("Found 'self-sufficient' activity: %s", activity)
                metrics.incr("self sufficient activities", activity.duration())
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

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        result_tree: intervaltree.IntervalTree = context["result_tree"]
        candidates_tree: intervaltree.IntervalTree = context["candidates_tree"]
        debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}999_activities"  # 999 - to place it last in UI.

        current_start_point: datetime.datetime = candidates_tree.begin()  # Start from leftest/oldest activity.
        while current_start_point and len(result_tree) < LIMIT_OF_RESULTING_ACTIVITIES:
            metrics.incr("iterations to assemble remaining activities")
            # Find "basic activity" to base "result" activity on interval of it.
            ba_interval = _find_basic_activity_interval(candidates_tree, current_start_point, metrics)
            if ba_interval is None:
                break  # No more activities.
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
            # Add RA to debug bucket if need.
            if self.is_add_debug_buckets:
                # Use the only debug bucket here because events should be consequtive.
                debug_buckets_handler.add_event(
                    bucket_id=debug_bucket_prefix,
                    timestamp=ra.start_time,
                    duration=ra.end_time - ra.start_time,  # Use end-start time, not duration by events.
                    description=ra.description,
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


def merge_activities(
    activities_by_strategy: List[ActivitiesByStrategy],
    steps: List[AnalyzerStep],
    ignore_substrings: List[str],
) -> AnalyzerResult:
    context = {"activities_by_strategy": activities_by_strategy}
    for step in steps:
        metrics = Metrics({})
        step.check_context(context)
        step_description = step.get_description()
        time_tmp = datetime.datetime.now()
        LOG.info("STEP START: %s", step_description)
        step_result = step.run(context, metrics)
        time_tmp = datetime.datetime.now() - time_tmp
        if step_result:
            metrics_strings = list(metrics.to_strings(ignore_with_substrings=ignore_substrings))
            LOG.info("STEP FINISH: %s\n%s\n", time_tmp, "\n".join(metrics_strings))
        else:
            metrics_strings = list(metrics.to_strings())
            LOG.error("STEP FAILED: %s\n%s", time_tmp, "\n".join(metrics_strings))
            return None
    return context["analyzer_result"]


def analyze_activities_per_strategy(
    activities_by_strategy: List[ActivitiesByStrategy],
    is_only_good_strategies_for_description: bool,
    is_add_debug_buckets: bool = True,
) -> AnalyzerResult:
    """
    Analyzes activities-by-strategy to produce a result.
    Actually it handles all "out" parameters in `Strategy` entity.
    It extensively uses `IntervalTree` structure and works in the following steps:
    1. Creates table of not-AFK events for `out_only_not_afk=True` support.
    2. Cuts activities from `out_only_not_afk=True` strategies by not-AFK intervals.
    3. Builds "results" tree from `out_self_sufficient=True` activities. Checks for overlapping.
    4. Builds one more "candidates" tree from `out_only_not_afk=True` activities.
    5. Fills "results" tree gaps with "candidates" tree intervals.
       For each gap or just "interval start" it tries to find "base activity" for the some interval
       and adds everything overlapping into it with chopping activities if need.
       Name of such "result activity" is assembled from the all activities inside.
       TODO improve this last step
    """
    """
    Investigation results and thoughts:
    20230921 investigation about 20230120 data:
    1. Outlook activity "Java Exam" matches good.
       It includes PPT-6447, Slack, libreoffice-writer, zoom, Firefox "by density" activities. 'pipeline-tools' project(s) in IDEA.
       IDEA - wrongly! (it was after the meeting but there is small overlap)
    2. Outlook "Plan Team Standup" disappeared - swallowed by more wide (started earlier and finished later)
       "Code, Gnome-terminal, Slack, firefox, flameshot, jetbrains-idea, thunderbird, zoom application(s).
       'pipeline-tools' project(s) in IDEA."
    3. Outlook "Pre-story meeting" disappeared in right way.
       But here was (?) Slack call with Sergii Iurchenko and it disappeared. Even not found by "Window" bucket.
    4. Preparation to Java exam (10:03-10:30) is not separated from other events because Window watcher "title" field is not used?
    5. Jira events are chopped too hard by AFK. Sometimes they even disappear (11:21-12:03).
    6. Jira events for the same jira_id need to merge, like near 9:35.
    7. Use IDEA events only if there is window event for this.
    8. Use Firefox events only if there is window event for this.

    Ways to improve:
    - [questionable] Don't count few seconds events as activity (like ignore all shorter than 1 minute).
      WON'T work because even small activities (from Window) may add information into activity description.
    - [imprtant] Add "out_only_if_window_app" to strategies (similar to out_only_not_afk).
    - Don't cut Jira events by AFK but cut by out_only_if_window_app (configuration).
      WON'T work because it is even stricter than AFK.
      In result 11:21-12:03 is fixed but there are no activities after 16:01 BUT should.
    - Don't cut IDEA events (configuration).
    - [optional] Investigate why Jira events disappear.

    20230921 investigation about 20230120 data:
    1. Result activities are bad - only 1 on 5+ hours.
    2. IDEA activities become better but don't end in a time, often are cut but 1-2 minutes.
    3. Window activity on 5+ hours becomes "ba" for bad results and wasn't cut to 2 hours - bug!
    4. If "window" shows "browser" and "browser" switches tabs then it is active work in browser.
       If it is the same tab in browser then it is probably the same web app.
    5. Preparation to Java exam (10:03-10:30) is not separated from other events
       because Window watcher "title" field is not used.
    6. If there was Zoom meeting 15 minutes and no meetings before and after then it probably was a meeting.
       If it matches Outlook event then it is a name for the meeting. "Window" should show Zoom or Slack.
       AFK is probably "afk" here due to idle.
    7. If "window" shows "browser" and "browser" switches tabs then it is active work in browser.
       If it is the same tab in browser then it is probably the same web app.

    Exact logic:
    # Take all remained intervals and using `in_trustable_boundaries` value make activites with logic:
    #   - Make tree of remained intervals by cutting out all activities overlapping `result_tree` intervals.
    #   - Iterate this tree to find activities with at least MIN_DURATION_SEC duration and try to make them minimal.
    #     In details at start and at the middle of the gap:
    #       - choose shortest activity with `in_trustable_boundaries` "whole" or "start"
    #         and duration >= MIN_DURATION_SEC, name it "basic activity" (BA)
    #       - find overlapping activities (OA-s)
    #           - if OA-s is smaller than BA then merge it into BA as is;
    #           - if OA is bigger than BA then check if `in_trustable_boundaries` value allows to cut it
    #               - if `in_trustable_boundaries` value allows then cut and merge overlapping part into BA;
    #               - if `in_trustable_boundaries` value is against then cut until BA end
    #       - Sort longest actvities by `out_activity_name` and make name
    #     At the end of the gap (when length of all activities is <= MIN_DURATION_SEC):
    #       - take first longest activity with `in_trustable_boundaries` "whole" or "start", name it BA
    #       - find OA-s and merge with the same logic
    #       - Sort longest actvities by `out_activity_name` and make name
    """

    # # Make activities in "sure that activity" order.
    # # In this way "remained gaps" will shape activities for which data is unclear.
    # metrics = Metrics({})
    # debug_dict: Dict[str, List[Event]] = {}
    # debug_buckets_cnt = 1

    # # 1. Find AFK strategy. It is required for `out_only_not_afk` handling.
    # LOG.info("Searching not-AFK intervals.")
    # not_afk_tree = intervaltree.IntervalTree()
    # for strategy_result in activities_by_strategy:
    #     bucket_prefix = strategy_result.strategy.bucket_prefix
    #     if not bucket_prefix.startswith(BUCKET_AFK_PREFIX):
    #         continue
    #     metrics.incr("AFK strategies")
    #     if strategy_result.strategy.in_activities_may_overlap:
    #         LOG.warning(
    #             "Unsupported setup for %s* strategy - in_activities_may_overlap=True."
    #             "Skipping any AFK-related logic populated by this strategy.",
    #             bucket_prefix,
    #         )
    #         continue
    #     # Add to not_afk_tree only not-AFK activities. Expect that they are not intersect.
    #     for activity in strategy_result.activities:
    #         status = activity.events[0].data["status"]
    #         if status == "not-afk":
    #             not_afk_tree.addi(activity.suggested_start_time, activity.suggested_end_time, activity)
    #             metrics.incr("not-afk intervals", activity.duration)
    #         else:
    #             metrics.incr("afk intervals", activity.duration)

    # # 2. Cut activities from `out_only_not_afk=True` strategies.
    # # Note that some "should be in not-afk" events may be produced as started before AFK watcher started
    # # (for example IDEA events when computer in hibernate mode).
    # LOG.info("Chopping activities by AFK intervals.")
    # for strategy_result in activities_by_strategy:
    #     if not strategy_result.strategy.out_only_not_afk:
    #         continue
    #     metrics.incr("strategies to cut by AFK")
    #     strategy_result.activities = _include_tree_intervals(
    #         strategy_result.activities,
    #         strategy_result.strategy.in_trustable_boundaries,
    #         not_afk_tree,
    #         metrics,
    #         "not-AFK",
    #     )
    #     if is_add_debug_buckets:
    #         debug_buckets_cnt = _add_debug_events_to_not_overlap(
    #             activites=strategy_result.activities,
    #             debug_dict=debug_dict,
    #             bucket_name=strategy_result.strategy.bucket_prefix,
    #             metrics=metrics,
    #             debug_buckets_cnt=debug_buckets_cnt,
    #         )

    # # 3. Add into result activities from `out_self_sufficient=True` strategies. Check for overlappings.
    # LOG.info('Adding "out_self_sufficient" strategies activities.')
    # debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}{debug_buckets_cnt:03}_self_sufficient"
    # result_tree = intervaltree.IntervalTree()
    # for strategy_result in activities_by_strategy:
    #     if not strategy_result.strategy.out_self_sufficient:
    #         continue
    #     metrics.incr("self sufficient strategies")
    #     for activity in strategy_result.activities:
    #         overlap: List[intervaltree.Interval] = result_tree.overlap(activity.start_time, activity.end_time)
    #         if overlap:
    #             raise ValueError(
    #                 f"Overlapping activities from {strategy_result.strategy}: "
    #                 f"{activity} is overlapping with " + ", ".join(x.data for x in overlap)
    #             )
    #         result_tree.addi(activity.start_time, activity.end_time, activity)
    #         LOG.info("Found 'self-sufficient' activity: %s", activity)
    #         metrics.incr("self sufficient activities", activity.duration)
    #         if is_add_debug_buckets:
    #             _add_debug_event(
    #                 debug_dict=debug_dict,
    #                 bucket_id=debug_bucket_prefix,
    #                 timestamp=activity.start_time,
    #                 duration=activity.end_time - activity.start_time,  # Use end-start time, not duration by events.
    #                 description=activity.description,
    #                 events_count=len(activity.events),
    #             )
    # # Check that at least something was added to the result tree. Otherwise no debug events were added.
    # if len(result_tree) > 0:
    #     debug_buckets_cnt += 1

    # # 4. Make tree from remained activities. Chop them by existing `result` activities if there are such.
    # candidates_tree = intervaltree.IntervalTree()
    # if len(result_tree) > 0:
    #     LOG.info('Building "candidate" activities by chopping remaining activities by activities added so far.')
    #     for strategy_result in activities_by_strategy:
    #         strategy = strategy_result.strategy
    #         bucket_prefix = strategy.bucket_prefix
    #         # Skip already handled/contributed strategies.
    #         if bucket_prefix.startswith(BUCKET_AFK_PREFIX) or strategy.out_self_sufficient:
    #             continue
    #         # Cut activities from strategy by result tree. Do it strictly, i.e. boundaries=whole.
    #         remained_activities = _exclude_tree_intervals(
    #             strategy_result.activities, result_tree, metrics, "activities built from self sufficient strategies"
    #         )
    #         # Iterate remained activities and put them into common candidates tree and into per-strategy tree if need.
    #         # Note that activities here are not in order!
    #         for activity in remained_activities:
    #             candidates_tree.addi(activity.suggested_start_time, activity.suggested_end_time, activity)
    #         # Add debug events and buckets if need.
    #         if is_add_debug_buckets:
    #             debug_buckets_cnt = _add_debug_events_to_not_overlap(
    #                 remained_activities, debug_dict, strategy.bucket_prefix, debug_buckets_cnt, metrics
    #             )
    # else:
    #     LOG.info(
    #         'Skipping chopping of remained activities because there were no "out_self_sufficient"'
    #         " strategies activities found this day."
    #     )
    #     for strategy_result in activities_by_strategy:
    #         # Don't use AFK activities to build resulting activities on.
    #         if strategy_result.strategy.bucket_prefix.startswith(BUCKET_AFK_PREFIX):
    #             continue
    #         for activity in strategy_result.activities:
    #             candidates_tree.addi(activity.suggested_start_time, activity.suggested_end_time, activity)

    # # 5. Iterate remained activities to fill `result` remained gaps.
    # LOG.info("Assemble activities-by-strategies into result activities.")
    # current_start_point: datetime.datetime = candidates_tree.begin()  # Start from leftest/oldest activity.
    # debug_bucket_prefix = f"{DEBUG_BUCKET_PREFIX}999_activities"  # 999 - to place it last in UI.

    # # Limit number of iterations to avoid infinite loops on bugs or wrong input.
    # while current_start_point and len(result_tree) < LIMIT_OF_RESULTING_ACTIVITIES:
    #     metrics.incr("iterations to assemble remaining activities")
    #     # Find "basic activity" to base "result" activity on interval of it.
    #     ba_interval = _find_basic_activity_interval(candidates_tree, current_start_point, metrics)
    #     if ba_interval is None:
    #         break  # No more activities.
    #     # Find all overlapping activities and make new `result` activity (RA).
    #     ra = _build_result_activity(ba_interval, candidates_tree, is_only_good_strategies_for_description, metrics)
    #     # Check RA doesn't overlaps with existing result activities at the end.
    #     result_tree_overlapped_with_ra_end: Set[intervaltree.Interval] = result_tree.at(ra.end_time)
    #     # If we had interval in `result_tree` when added RA then we need to search next gap.
    #     # Note that `result_tree_overlapped_with_ra_end` may contain few intervals not in order.
    #     for existing_interval in result_tree_overlapped_with_ra_end:
    #         if existing_interval.begin < ra.end_time:
    #             # Chop end of resulting activity if it overlaps with already existing iterval in `result_tree`.
    #             ra.end_time = existing_interval.begin
    #             ra.events = [x for x in ra.events if x.timestamp < ra.end_time]
    #             metrics.incr(
    #                 "result activities shrinked because it overlaps by end with alredy existing",
    #                 (ra.end_time - existing_interval.begin).total_seconds(),
    #             )
    #     # Add RA into the result tree.
    #     result_tree.addi(ra.start_time, ra.end_time, ra)
    #     # Add RA to debug bucket if need.
    #     if is_add_debug_buckets:
    #         # Use the only debug bucket here because events should be consequtive.
    #         _add_debug_event(
    #             debug_dict=debug_dict,
    #             bucket_id=debug_bucket_prefix,
    #             timestamp=ra.start_time,
    #             duration=ra.end_time - ra.start_time,  # Use end-start time, not duration by events.
    #             description=ra.description,
    #             events_count=len(ra.events),
    #         )
    #     # Configure next iteration.
    #     current_start_point = ra.end_time
    # return AnalyzerResult(
    #     sorted([x.data for x in result_tree], key=lambda x: x.start_time),
    #     None,
    #     metrics,
    #     debug_dict,
    # )
    return None
