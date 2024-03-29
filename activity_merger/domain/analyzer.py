import collections
import datetime
from operator import attrgetter
from typing import Dict, List, Optional, Set, Tuple

import intervaltree

from activity_merger.domain.basic_interval_finder import BIFinder
from activity_merger.helpers.event_helpers import activity_by_strategy_to_str, event_data_to_str
from activity_merger.helpers.helpers import from_start_to_end_to_str

from ..config.config import DEBUG_BUCKET_PREFIX, LIMIT_OF_RESULTING_ACTIVITIES, LOG, MAX_ACTIVITY_DURATION_SEC
from .input_entities import Event, IntervalBoundaries, Strategy
from .metrics import Metrics
from .output_entities import Activity, AnalyzerResult
from .strategies import BUCKET_AFK_PREFIX, ActivityByStrategy, StrategyApplyResult, calculate_interval_density


RA_DEBUG_BUCKET_NAME = f"{DEBUG_BUCKET_PREFIX}999_activities"  # 999 - to place it last in UI.


def _cut_activity_start(
    activity: ActivityByStrategy, point: datetime.datetime, new_id: int = None
) -> ActivityByStrategy:
    """
    Cuts start from activity (suggested and 'max_start_time' if need) and returns only tail after specified point.
    """
    events = [x for x in activity.events if (x.timestamp + x.duration) > point]  # Keep "border" event.
    density_zero = activity.density == 0  # If density was zero before cutoff then it will remain same.
    return ActivityByStrategy(
        id=activity.id if new_id is None else new_id,
        suggested_start_time=point,
        suggested_end_time=activity.suggested_end_time,
        max_start_time=max(point, activity.max_start_time),
        min_end_time=activity.min_end_time,
        events=events,
        density=0 if density_zero else calculate_interval_density(events, point, activity.suggested_end_time),
        grouping_data=activity.grouping_data,
        strategy=activity.strategy,
    )


def _cut_activity_end(activity: ActivityByStrategy, point: datetime.datetime, new_id: int = None) -> ActivityByStrategy:
    """
    Cuts end from activity (suggested and 'min_end_time' if need) and returns only head before specified point.
    """
    events = [x for x in activity.events if x.timestamp < point]  # Keep "border" event.
    density_zero = activity.density == 0  # If density was zero before cutoff then it will remain same.
    return ActivityByStrategy(
        id=activity.id if new_id is None else new_id,
        suggested_start_time=activity.suggested_start_time,
        suggested_end_time=point,
        max_start_time=activity.max_start_time,
        min_end_time=min(point, activity.min_end_time),
        events=events,
        density=0 if density_zero else calculate_interval_density(events, activity.suggested_start_time, point),
        grouping_data=activity.grouping_data,
        strategy=activity.strategy,
    )


def _split_activity(
    activity: ActivityByStrategy, start_point: datetime.datetime, end_point: datetime.datetime, new_id: int
) -> List[ActivityByStrategy]:
    """
    Cuts some middle part from the activity to get 2 new 'ActivityByStrategy'-s:
    - from the start of initial activity to the 'start_point',
    - from the 'end_point' to the end of initial activity.
    """
    return [_cut_activity_end(activity, start_point), _cut_activity_start(activity, end_point, new_id)]


def _exclude_tree_intervals(
    activities: List[ActivityByStrategy],
    tree: intervaltree.IntervalTree,
    id_for_new: int,
    metrics: Metrics,
    name_of_tree: str,
) -> Tuple[List[ActivityByStrategy], int]:
    """
    Cuts different parts of activities which overlap with intervals in the specified tree.
    Note that for metrics are used bounds of activities, not inner event intervals sum.
    :param activities: List of activities to cut tree intervals from.
    :param tree: Tree with intervals to cut activities by.
    :param id_for_new: Identifier with auto-increment for new activities.
    :param metrics: Metrics instance.
    :param name_of_tree: Name of the tree for metrics.
    :return: Tuple with list of chopped activities and last new activity ID assigned.
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
        # Sort intervals to put new activities ID-s in deterministic order.
        intervals = sorted(intervals)
        # Handle activities one by one. In result activity would be either skipped or modified.
        # In "modified" case need to search overlaps again.
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
                raise NotImplementedError(
                    f"Inner error with {interval} actually not overlapping with"
                    f" {activity_by_strategy_to_str(activity)}."
                )
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
                raise NotImplementedError(
                    f"Inner error with {interval} actually not overlapping with"
                    f" {activity_by_strategy_to_str(activity)}."
                )
        else:
            # Interval is placed in the middle of the current activity.
            id_for_new += 1
            split_activities = _split_activity(activity, interval.begin, interval.end, id_for_new)
            # Due to interval was random, both parts of the initial activity should be checked again.
            activities_queue.extend(split_activities)
            # Update metric without duration because very often activity is cut few times and resulting
            # cumulative duration is bigger than initial activity duration.
            metrics.incr("activities with middle cut by " + name_of_tree)
            continue
    # Sort result because parts of long activities may be placed into it randomly.
    return (
        sorted(result, key=lambda x: x.suggested_start_time),
        id_for_new,
    )


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
                f"activities not completely covered by {name_of_tree}",
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
        """
        Builds a new debug bucket.
        """
        result = f"{DEBUG_BUCKET_PREFIX}{self.cnt:03}_{bucket_name}"
        self.cnt += 1
        return result

    def add_debug_event(
        self,
        bucket_id: str,
        debug_event_id: int,
        timestamp: datetime.datetime,
        duration: datetime.timedelta,
        description: str,
        aw_events: List[Event],
        density: float,
    ):
        """
        Adds a new event into the specified debug bucket.
        :param bucket_id: Name of the bucket to add new event to.
        :param debug_event_id: ID of the event. Should be unique.
        :param timestamp: Event start.
        :param duration: Event duration.
        :param description: Event description.
        :param aw_events: List of ActivityWatch events this event consists of.
        :param density: Event density based on the ActivityWatch events it consists of.
        """
        events = ", ".join(from_start_to_end_to_str(x.timestamp, x.timestamp + x.duration) for x in aw_events)
        events_cnt = str(len(aw_events))  # Note that ActivityWatch UI shows only strings
        if density:
            data = {
                "id": str(debug_event_id),
                "desc": description,
                "density": f"{density:.2f}",  # Density is more important than events information, should be earlier.
                "events_cnt": events_cnt,
                "events": events,
            }
        else:
            data = {"id": str(debug_event_id), "desc": description, "events_cnt": events_cnt, "events": events}
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
                self.add_debug_event(
                    bucket_id=debug_bucket_prefix,
                    debug_event_id=activity.id,
                    timestamp=activity.suggested_start_time,
                    # For debug events need to use end-start time, not duration by events.
                    duration=activity.suggested_end_time - activity.suggested_start_time,
                    # Build description in easiest way.
                    description=", ".join(f"{k}={v}" for k, v in activity.grouping_data.get_data().items()),
                    aw_events=activity.events,
                    density=activity.density,
                )
                metrics.incr("debug events in " + debug_bucket_prefix, activity.duration())


def find_next_uncovered_intervals(
    candidates_tree: intervaltree.IntervalTree,
    result_tree: intervaltree.IntervalTree,
    start_point: datetime.datetime = None,
) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Finds closest to "start_point" interval in "candidates_tree" which is not covered by "result_tree".
    :param candidates_tree: Tree of intervals to choose from.
    :param result_tree: Tree of intervals which are forbidden.
    :param start_point: Point to start to search interval since.
    :return: Tuple with next start and end points. When there is no more suitable interval returns None-s.
    """
    candidates_begin = candidates_tree.begin()
    candidates_end = candidates_tree.end()

    # If start_point is None, take the begin of the candidates_tree.
    # Also check that in candidates_tree range, otherwise there are no more intervals possible.
    if start_point is None:
        start_point = candidates_begin
    elif start_point > candidates_end:
        return (None, None)

    # Take out start_point under result_tree by shifting it to the end of overlapping interval in result_tree.
    while result_tree.overlaps_point(start_point) and start_point < candidates_end:
        overlapping_intervals = sorted(result_tree[start_point])
        # Set start_point as the end of the last overlapping interval.
        start_point = overlapping_intervals[-1].end

    # Check if we reached the end of the candidates_tree without finding an uncovered interval/gap.
    if start_point >= candidates_end:
        return (None, None)

    # Set end_point. Need to check each interval in result_tree.
    end_point = candidates_end
    # If there's a result interval starting after our current start_point and before our current end_point
    # we update the end_point to the beginning of that result interval.
    for interval in result_tree[start_point:end_point]:
        if start_point < interval.begin < end_point:
            end_point = interval.begin
    return (start_point, end_point)


def _extend_events_from_activitybs(events: List, activitybs: ActivityByStrategy, interval: intervaltree.Interval):
    for event in activitybs.events:
        if interval.overlaps(event.timestamp, event.timestamp + event.duration):
            events.append(event)


def fit_interval_to_tree_to_dont_overlap(
    tree: intervaltree.IntervalTree, interval: intervaltree.Interval
) -> Optional[intervaltree.Interval]:
    """
    Cuts or removes given interval if it overlaps with the given tree.
    :param tree: Tree with intervals to don't overlap with.
    :param interval: Interval to fit.
    :returns: Interval with chopped edges or `None` if it overlaps completely.
    """
    overlapping_intervals = tree.overlap(interval.begin, interval.end)
    sorted_overlapping_intervals = sorted(overlapping_intervals, key=lambda x: x.end)

    new_begin = interval.begin
    new_end = interval.end

    for existing_interval in sorted_overlapping_intervals:
        # If the new interval starts before the existing interval, check if there's enough space
        # before the existing interval starts.
        if new_begin < existing_interval.begin:
            new_end = min(new_end, existing_interval.begin)
            break
        # Adjust the new interval's start to be after the existing interval's end.
        new_begin = max(new_begin, existing_interval.end)

    # Check if the new interval is valid.
    if new_begin < new_end:
        return intervaltree.Interval(new_begin, new_end, interval.data)
    else:
        return None


def build_result_activity(
    basic_interval: intervaltree.Interval,
    candidates_tree: intervaltree.IntervalTree,
    bi_description: Optional[str],
    is_only_good_strategies_for_description: bool,
    metrics: Metrics,
) -> Activity:
    """
    Builds "result" activity.
    :param basic_interval: "Base" interval to build activity on.
    :param candidates_tree: Tree of candidates to add parts of overlapped activity-by-strategy-es into
    "result" activity.
    :param bi_description: First entry for activity description.
    :param is_only_good_strategies_for_description: Flat to use for "result" activity description
    only activity-by-strategy-es created by strategies marked as producing good description.
    :param metrics: Metrics instance.
    :return: "Result" activity.
    """
    # Find all overlapping activities, even including those which was used for "basic interval".
    overlapping_intervals = candidates_tree.overlap(basic_interval.begin, basic_interval.end)
    LOG.info("Basic interval is overlapped by %d 'candidate' activities.", len(overlapping_intervals))
    ra_duration = basic_interval.length().total_seconds()
    ra_events = []
    overlapping_activities: List[ActivityByStrategy] = []
    for interval in overlapping_intervals:
        activitybs: ActivityByStrategy = interval.data
        boundaries = activitybs.strategy.in_trustable_boundaries
        # Check "absolute" cases first - completely add or completely skip.
        if basic_interval.contains_interval(interval):
            # If base interal overlaps activity completely then just concatenate data from it into result.
            ra_events.extend(activitybs.events)  # All events in interval for sure.
            overlapping_activities.append(activitybs)
            metrics.incr(
                "activities added completely because fits into basic interval", interval.length().total_seconds()
            )
            continue
        elif boundaries == IntervalBoundaries.STRICT:
            # If activity declared with STRICT boundaries and border is overlapped then skip activity completely.
            LOG.info(
                "Excluding from results activity with %s boundaries because it overlaps border of 'base interval': %s",
                boundaries,
                activity_by_strategy_to_str(activitybs),
            )
            metrics.incr(
                f"activities with {boundaries} boundaries skipped completely from result",
                interval.length().total_seconds(),
            )
            continue
        # 2. Handle case when base interal overlaps the start of the activity (not IntervalBoundaries.STRICT here).
        if basic_interval.contains_point(interval.begin):
            if boundaries == IntervalBoundaries.START:
                # Add the whole activity to the result.
                _extend_events_from_activitybs(ra_events, activitybs, basic_interval)
                overlapping_activities.append(activitybs)
                metrics.incr(
                    f"activities with {boundaries} boundaries added completely into result",
                    interval.length().total_seconds(),
                )
            elif boundaries == IntervalBoundaries.DIM:
                # Add only start of the activity to the result.
                split_activity = _cut_activity_end(activitybs, basic_interval.end)
                ra_events.extend(split_activity.events)  # All events in interval for sure.
                overlapping_activities.append(split_activity)
                metrics.incr("activities with only start added into result", split_activity.duration())
            else:
                # Skip activity completely.
                metrics.incr(
                    f"activities with {boundaries} boundaries skipped completely from result",
                    interval.length().total_seconds(),
                )
            continue
        # 3. Handle case when base interval overlaps the end of activity (not IntervalBoundaries.STRICT here).
        if basic_interval.contains_point(interval.end):
            if boundaries == IntervalBoundaries.END:
                # Add the whole activity to the result.
                _extend_events_from_activitybs(ra_events, activitybs, basic_interval)
                overlapping_activities.append(activitybs)
                metrics.incr(
                    f"activities with {boundaries} boundaries added completely into result",
                    interval.length().total_seconds(),
                )
            elif boundaries == IntervalBoundaries.DIM:
                # Add only end of the activity to the result.
                split_activity = _cut_activity_start(activitybs, basic_interval.begin)
                ra_events.extend(split_activity.events)  # All events in interval for sure.
                overlapping_activities.append(split_activity)
                metrics.incr("activities with only end added into result", split_activity.duration())
            else:
                # Skip activity completely.
                metrics.incr(
                    f"activities with {boundaries} boundaries skipped completely from result",
                    interval.length().total_seconds(),
                )
            continue
        # 4 Handle case when base interval itself is placed inside actvity.
        if boundaries == IntervalBoundaries.DIM:
            tmp = _cut_activity_start(activitybs, basic_interval.begin)
            tmp = _cut_activity_end(tmp, basic_interval.end)
            ra_events.extend(tmp.events)  # All events in interval for sure.
            overlapping_activities.append(tmp)
            metrics.incr("activities with only middle added into result", ra_duration)
            continue
        else:
            # Otherwise BA is in the middle of activity with START or END boundaries.
            metrics.incr(
                f"activities with {boundaries} boundaries skipped completely from result",
                interval.length().total_seconds(),
            )
            continue

    # Build raw RA, i.e. with "not sure end". Fill it with all the events from the "enhancing" activities.
    name = _build_activity_name(
        bi_description, overlapping_activities, metrics, ra_duration, is_only_good_strategies_for_description
    )
    metrics.incr("result activities", ra_duration)
    return Activity(
        basic_interval.begin,
        basic_interval.end,
        ra_events,
        name,
    )


def _build_activity_name(
    description_first_entry: Optional[str],
    activitiesbs: List[ActivityByStrategy],
    metrics: Metrics,
    duration_sec: float,
    is_only_good_strategies_for_description: bool,
) -> str:
    """
    Builds name of the resulting activity from list of `ActivityByStrategy`-s.
    In details all activities are grouped by strategy, sorted by "out_produces_good_activity_name"
    to get "good" names first or the only (depending on the settings), next processed to call
    "out_activity_name_sentence_builder" with key-value pairs received from related `GroupingDescriptor`-s.
    Handles "out_activity_name_sentence_builder" absence with default key-value pairs stringifier and
    with adding nothing if "out_activity_name_sentence_builder" returns empty value.
    :param description_first_entry: Optional first entry in the description.
    :param activitiesbs: List of `ActivityByStrategy` making resulting activity.
    :param metrics: Metrics object to report details into.
    :param duration_sec: Duration of the resulting activity.
    :param is_only_good_strategies_for_description: Flag to build activity description only from "good" ones.
    :return: Resulting activity description.
    """
    # Group activities by Strategy. Keep map of strategies to name. Name is a key in both dictionaries.
    strategy_to_name: Dict[str, Strategy] = {}
    grouped_activitiesbs: Dict[str, List[ActivityByStrategy]] = collections.defaultdict(list)
    for activitybs in activitiesbs:
        strategy_name = activitybs.strategy.name
        grouped_activitiesbs[strategy_name].append(activitybs)
        strategy_to_name[strategy_name] = activitybs.strategy
    # Sort groups to get "out_produces_good_activity_name" strategy and longest activities first.
    sorted_strategies = sorted(
        strategy_to_name.values(),
        key=lambda x: (
            x.out_produces_good_activity_name,
            sum(a.duration() * a.density for a in grouped_activitiesbs[x.name]),
        ),
        reverse=True,
    )
    # Get sorted list of activity-by-strategy-es with "longest strategies" first and
    # in each group "longest activities" first.
    sorted_grouped_activities = [
        sorted(grouped_activitiesbs[x.name], key=lambda x: x.duration() * x.density, reverse=True)
        for x in sorted_strategies
    ]
    # Build a list of descriptions.
    resulting_names: List[str] = []
    if description_first_entry:
        resulting_names.append(description_first_entry)
    # Process each group of activities to build sentence.
    is_contained_activity_with_good_name = False
    for grouped_activity_list in sorted_grouped_activities:
        # Note that mixing all of the a-b-s 'get_data' results into the one dictionary would cause data loosing.
        # For example if 2 Windows activities with {app=Slack, title=Meeting} and {app=Code, title=MyProject} are
        # grouped here then resulting dictionary would look like {app=[Slack, Code], title=[Meeting, MyProject]}.
        # Use the strategy's out_activity_name_builder to get the name, or use a default logic.
        strategy = strategy_to_name[grouped_activity_list[0].strategy.name]
        if strategy.out_produces_good_activity_name:
            is_contained_activity_with_good_name = True
        elif is_only_good_strategies_for_description:
            # If strategy doesn't produce good activity name but it is required then skip.
            break
        name_builder = (
            strategy.out_activity_name_sentence_builder
            if strategy.out_activity_name_sentence_builder
            else lambda x: ", ".join([f"{k}={v}" for k, v in x.items()])
        )
        groups_data = [x.grouping_data.get_data() for x in grouped_activity_list]
        generated_name = name_builder(groups_data)
        if generated_name:
            resulting_names.append(generated_name)
    if is_contained_activity_with_good_name:
        metrics.incr("result activities with good name", duration_sec)
    metrics.incr(f"result activities with name combined from {len(sorted_grouped_activities)} strategies", duration_sec)
    return " ".join(resulting_names)


def add_activity_to_result_tree(
    ra: Activity,
    result_tree: intervaltree.IntervalTree,
    is_add_debug_buckets: bool,
    debug_buckets_handler: Optional[DebugBucketsHandler],
):
    """
    Verifies that given activity doesn't overlap with existing intervals in "result_tree",
    adds it into "result_tree" and debug buckets if need.
    :param ra: Activity to add.
    :param result_tree: Interval tree to add activity into.
    :is_add_debug_buckets: Flag to add debug event based on activity.
    :param debug_buckets_handler: Debug buckets handler.
    """
    overlaps = result_tree.overlap(ra.start_time, ra.end_time)
    if overlaps:
        overlaps_str = "".join("\n- " + str(x.data) for x in overlaps)
        raise ValueError(f"Wrong interval for result activity {ra}\nIt overlaps with:{overlaps_str}")
    result_tree.addi(ra.start_time, ra.end_time, ra)
    LOG.info("Added into 'result tree' activity: %s", ra)
    # Add RA to debug bucket if need.
    if is_add_debug_buckets:
        # Use the only debug bucket here because events should be consequtive.
        debug_buckets_handler.add_debug_event(
            bucket_id=RA_DEBUG_BUCKET_NAME,
            debug_event_id=0,  # Don't set ID for events which shouldn't be pointed.
            timestamp=ra.start_time,
            duration=ra.end_time - ra.start_time,  # Use end-start time, not duration by events.
            description=ra.description,
            aw_events=ra.events,
            density=0,  # Don't set density for "resulting" activities.
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

    def check_context(self, context: Dict[str, any]) -> None:  # TODO (impr) consider to use contextvars package.
        """
        Checks that all required items are present in the context.
        Raises an exception if the context is invalid. Expected to be executed before `run` method.
        By-default checks nothing.
        """

    def run(self, context: Dict[str, any], metrics: Metrics):
        """
        Executes this step on the given context. All results are placed back to the context.
        By-default does nothing.
        """
        raise NotImplementedError("AnalyzerStep.run is not implemented.")


class MakeResultTreeFromSelfSufficientActivitiesStep(AnalyzerStep):
    """
    Makes "result_tree" `IntervalTree` from "out_self_sufficient" activities.
    Handles multiple strategies at once.
    """

    def __init__(
        self,
        is_add_debug_buckets: bool = False,
        is_only_good_strategies_for_description: bool = True,
    ):
        super(MakeResultTreeFromSelfSufficientActivitiesStep, self).__init__()
        self.is_add_debug_buckets = is_add_debug_buckets
        self.is_only_good_strategies_for_description = is_only_good_strategies_for_description

    def get_description(self) -> str:
        return "Making 'result_tree' from self sufficient activities."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "strategy_apply_result" in context, "Need in 'strategy_apply_result' property"
        if self.is_add_debug_buckets:
            if "debug_buckets_handler" not in context:
                context["debug_buckets_handler"] = DebugBucketsHandler()

    def _make_ra(self, group: List, metrics: Metrics) -> Activity:
        # Find activity with top priority strategy.
        top_item = max(group, key=lambda x: x.data["strategy"].out_self_sufficient_interval_rank)
        top_activity = top_item.data["activity"]
        top_interval = intervaltree.Interval(
            top_activity.suggested_start_time, top_activity.suggested_end_time, top_activity
        )
        # Combine events from all activities.
        all_activities = [x.data["activity"] for x in group]
        all_events: List[Event] = []
        for activitybs in all_activities:
            _extend_events_from_activitybs(all_events, activitybs, top_interval)
        all_events = sorted(all_events, key=lambda x: x.timestamp)
        # Build name of the activity.
        # Don't make priority for "out_self_sufficient_interval_rank" because it is only about interval!
        duration = top_activity.duration()
        name = _build_activity_name(
            None, all_activities, metrics, duration, self.is_only_good_strategies_for_description
        )
        metrics.incr("result activities", duration)
        ra = Activity(
            top_interval.begin,
            top_interval.end,
            all_events,
            name,
        )
        return ra

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        data: List[StrategyApplyResult] = context["strategy_apply_result"]
        # Filter only self-sufficient strategies.
        data = [x for x in data if x.strategy.out_self_sufficient]
        # Prepare candidates tree. We would change activities right inside it.
        candidates_tree = intervaltree.IntervalTree()

        # Add all activities to the candidates_tree with their strategy.
        for strategy_result in data:
            for activity in strategy_result.activities:
                start = activity.suggested_start_time.timestamp()
                end = activity.suggested_end_time.timestamp()
                candidates_tree[start:end] = {"strategy": strategy_result.strategy, "activity": activity}
            if self.is_add_debug_buckets:
                debug_buckets_handler.add_debug_events_to_not_overlap(
                    activites=strategy_result.activities,
                    bucket_suffix=strategy_result.strategy.name.replace(" ", "-") + "_candidate",
                    metrics=metrics,
                )

        # Group overlapping activities.
        overlapping_groups = []
        for interval in candidates_tree:
            current_group = candidates_tree[interval.begin : interval.end]
            if current_group not in overlapping_groups:
                overlapping_groups.append(current_group)

        # Process each group of overlapping activities.
        result_tree = intervaltree.IntervalTree()
        for group in overlapping_groups:
            # TODO if 2 Zoom meetings happens during Outlook meeting need measure interval only for meeting
            # overlapping start of Outlook meeting. Add this logic by strategy boundaries.
            metrics.incr(f"intervals with {len(group)} overlapping self sufficient activities")
            ra = self._make_ra(group, metrics)
            add_activity_to_result_tree(
                ra=ra,
                result_tree=result_tree,
                is_add_debug_buckets=self.is_add_debug_buckets,
                debug_buckets_handler=debug_buckets_handler,
            )
            metrics.incr("self sufficient activities", ra.duration.total_seconds())
        context["result_tree"] = result_tree


class ChopActivitiesByResultTreeStep(AnalyzerStep):
    """
    Cuts activites by "result_tree" `IntervalTree` intervals.
    :param is_skip_afk: Flag to don't chop AFK activities by current 'result_tree'.
    :param is_skip_self_sufficient_strategies: Flag to don't chop activities from "self sufficient"
    strategies by current 'result_tree'.
    """

    def __init__(self, is_skip_afk: bool = False, is_skip_self_sufficient_strategies: bool = True):
        super(ChopActivitiesByResultTreeStep, self).__init__()
        self.is_skip_afk = is_skip_afk
        self.is_skip_self_sufficient_strategies = is_skip_self_sufficient_strategies

    def get_description(self) -> str:
        return "Chopping 'candidates_tree' from all not self sufficient activities."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "strategy_apply_result" in context, "Need in 'strategy_apply_result' property"
        assert "result_tree" in context, "Need in 'result_tree' property"

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        result_tree: intervaltree.IntervalTree = context.get("result_tree")
        last_id: int = context.get("last_id")
        if len(result_tree) > 0:
            strategy_result: StrategyApplyResult
            for strategy_result in context["strategy_apply_result"]:
                bucket_prefix = strategy_result.strategy.bucket_prefix
                # Skip AFK activities and self sufficient if need.
                if (self.is_skip_afk and bucket_prefix.startswith(BUCKET_AFK_PREFIX)) or (
                    self.is_skip_self_sufficient_strategies and strategy_result.strategy.out_self_sufficient
                ):
                    continue
                # Cut activities from strategy by result tree. Do it strictly, i.e. boundaries=whole.
                strategy_result.activities, last_id = _exclude_tree_intervals(
                    strategy_result.activities, result_tree, last_id, metrics, "result_tree"
                )


class MakeCandidatesTreeStep(AnalyzerStep):
    """
    Makes "candidates_tree" `IntervalTree` with provided filters to get many overlappling intervals from many
    strategies in one tree.
    :param is_add_debug_buckets: Flag to build events into "debug" buckets.
    :param is_add_afk: Flag to add AFK activities into "result_tree".
    :param is_add_self_sufficient: Flag to add activities from "self sufficient" strategies into "result_tree".
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
        assert "strategy_apply_result" in context, "Need in 'strategy_apply_result' property"
        if self.is_add_debug_buckets:
            if "debug_buckets_handler" not in context:
                context["debug_buckets_handler"] = DebugBucketsHandler()

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        last_id: int = context.get("last_id")
        candidates_tree = intervaltree.IntervalTree()
        strategy_result: StrategyApplyResult
        for strategy_result in context["strategy_apply_result"]:
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
                    bucket_suffix=strategy_result.strategy.name.replace(" ", "-") + "_candidate",
                    metrics=metrics,
                )
        metrics.override("last activity id", last_id, 0)
        context["candidates_tree"] = candidates_tree


class MergeCandidatesTreeIntoResultTreeWithBIFinderStep(AnalyzerStep):
    """
    Makes "candidates_tree" `IntervalTree`.
    :param bi_finder: `BIFinder` instance to use for choosing "base interval" on each step.
    :param is_add_debug_buckets: Flag to build events into "debug" buckets.
    :param is_only_good_strategies_for_description: Flag to use only activities from "good" strategies to
    build description of result activities.
    """

    def __init__(
        self,
        bi_finder: BIFinder,
        is_add_debug_buckets: bool = False,
        is_only_good_strategies_for_description: bool = True,
    ):
        self.is_only_good_strategies_for_description = is_only_good_strategies_for_description
        self.is_add_debug_buckets = is_add_debug_buckets
        self.bi_finder = bi_finder

    def get_description(self) -> str:
        return f"Merging 'candidates_tree' into 'result_tree' with {self.bi_finder}."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "result_tree" in context, "Need in 'result_tree' property"
        assert "candidates_tree" in context, "Need in 'candidates_tree' property"
        if self.is_add_debug_buckets:
            if "debug_buckets_handler" not in context:
                context["debug_buckets_handler"] = DebugBucketsHandler()

    def try_convert_basic_interval_to_ra(
        self,
        interval: intervaltree.Interval,
        bi_score: float,
        bi_description: Optional[str],
        closest_candidate_score: float,
        candidates_tree: intervaltree.IntervalTree,
        result_tree: intervaltree.IntervalTree,
        metrics: Metrics,
        debug_buckets_handler: DebugBucketsHandler,
    ) -> Optional[Activity]:
        """
        Tries to build a new `Activity` based on the provided interval with provided data (description and overlapping
        items in 'candidates_tree'), adds it into 'result_tree' and debug buckets (if need).
        """
        # Provide metric and log about new base interval found.
        bi_score_description = self.bi_finder.score_to_desc(bi_score)
        metrics.incr(
            f"base intervals with {bi_score_description} score", (interval.end - interval.begin).total_seconds()
        )
        actvity = (
            activity_by_strategy_to_str(interval.data)
            if isinstance(interval.data, ActivityByStrategy)
            else str(interval.data)
        )
        LOG.info(
            "Found 'base interval' with %.2f '%s' score on %s with data: %s",
            bi_score,
            bi_score_description,
            from_start_to_end_to_str(interval.begin, interval.end),
            actvity,
        )
        # Provide metric about "how distinguishable basic interval was".
        interval_duration = (interval.end - interval.begin).total_seconds()
        if closest_candidate_score is not None:
            distance_desc = self.bi_finder.score_to_desc(bi_score - closest_candidate_score)
            metrics.incr(f"base intervals with {distance_desc} distance from other candidates", interval_duration)
        else:
            metrics.incr("base intervals with single activity-by-strategy on interval", interval_duration)
        # Check RA doesn't overlaps with existing result activities.
        fit_interval = fit_interval_to_tree_to_dont_overlap(result_tree, interval)
        if fit_interval is None:
            metrics.incr("base intervals which completely overlap with existing result activities", interval_duration)
            return None
        diff_with_fit = interval_duration - (fit_interval.end - fit_interval.begin).total_seconds()
        if diff_with_fit > 0:
            metrics.incr(
                "base intervals which were cut due to partially overlap with existing result activities", diff_with_fit
            )
        # Make new `result` activity (RA) and add into the result tree.
        ra = build_result_activity(
            fit_interval, candidates_tree, bi_description, self.is_only_good_strategies_for_description, metrics
        )
        add_activity_to_result_tree(
            ra=ra,
            result_tree=result_tree,
            is_add_debug_buckets=self.is_add_debug_buckets,
            debug_buckets_handler=debug_buckets_handler,
        )
        return ra

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        result_tree: intervaltree.IntervalTree = context["result_tree"]
        candidates_tree: intervaltree.IntervalTree = context["candidates_tree"]

        # Iterate through candidates tree and try to fill up gaps in result tree with intervals from here.
        # Note that very often results tree will be empty and need to make up all activities from candidates tree.
        current_start_point: datetime.datetime
        current_end_point: datetime.datetime
        current_start_point, current_end_point = find_next_uncovered_intervals(
            candidates_tree=candidates_tree, result_tree=result_tree
        )
        while current_start_point and len(result_tree) < LIMIT_OF_RESULTING_ACTIVITIES:
            metrics.incr("iterations to assemble remaining activities")
            # Find all candidates which overlap "interval to make activity in" somehow.
            candidates: List[intervaltree.Interval] = list(
                candidates_tree.overlap(current_start_point, current_end_point)
            )
            if not candidates:
                # If no more activities are possible the stop loop.
                break
            bi = candidates[0]
            bi_score = 1.0
            closest_candidate_score = None
            # Check if more than 1 candidate is available (most common case).
            if len(candidates) > 1:
                # Find "base interval" to base "result activity" on.
                bi, bi_score, closest_candidate_score, bi_description = self.bi_finder.find_top(
                    candidates, current_start_point, current_end_point, MAX_ACTIVITY_DURATION_SEC, metrics
                )
            # Save basic interval end before trying to build result activity - it may be shrunk or even disappear
            # but we need to start next iteration from the place where current iteration is over.
            bi_end = bi.end
            self.try_convert_basic_interval_to_ra(
                interval=bi,
                bi_score=bi_score,
                bi_description=bi_description,
                closest_candidate_score=closest_candidate_score,
                candidates_tree=candidates_tree,
                result_tree=result_tree,
                metrics=metrics,
                debug_buckets_handler=debug_buckets_handler,
            )
            # Configure next iteration.
            current_start_point, current_end_point = find_next_uncovered_intervals(
                candidates_tree=candidates_tree,
                result_tree=result_tree,
                start_point=bi_end,
            )
        context["analyzer_result"] = AnalyzerResult(
            sorted([x.data for x in result_tree], key=lambda x: x.start_time),
            None,
            None,  # Don't put metrics here to avoid double printing of them.
            debug_buckets_handler.events if debug_buckets_handler else None,
        )


def aggregate_strategies_results_to_activities(
    strategy_apply_results: List[StrategyApplyResult],
    steps: List[AnalyzerStep],
    ignore_substrings: List[str] = None,
) -> AnalyzerResult:
    """
    Merges `StrategyApplyResult`-s into `AnalyzerResult` (which includes aggregated activities) by the given steps.
    :param strategy_apply_results: List of events aggregated by strategies.
    :param steps: List of operations to apply on `StrategyApplyResult`-s.
    :param ignore_sugstrings: List of things to suppress in logs.
    :return: Result of analysing events which includes aggregated activities, metrics and debug events.
    """
    context = {
        "strategy_apply_result": strategy_apply_results,
        "last_id": max(x.last_id for x in strategy_apply_results),
    }
    for step in steps:
        metrics = Metrics({})
        step.check_context(context)
        step_description = step.get_description()
        LOG.info("STEP START: %s", step_description)
        time = datetime.datetime.now()
        step.run(context, metrics)
        time = datetime.datetime.now() - time
        metrics_strings = list(metrics.to_strings(ignore_with_substrings=ignore_substrings))
        if metrics_strings:
            LOG.info("STEP FINISH: %s, metrics:\n%s", time, "\n".join(metrics_strings))
        else:
            LOG.info("STEP FINISH: %s, no metrics", time)
    return context["analyzer_result"]
