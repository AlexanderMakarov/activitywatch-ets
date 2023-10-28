import datetime
from typing import List, Tuple

import aw_client
from aw_core import Event as AWEvent

from ..config.config import LOG
from .input_entities import Event, Strategy
from .metrics import Metrics
from .strategies import InStrategyPropertiesHandler, StrategyApplyResult


def _add_raw_event(raw_event: AWEvent, list_to_add: List[Event], bucket_id: str, metrics: Metrics):
    # Remove part which is less than a second.
    resulting_event = Event(bucket_id, raw_event.timestamp.replace(microsecond=0), raw_event.duration, raw_event.data)
    list_to_add.append(resulting_event)
    # Note that total duration of "events to handle" may be bigger than total duration of
    # "raw events" for Window and other buckets because of `sort_merge_convert_raw_events` logic - 
    # if between 2 similar events there is a gat then "merged" event becomes bigger on this gap.
    metrics.incr("events to handle", resulting_event.duration.seconds)


def sort_merge_convert_raw_events(
    raw_events: List[AWEvent], bucket_id: str, tolerance: datetime.timedelta, metrics: Metrics
) -> List[Event]:
    """
    Sorts raw `AWEvent`-s in time order, merges events with the same data, cuts "overlapping by tail" events,
    converts them into `Event`-s.
    :param raw_events: Events from ActivityWatcher.
    :param bucket_id: Event's source bucket ID.
    :param tolerance: tolerance for comparing and merging events.
    :returns: List of normalized `Event`-s.
    """

    # 1. Sort events in time order. Make sure that events started in the same time placed in "shorter first" order.
    raw_events.sort(key=lambda e: (e.timestamp, e.duration))
    result = []

    # Probably there is a bug with Window events duration "normalized" > "raw" on 16 seconds.
    # 2. Iterate events with postponing each previous event.
    prev_event: AWEvent = None
    for event in raw_events:
        metrics.incr("raw events", event.duration.seconds)
        # Filter out too short events.
        if event.duration < tolerance:
            # Note that even with default settings duration is always 0 it may count with custom settings.
            metrics.incr(f"{bucket_id} too short events", event.duration.seconds)
            continue
        # If it is first event then just postpone.
        if not prev_event:
            prev_event = event
            continue
        # There are a number of cases how current event intersects with previous:
        # - same start, longer, same data => merge
        # - same start, longer, other data => remove previous as overlapped
        # - same start and end, same data => remove previous as the same
        # - same start and end, other data => remove previous as overlapped
        # - start before previous end, same data => merge
        # - start before previous end, other data => cut and save previous, postpone current
        # - start = previous end, same data => merge
        # - start = previous end, other data => save previous, postpone current
        # - start later than previous end, any data => save previous, postpone current
        # They may be merged into:
        # - same data, same start and end => remove previous as the same
        # - same data, start not later than previous end => merge
        # - other data, same start => remove previous as overlapped
        # - start before previous end, other data => cut and save previous, postpone current
        # - start later than previous end OR start = previous end, other data => save previous, postpone current
        prev_end = prev_event.timestamp + prev_event.duration
        current_end = event.timestamp + event.duration
        same_start = abs(event.timestamp - prev_event.timestamp) < tolerance
        same_end = abs(current_end - prev_end) < tolerance
        same_data = str(event.data) == str(prev_event.data)
        if same_data and same_start and same_end:
            # same data, same start and end => remove previous as the same
            metrics.incr(f"{bucket_id} skipped duplicated by interval events", prev_event.duration.seconds)
            prev_event = event
            continue
        start_later_than_prev_end = event.timestamp - prev_end > tolerance
        if same_data and not start_later_than_prev_end:
            # same data, start not later than previous end => merge
            prev_event_duration_before_merge = prev_event.duration
            prev_event.duration = max(current_end, prev_end) - prev_event.timestamp
            metrics.incr(f"{bucket_id} merged the same data events", prev_event_duration_before_merge.seconds)
            continue
        if same_start:
            # other data, same start => remove previous as overlapped
            metrics.incr(f"{bucket_id} skipped shorter and overlapped by interval events", prev_event.duration.seconds)
            prev_event = event
            continue
        if not start_later_than_prev_end:
            # start before previous end, other data => cut and save previous, postpone current
            prev_event_duration_before_cut = prev_event.duration
            prev_event.duration = event.timestamp - prev_event.timestamp
            diff_prev_event_duration = prev_event_duration_before_cut - prev_event.duration
            if diff_prev_event_duration > tolerance:
                metrics.incr(
                    f"{bucket_id} cut duration of different overlapping events", diff_prev_event_duration.seconds
                )
            _add_raw_event(prev_event, result, bucket_id, metrics)
            prev_event = event
            continue
        # Remains only cases when need to save previous event as is.
        _add_raw_event(prev_event, result, bucket_id, metrics)
        prev_event = event
        continue
    # Add last postponed event as is.
    if prev_event is not None:
        _add_raw_event(prev_event, result, bucket_id, metrics)
    # If in result got different number of events then note about it.
    if len(result) != len(raw_events):
        LOG.info("%s: %d raw events normalized and cleansed into %d.", bucket_id, len(raw_events), len(result))
    return result


def apply_strategies_on_events(
    activity_watch_client: aw_client.ActivityWatchClient,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    buckets: List[str],
    strategies: List[Strategy],
    tolerance: datetime.timedelta,
) -> Tuple[List[StrategyApplyResult], Metrics]:
    """
    Gets ActivityWatch events for specified time duration and applies strategies "in" parameters on them.
    Note that one strategy may either span few buckets if data was taken from the few machines or few strategies
    handle the same bucket if prefixes are the same.
    :param activity_watch_client: ActivityWatch client to use for fetching events.
    :param start_time: Analyzing interval start.
    :param end_time: Analyzing interval end.
    :param buckets: List of buckets to analyze.
    :param strategies: List of strategies to analyze with.
    :param tolerance: Tolerance for comparing and merging events.
    :returns: Tuple with list of `StrategyApplyResult` and metrics.
    """
    metrics = Metrics({})
    bucket_ids_to_handle = list(buckets.keys())
    metrics.override("total buckets", len(buckets), 0)
    strategies_handler = InStrategyPropertiesHandler()
    result: List[StrategyApplyResult] = []
    for strategy in strategies:
        metrics.incr("total strategies")
        strategy_buckets = [x for x in bucket_ids_to_handle if x.startswith(strategy.bucket_prefix)]
        # Check there is something to handle by this strategy.
        if not strategy_buckets:
            metrics.incr("strategies without buckets")
            LOG.info("%s* strategy: there are no buckets.", strategy.bucket_prefix)
            continue
        # Create containers for the strategy result.
        events: List[Event] = []
        strat_metrics = Metrics({})  # These Metrics are per strategy, not per bucket!
        total_raw_events = 0
        # Fetch and normilize events.
        # TODO (performance) do in parallel, AFK and Window before all. Think about activitybs ID-s.
        for bucket_id in strategy_buckets:
            metrics.incr("handled buckets")
            strat_metrics.incr("total buckets")
            bucket_events: List[AWEvent] = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
            total_raw_events += len(bucket_events)
            if bucket_events:
                normilized_events = sort_merge_convert_raw_events(bucket_events, bucket_id, tolerance, strat_metrics)
                metrics.incr("not-empty buckets handled")
                strat_metrics.incr("not-empty buckets")
                events.extend(normilized_events)
        # Handle case when there are no events for the strategy.
        if not events:
            metrics.incr("strategies without events")
            if total_raw_events:
                # Log strategy handling metrics here because it won't pass it further.
                strat_metrics_str = "\n  ".join(x for x in strat_metrics.to_strings(is_exclude_empty=False))
                LOG.warning(
                    "%s* strategy: After normalizing %d events nothing left:\n  %s",
                    strategy.bucket_prefix,
                    total_raw_events,
                    strat_metrics_str,
                )
            else:
                LOG.info("%s* strategy: No events found.", strategy.bucket_prefix)
            continue  # Stop handling this strategy.
        # Handle events within the strategy using stateful handler.
        strategy_activities = strategies_handler.handle_events(strategy, events, strat_metrics)
        result.append(strategy_activities)
        # Calculate common sum of activities. Note that to have count of activities need to add them by one.
        for activity in strategy_activities.activities:
            metrics.incr("total activities", activity.duration())
    return result, metrics
