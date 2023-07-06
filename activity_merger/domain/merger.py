import datetime
import logging
from typing import List, Dict, Tuple
from aw_core import Event as AWEvent

from .metrics import Metrics

from .strategies import ActivitiesByStrategy, Strategy, StrategyHandler
from .interval import Interval
from .input_entities import Event
from ..helpers.helpers import event_to_str
from ..config.config import LOG


def _span_event_down(event: Event, interval: Interval, tolerance: datetime.timedelta, is_make_intervals: bool)\
        -> Interval:
    """
    Recursively applies event down on `Interval`'s linked list starting after given interval.
    In other words it appends specified event to all `Intervals` later when event covers them and slaces last
    `Interval` on 2 if event ends inside it. 
    :param event: Event to apply.
    :param interval: Interval to start apply after.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :param is_make_intervals: Flag to make new intervals if there are no existing covering event's time span.
    :return: Interval where given event ended.
    """
    # Note that recursive implementation causes stack overflow.
    while interval.next:
        interval = interval.next
        ends_diff = interval.compare_with_time(event.timestamp + event.duration, tolerance, is_start=False)
        if ends_diff == 0:  # Event end matches interval end.
            interval.events.append(event)
            return interval
        elif ends_diff < 0:  # Event end within interval.
            interval = interval.separate_new_at_start(event, tolerance)
            return interval
        else:  # Event end after the interval.
            interval.events.append(event)
    # If we need to make new intervals and event doesn't end within last interval then append interval in the end.
    if is_make_intervals and interval.next is None:
        ends_diff = interval.compare_with_time(event.timestamp + event.duration, tolerance, is_start=False)
        if ends_diff > 0:
            tmp = Interval(interval.end_time, event.timestamp + event.duration, interval, None)
            tmp.events.append(event)
            interval = tmp
    return interval


def apply_events(events: List[Event], interval: Interval, tolerance: datetime.timedelta,
        is_make_intervals: bool = False, is_fail_on_overlap: bool = False) -> Tuple[Interval, Dict[str, int]]:
    """
    Applies events to the specified linked list of intervals with appending events to `Interval`-s and splitting
    them on parts if need.
    May work in mode when new `Interval`-s are created and when all events should fit into existing intervals
    time bounds. In both cases splits existing `Interval`-s on borders of each event if need.
    Also may be instructed to fail when new event overlaps existing interval - it is useful for high priority
    events, like for Stopwatch - they shouldn't intersect by nature.
    :param events: List of events to apply.
    :param interval: Node of linked list to apply events onto. Out parameter.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :param is_make_intervals: Flag to make new intervals if there are no existing covering event's time span.
    :param is_fail_on_overlap: Flag to fail if there are intervals overlaping event's time span.
    :return: Dictionary of metrics which allows to estimate contrubution of given bunch of events.
    """
    # Metrics to measure each bucket importance.
    metrics = {
        'cnt_new_interval': 0,
        'cnt_skipped_too_short': 0,
        'cnt_skipped_before_afk': 0,
        'cnt_skipped_after_afk': 0,
        'cnt_handled_events': 0,
        'cnt_match_interval': 0,
        'cnt_inside_interval': 0,
        'cnt_split_one_interval': 0,
        'cnt_split_few_intervals': 0,
    }
    for event in events:
        # Skip too short events.
        if event.duration < tolerance:
            metrics['cnt_skipped_too_short'] += 1
            continue
        # Basing on `Interval.find_closest` behavior - it returns interval with the given time inside or on the start.
        # Compare closest interval boundaries with event boundaries and update intervals linked list.
        # Following cases are possible (excluding long StopWatch events, see 'test_mergerer.py' TO DO-s):
        #    [-----]      <- existing interval boundaries
        # ----------------------------------------------------------- event relation of position cases:
        # [] |     |       <- 1. skip as 'out of boundaries' or make an interval
        # [--]     |       <- 1. skip as 'out of boundaries' or make an interval
        # [---]    |       <- 2. split interval with new on the start and optionaly add new interval at start
        # [--------]       <- 3. just add event to interval and optionaly add new interval before
        # [----------]     <- 4. span event on current and next intervals or add one at the end
        #    [--]  |         <- 5. split interval with new on the start
        #    [-----]         <- 6. just add event to interval
        #    [-------]       <- 7. span event on few intervals and optionaly add new one
        #    | [-] |         <- 8. split interval on 3
        #    | [---]         <- 9. split interval with new on the end
        #    | [-----]       <- 10. split current interval with new on the end and span event on few intervals
        #    |     [--]      <- 11. span event on intervals later
        #    |     |  [---]  <- 12. skip as 'out of boundaries' or add new interval
        event_end = event.timestamp + event.duration
        # Make new interval if it is the first event.
        if is_make_intervals and interval is None:
            interval = Interval(event.timestamp, event.timestamp + event.duration)
            interval.events.append(event)
            metrics['cnt_new_interval'] += 1
            metrics['cnt_handled_events'] += 1
            continue
        # Find closest interval started before event start time.
        interval = interval.find_closest(event.timestamp, tolerance)
        # Declare some points to compare.
        interval_end_minus_event_start = interval.compare_with_time(event.timestamp, tolerance, is_start=False)
        interval_start_minus_event_end = interval.compare_with_time(event_end, tolerance, is_start=True)
        # 1) If event completely before closest interval then skip or add new interval.
        if interval_start_minus_event_end <= 0:
            if is_make_intervals:
                tmp = Interval(event.timestamp, event.timestamp + event.duration, None, interval)
                tmp.events.append(event)
                interval = tmp
                metrics['cnt_new_interval'] += 1
                metrics['cnt_handled_events'] += 1
            else:
                LOG.debug("  Skipping %s event happened before all 'AFK' events "
                            "- user didn't work that time.", event_to_str(event))
                metrics['cnt_skipped_before_afk'] += 1
            continue
        # 11, 12) If event is completely after closest interval then it is "out of AFK" - skip it or make new.
        if interval_end_minus_event_start >= 0:
            if is_make_intervals:
                tmp = Interval(event.timestamp, event.timestamp + event.duration, interval, None)
                tmp.events.append(event)
                interval = tmp
                metrics['cnt_new_interval'] += 1
                metrics['cnt_handled_events'] += 1
            else:
                LOG.debug("  Skipping %s event happened after/out of 'AFK' events "
                            "- user didn't work that time.", event_to_str(event))
                metrics['cnt_skipped_after_afk'] += 1
            continue
        starts_diff = interval.compare_with_time(event.timestamp, tolerance, True)
        ends_diff = interval.compare_with_time(event_end, tolerance, False)
        if starts_diff == 0:
            if is_fail_on_overlap:
                raise ValueError(f"After '{event_to_str(event)}' overlaps with {interval.to_str(debug=True)}."
                                 " It is not supported for the current bucket type.")
            if ends_diff == 0:
                # 6) Event matches interval.
                interval.events.append(event)
                metrics['cnt_match_interval'] += 1
            elif ends_diff < 0:
                # 5) Event start matches interval and event ends inside it.
                interval = interval.separate_new_at_start(event, tolerance)
                metrics['cnt_split_one_interval'] += 1
            else:
                # 7) Event start matches interval and event ends after it.
                interval.events.append(event)
                interval = _span_event_down(event, interval, tolerance, is_make_intervals)
                metrics['cnt_split_few_intervals'] += 1
        elif starts_diff > 0:
            if is_fail_on_overlap:
                raise ValueError(f"After '{event_to_str(event)}' overlaps with {interval.to_str(debug=True)}."
                                 " It is not supported for the current bucket type.")
            if ends_diff == 0:
                # 9) Event starts inside interval and ends matches.
                interval = interval.separate_new_at_end(event, tolerance)
                metrics['cnt_split_one_interval'] += 1
            elif ends_diff < 0:
                # 8) Event starts and ends inside interval.
                interval = interval.separate_new_at_middle(event, tolerance)
                metrics['cnt_inside_interval'] += 1
            else:
                # 10) Event starts inside interval and ends after it.
                interval = interval.separate_new_at_end(event, tolerance)
                interval = _span_event_down(event, interval, tolerance, is_make_intervals)
                metrics['cnt_split_few_intervals'] += 1
        else:
            if is_fail_on_overlap:
                raise ValueError(f"After '{event_to_str(event)}' overlaps with {interval.to_str(debug=True)}."
                                 " It is not supported for the current bucket type.")
            if ends_diff == 0:
                # 3) Event started before "AFK" and ends exactly on first interval end.
                interval.events.append(event)
                metrics['cnt_match_interval'] += 1
            elif ends_diff < 0:
                # 2) Event started before "AFK" and ends inside first interval.
                interval = interval.separate_new_at_start(event, tolerance)
                metrics['cnt_split_one_interval'] += 1
            else:
                # 4) Event started before "AFK" and ends after first interval.
                interval.events.append(event)
                interval = _span_event_down(event, interval, tolerance, is_make_intervals)
                metrics['cnt_split_few_intervals'] += 1
        metrics['cnt_handled_events'] += 1
    return (interval, metrics)


def check_and_print_intervals(interval: Interval, max_events_per_interval: int, last_bucket_name: str,
                              level: int = logging.INFO) -> None:
    """
    Performs checks for validity of resulting intervals and prints results for manual checks.
    :param interval: Interval to check linked list from.
    :param max_events_per_interval: Expected max number of events in each interval.
    :param last_bucket_name: Name of the last bucket events were applied from. Used for logs.
    :param level: Logging level representing how to print resulting intervals. 'WARN' and higher - doesn't print
    intervals, 'INFO' print all intervals without debug information, 'DEBUG' and lower - print everything.
    """
    interval = interval.iterate_prev()
    intervals = []

    def check_and_print(interval: Interval) -> bool:
        if len(interval.events) > max_events_per_interval:
            raise ValueError(
                f"After '{last_bucket_name}' bucket events applied greater than {max_events_per_interval} events "\
                f"number appeared in {interval}. Surroundings: {interval.intervals_to_string(-2, 4, False)}"
            )
        last_bucket_id = None
        for event in interval.events:
            if event.bucket_id == last_bucket_id:
                raise ValueError(
                    f"After '{last_bucket_name}' bucket events applied few events from the same '{last_bucket_id}' "\
                    f"bucket appeared in {interval}. Surroundings: {interval.intervals_to_string(-2, 4, False)}"
                )
            last_bucket_id = event.bucket_id
        intervals.append(interval)
        return False  # Never stop.

    interval.iterate_next(check_and_print)
    if level >= logging.WARN:
        intervals_str = str(len(intervals)) + " intervals"
    elif level == logging.INFO:
        intervals_str = str(len(intervals)) + " intervals:\n  " + "\n  ".join(x.to_str(False) for x in intervals)
    else:
        intervals_str = str(len(intervals)) + " intervals:\n  " + "\n  ".join(x.to_str(True) for x in intervals)
    LOG.info("After '%s' handling got %s", last_bucket_name, intervals_str)


def print_metrics(metrics: Dict[str, int], intervals_count: int):
    """
    Prints INFO log with current number of intervals and metrics which values > 0.
    :param metrics: Dictionary of metrics.
    :param intervals_count: Number of intervals measured separately from metrics to put into logs.
    """
    metrics_strings = (f"{k}: {v}" for k, v in metrics.items() if v > 0)
    LOG.info("  In result got %d intervals. Details:\n  %s", intervals_count, "\n  ".join(metrics_strings))


def _sort_and_merge_events(events: List[Event]) -> List[Event]:
    events.sort(key=lambda e: e.timestamp)
    TIMEDELTA_0 = datetime.timedelta()
    result = []
    p = None
    for e in events:
        # Note that AFK sometimes contains "0 timedelta" events which breaks merging logic - omit them.
        if e.duration == TIMEDELTA_0:
            continue
        if p is not None:
            # Note that we want to merge events with the same data and overlapping time spans.
            # But differnt buckets are allowed. Because for AFK it is possible to work on few machines in parallel,
            # and for Stopwatch 'status' is expected to be different anyway.
            if e.timestamp <= (p.timestamp + p.duration) and str(e.data) == str(p.data):
                p = Event(e.bucket_id, p.timestamp,
                          max(e.timestamp + e.duration, p.timestamp + p.duration) - p.timestamp, e.data)
            else:
                result.append(p)
                p = e
        else:
            p = e
    if p is not None:
        result.append(p)
    return result


def report_from_buckets(activity_watch_client, start_time: datetime.datetime, end_time: datetime.datetime,
                        buckets: List[str], tolerance: datetime.timedelta) -> Interval:
    """
    Gets events from specified buckets, prints report by them and returns time-ordered linked list of `Interval`-s.
    :param awc: ActivityWatch client to use.
    :param start_time: Start time for the report.
    :param end_time: End time for the report.
    :param buckets: List of buckets to report events from.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :return: Last touched `Interval` from linked list built from events.
    """
    cur_interval: Interval = None
    bucket_ids_to_handle = list(buckets.keys())
    buckets_cnt = 0
    events: List = []

    # For all events perform following normalization:
    # - Convert to inner lightweight 'Event's with 'bucket_id' attached.
    # - Sort by timestamp to minimize "patches" in intervals.
    # - Merge ajusent or overlaping events with the same 'data'.

    # 1) Handle "aw-stopwatch" bucket events have highest priority. They may cover cases where 'AFK' events are absent.
    # Need to create intervals with failing on intersecting intervals - it is not supported.
    # "aw-stopwatch" - active watcher, it is managed by user directly.
    # Make sense measuring duration per unique label from "running=true" to "running=false".
    # data={label: str, running: bool}
    stopwatch_buckets = [x for x in bucket_ids_to_handle if x.startswith("aw-stopwatch")]
    if stopwatch_buckets:
        for bucket_id in stopwatch_buckets:
            bucket_events: List[object] = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
            if bucket_events:
                events.extend([Event(bucket_id, x.timestamp, x.duration, x.data) for x in bucket_events])
                buckets_cnt += 1
            bucket_ids_to_handle.remove(bucket_id)
    if events:
        events = _sort_and_merge_events(events)
        cur_interval, metrics = apply_events(events, None, tolerance, is_make_intervals=True, is_fail_on_overlap=True)
        print_metrics(metrics, cur_interval.get_count() if cur_interval else 0)
    if cur_interval:
        check_and_print_intervals(cur_interval, buckets_cnt, "Stopwatch")

    # 2) Handle "aw-watcher-afk" bucket(s) events to build base intervals to determine activities by.
    # "aw-watcher-afk" - active watcher, it doesn't show "active" when user is not.
    # But it (on 2023/02) doesn't watch mic on meetings so may produce wrong AFK status.
    # data={status: Optional[afk, not-afk]} It make sense to convert other watchers events into activities within
    # "not-afk" periods and ocasionally add activities from other watchers like "meetings" or "out of comp activity".
    afk_buckets = [x for x in bucket_ids_to_handle if x.startswith("aw-watcher-afk")]
    # If we have AFK events from few computers then merge them together.
    events = []
    for bucket_id in afk_buckets:
        bucket_events = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
        if bucket_events:
            events.extend([Event(bucket_id, x.timestamp, x.duration, x.data) for x in bucket_events])
            buckets_cnt += 1
        bucket_ids_to_handle.remove(bucket_id)
    if events:
        events = _sort_and_merge_events(events)
        cur_interval, metrics = apply_events(events, cur_interval, tolerance, is_make_intervals=True)
        print_metrics(metrics, cur_interval.get_count() if cur_interval else 0)
    if buckets_cnt <= 0:
        LOG.info("No buckets except AFK and/or Stopwatch found. Stopping here - no more events expected.")
        return None
    if cur_interval:
        check_and_print_intervals(cur_interval, buckets_cnt, "AFK")
    else:
        LOG.info("No AFK events found in %s..%s. Stopping here - no more events expected.", start_time, end_time)
        return None

    # 3) Iterate through remained buckets by given rules.
    # Assume that they are all "passive" watchers reacting on 3d-party application events which leads to:
    # - Events may have bounds out of 'AFK' events therefore don't represent user activity (like "aw-idea").
    # - Event start means some user activity, but event end may be just some timeout or "computer waken up".
    # - Events may have not enough data to describe state (like "aw-window-watcher" behavior on Linux with Wayland).
    for bucket_id in bucket_ids_to_handle:
        raw_events = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
        if raw_events:
            LOG.info("Applying '%s' bucket %d events:", bucket_id, len(raw_events))
            # Note that some watchers (like IDEA watcher from few windows/projects) makes events covering each other,
            # i.e. not adjacent, but on each focus change it do generates new event.
            # So first sort all events and cut to make adjacent in scope of a bucket.
            raw_events.sort(key=lambda e: e.timestamp)
            prev_event = None
            events = []  # Convert all events into inner named tuple with more fields.
            for event in raw_events:  # Note that input events are mutable, but `Event` is immutable.
                if event.duration < tolerance:  # Filter out too short events.
                    continue
                if prev_event:
                    if prev_event.timestamp + prev_event.duration > event.timestamp:
                        prev_event.duration = event.timestamp - prev_event.timestamp
                    events.append(Event(bucket_id, prev_event.timestamp, prev_event.duration, prev_event.data))
                prev_event = event
            # Add last event.
            events.append(Event(bucket_id, prev_event.timestamp, prev_event.duration, prev_event.data))
            if len(events) != len(raw_events):
                LOG.info("  Normalized and cleansed %d events into %d.", len(raw_events), len(events))
            # Handle events.
            cur_interval, metrics = apply_events(events, cur_interval, tolerance)
            # FYI: [x.to_str(debug=True) for x in cur_interval.get_range(offset=-100000, num=3)]
            print_metrics(metrics, cur_interval.get_count())
            buckets_cnt += 1
            check_and_print_intervals(cur_interval, buckets_cnt, bucket_id, logging.WARN)
        else:
            LOG.info("'%s' bucket doesn't have events in %s..%s.", bucket_id, start_time, end_time)
    return cur_interval


def _add_raw_event(raw_event: AWEvent, list_to_add: List[Event], bucket_id: str, metrics: Metrics):
    resulting_event = Event(bucket_id, raw_event.timestamp, raw_event.duration, raw_event.data)
    list_to_add.append(resulting_event)
    metrics.incr('events to handle', resulting_event.duration.seconds)


def sort_merge_convert_raw_events(raw_events: List[AWEvent], bucket_id: str, tolerance: datetime.timedelta,
                                  metrics: Metrics) -> List[Event]:
    """
    Sorts raw events in time order, merges events with the same data, cuts "overlapping by tail" events,
    converts them into "domain, with bucket ID" events.
    :param raw_events: Events from ActivityWatcher.
    :param bucket_id: Event's source bucket ID.
    :param tolerance: tolerance for comparing and merging events.
    :returns: List of normalized `Event`-s.
    """

    # 1. Sort events in time order. Make sure that events started in the same time placed in "shorter first" order.
    raw_events.sort(key=lambda e: (e.timestamp, e.duration))
    result = []

    # 2. Iterate events with postponing each previous event.
    prev_event: AWEvent = None
    for event in raw_events:
        metrics.incr('raw events', event.duration.seconds)
        # Filter out too short events.
        if event.duration < tolerance:
            metrics.incr(f'{bucket_id} too short events', event.duration.seconds)
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
            metrics.incr(f'{bucket_id} skipped duplicated by interval events', prev_event.duration)
            prev_event = event
            continue
        start_later_than_prev_end = event.timestamp - prev_end > tolerance
        if same_data and not start_later_than_prev_end:
            # same data, start not later than previous end => merge
            prev_event_duration_before_merge = prev_event.duration.seconds
            prev_event.duration = max(current_end, prev_end) - prev_event.timestamp
            metrics.incr(f'{bucket_id} merged the same data events', prev_event_duration_before_merge)
            continue
        if same_start:
            # other data, same start => remove previous as overlapped
            metrics.incr(f'{bucket_id} skipped shorter and overlapped by interval events', prev_event.duration)
            prev_event = event
            continue
        if not start_later_than_prev_end:
            # start before previous end, other data => cut and save previous, postpone current
            prev_event_duration_before_cut = prev_event.duration
            prev_event.duration = event.timestamp - prev_event.timestamp
            diff_prev_event_duration = prev_event_duration_before_cut - prev_event.duration
            if diff_prev_event_duration > tolerance:
                metrics.incr(f'{bucket_id} cut duration of different overlapping events', diff_prev_event_duration)
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


def analyze_buckets(activity_watch_client, start_time: datetime.datetime, end_time: datetime.datetime,
                    buckets: List[str], strategies: List[Strategy], tolerance: datetime.timedelta)\
                    -> Tuple[List[ActivitiesByStrategy], Metrics]:
    metrics = Metrics({})
    bucket_ids_to_handle = list(buckets.keys())
    metrics.override('total buckets', len(buckets), 0)
    result: List[ActivitiesByStrategy] = []
    for strategy in strategies:
        metrics.incr('total strategies')
        strategy_buckets = [x for x in bucket_ids_to_handle if x.startswith(strategy.bucket_prefix)]
        # Check there is something to handle by this strategy.
        if not strategy_buckets:
            metrics.incr('strategies without buckets')
            LOG.info("%s* strategy: there are no buckets.", strategy.bucket_prefix)
            continue
        # Create containers for the strategy result.
        events: List[Event] = []
        strat_metrics = Metrics({})  # These Metrics are per strategy, not per bucket!
        total_raw_events = 0
        # Normilize events.
        for bucket_id in strategy_buckets:
            metrics.incr('handled buckets')
            strat_metrics.incr('total buckets')
            bucket_events: List[AWEvent] = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
            total_raw_events += len(bucket_events)
            if bucket_events:
                normilized_events = sort_merge_convert_raw_events(bucket_events, bucket_id, tolerance, strat_metrics)
                metrics.incr('not-empty buckets handled')
                strat_metrics.incr('not-empty buckets')
                events.extend(normilized_events)
            bucket_ids_to_handle.remove(bucket_id)
        # Handle case when there are no events for the strategy.
        if not events:
            metrics.incr('strategies without events')
            if total_raw_events:
                # Log strategy handling metrics here because it won't pass it further.
                strat_metrics_str = "\n  ".join(x for x in strat_metrics.to_strings(is_exclude_empty=False))
                LOG.warning("%s* strategy: After normilizing %d events nothing left:\n  %s", strategy.bucket_prefix,
                            total_raw_events, strat_metrics_str)
            else:
                LOG.info("%s* strategy: No events found.", strategy.bucket_prefix)
            continue  # Stop handling this strategy.
        # Handle events with strategy.
        strategy_activities = StrategyHandler.handle_events(strategy, events, strat_metrics)
        result.append(strategy_activities)
        # Calculate common sum of activities.
        for activity in strategy_activities.activities:
            metrics.incr('total activities', activity.duration)
    return result, metrics
