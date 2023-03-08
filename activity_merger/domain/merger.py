import datetime
import logging
from typing import List, Dict, Tuple
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
    LOG.info("In result got %d intervals. Details:\n  %s", intervals_count, "\n  ".join(metrics_strings))


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
            LOG.info("Applying '%s' bucket %s events.", bucket_id, len(raw_events))
            # Note that some watchers (like IDEA watcher from few windows) makes events covering each other,
            # i.e. not adjacent. But on each focus change it do generates new event.
            # So first sort all events and cut to make adjacent in scope of a bucket.
            raw_events.sort(key=lambda e: e.timestamp)
            prev_event = None
            events = []  # Convert all events into inner named tuple with more fields.
            for event in raw_events:  # Note that input events are mutable, but `Event` is immutable.
                if prev_event:
                    if prev_event.timestamp + prev_event.duration > event.timestamp:
                        prev_event.duration = event.timestamp - prev_event.timestamp
                    events.append(Event(bucket_id, prev_event.timestamp, prev_event.duration, prev_event.data))
                prev_event = event
            # Add last event.
            events.append(Event(bucket_id, prev_event.timestamp, prev_event.duration, prev_event.data))
            # Handle events.
            cur_interval, metrics = apply_events(events, cur_interval, tolerance)
            # FYI: [x.to_str(debug=True) for x in cur_interval.get_range(offset=-100000, num=3)]
            print_metrics(metrics, cur_interval.get_count())
            buckets_cnt += 1
            check_and_print_intervals(cur_interval, buckets_cnt, bucket_id, logging.WARN)
        else:
            LOG.info("'%s' bucket doesn't have events in %s..%s.", bucket_id, start_time, end_time)
    return cur_interval
