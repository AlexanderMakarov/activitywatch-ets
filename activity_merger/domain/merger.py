import datetime
import logging
from typing import List, Dict
from .interval import Interval
from .input_entities import Event
from ..helpers.helpers import event_to_str
from ..config.config import LOG


def _span_event_down(event: Event, interval: Interval, tolerance: datetime.timedelta) -> Interval:
    """
    Recursively applies event down on `Interval`'s linked list starting after given interval.
    In other words it appends specified event to all `Intervals` later when event covers them and slaces last
    `Interval` on 2 if event ends inside it. 
    :param event: Event to apply.
    :param interval: Interval to start apply after.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :return: Interval where given event ended.
    """
    # Recursive implementation makes stack overflow.
    while interval.next:
        interval = interval.next
        ends_diff = interval.compare_with_time(event.timestamp + event.duration, tolerance, False)
        if ends_diff == 0:
            interval.events.append(event)
            return interval
        elif ends_diff < 0:
            interval = interval.separate_new_at_start(event, tolerance)
            return interval
        else:
            interval.events.append(event)
    return interval


def apply_events(events: List[Event], interval: Interval, tolerance: datetime.timedelta) -> Dict[str, int]:
    """
    Applies events to the specified linked list of intervals with appending events to `Interval`-s and slicing if need.
    Doesn't expands boundaries of `Interval`-s list because assumes that list is built from "afk" and "watchdog"
    watchers events, i.e. events out of bounds of existing intervals should be ignored.
    :param events: List of events to apply.
    :param interval: Node of linked list to apply events onto. Out parameter.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :return: Dictionary of metrics which allows to estimate contrubution of given bunch of events.
    """
    # Metrics to measure each bucket importance.
    metrics = {
        'cnt_skipped_too_short': 0,
        'cnt_skipped_before_afk': 0,
        'cnt_skipped_after_afk': 0,
        'cnt_handled': 0,
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
        # Compare closest interval boundaries with event boundaries and update intervals linked list.
        # Following cases are possible:
        #   [-----]      <- existing interval boundaries (like 'afk')
        # ----------------------------------------------------------- events
        # []|     |       <- 1. skip as 'out of boundaries'
        # [---]   |       <- 2. split interval with new on the start
        # [-------]       <- 3. just add event to interval
        # [---------]     <- 4. span event on few intervals
        #   [--]  |        <- 5. split interval with new on the start
        #   [-----]        <- 6. just add event to interval
        #   [-------]      <- 7. span event on few intervals
        #   | [-] |         <- 8. split interval on 3
        #   | [---]         <- 9. split interval with new on the end
        #   | [-----]       <- 10. split current interval with new on the end and span event on few intervals
        #   |     [--]       <- 11. skip as 'out of boundaries'
        #   |     |  [---]   <- 12. skip as 'out of boundaries'
        event_end = event.timestamp + event.duration
        # Find closest interval started before event start time.
        interval = interval.find_closest(event.timestamp, tolerance, by_end_time=False)
        # Declare some points to compare.
        interval_end_minus_event_start = interval.compare_with_time(event.timestamp, tolerance, False)
        interval_start_minus_event_end = interval.compare_with_time(event_end, tolerance, True)
        # 1) If event completely before closest interval then it is "out of AFK" - skip it.
        if interval_start_minus_event_end < 0:
            LOG.debug("  Skipping %s event happened before all 'AFK' events "
                        "- user didn't work those time.", event_to_str(event))
            metrics['cnt_skipped_before_afk'] += 1
            continue
        # 11, 12) If event is after closest interval then it is "out of AFK" - skip it.
        if interval_end_minus_event_start >= 0:
            LOG.debug("  Skipping %s event happened after/out of 'AFK' events "
                        "- user didn't work those time.", event_to_str(event))
            metrics['cnt_skipped_after_afk'] += 1
            continue
        starts_diff = interval.compare_with_time(event.timestamp, tolerance, True)
        ends_diff = interval.compare_with_time(event_end, tolerance, False)
        if starts_diff == 0:
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
                interval = _span_event_down(event, interval, tolerance)
                metrics['cnt_split_few_intervals'] += 1
        elif starts_diff > 0:
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
                interval = _span_event_down(event, interval, tolerance)
                metrics['cnt_split_few_intervals'] += 1
        else:
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
                interval = _span_event_down(event, interval, tolerance)
                metrics['cnt_split_few_intervals'] += 1
        metrics['cnt_handled'] += 1
    return metrics


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
            raise ValueError("After '%s' bucket events applied greater than %d events number appeared in %s."\
                             " Surroundings: %s"
                             % (last_bucket_name, max_events_per_interval, interval,
                                interval.intervals_to_string(-2, 4, True)))
        last_bucket_id = None
        for event in interval.events:
            if event.bucket_id == last_bucket_id:
                raise ValueError("After '%s' bucket events applied few events from same bucket appeared in %s."\
                                 " Surroundings: %s"
                                 % (last_bucket_name, last_bucket_id, interval,
                                    interval.intervals_to_string(-2, 4, True)))
            last_bucket_id = event.bucket_id
        intervals.append(interval)

    interval.iterate_next(check_and_print)
    if level >= logging.WARN:
        intervals_str = str(len(intervals)) + " intervals"
    elif level == logging.INFO:
        intervals_str = str(len(intervals)) + " intervals:\n  " + "\n  ".join(x.to_str(False) for x in intervals)
    else:
        intervals_str = str(len(intervals)) + " intervals:\n  " + "\n  ".join(x.to_str(True) for x in intervals)
    LOG.info("After '%s' handling got %s", last_bucket_name, intervals_str)


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

    # 1) Handle "aw-watcher-afk" bucket(s) events first to build base intervals to determine activity by.
    # "aw-watcher-afk" - active watcher, it doesn't show "active" when user is not.
    # But is doesn't catch mic on meetings so may produce wrong AFK status.
    # data={status: Optional[afk, not-afk]} It make sense to convert other watchers events into activities within
    # "not-afk" periods and ocasionally add activities from other watchers like "meetings" or "out of comp activity".
    afk_buckets = [x for x in bucket_ids_to_handle if x.startswith("aw-watcher-afk")]
    # If we have AFK events from few computers then merge them together.
    events: List = []
    for bucket_id in afk_buckets:
        bucket_events = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
        if bucket_events:
            events.extend(bucket_events)
            buckets_cnt += 1
        bucket_ids_to_handle.remove(bucket_id)
    events.sort(key=lambda e: e.timestamp)
    events = [Event(bucket_id, x.timestamp, x.duration, x.data) for x in events]
    for event in events:
        d: datetime.timedelta = event.duration
        if d.total_seconds() <= 0:
            LOG.info("JFYI: Skipped 0-duration AFK event at %s in '%s' bucket.", event_to_str(event), event.bucket_id)
            continue
        if not cur_interval:
            cur_interval = Interval(event.timestamp, event.timestamp + event.duration)
            cur_interval.events.append(event)
        else:
            interval_to_put_after = cur_interval.find_closest(event.timestamp, tolerance, by_end_time=True)
            if interval_to_put_after:
                # Doesn't use `Interval.new_after` because it doesn't use event start time i.e. doesn't
                # allow gaps in intervals linked list.
                interval = Interval(event.timestamp, event.timestamp + event.duration, cur_interval, None)
                interval.events.append(event)
                cur_interval.set_next(interval)
                cur_interval = interval
            else:
                raise AssertionError("Closest previous interval wasn't found during 'AFK' bucket events handling"
                                     f"on {event_to_str(event)}.")
    if buckets_cnt <= 0:
        LOG.info("No AFK buckets found. Stopping here - no more events expected.")
        return None
    if cur_interval:
        check_and_print_intervals(cur_interval, buckets_cnt, "AFK")
    else:
        LOG.info("No AFK events found in %s..%s. Stopping here - no more events expected.", start_time, end_time)
        return None

    # 2) "aw-stopwatch" bucket events have highest priority. They may cover cases where 'AFK' events are absent.
    # Need to create intervals splitting and appending base 'AFK' intervals. Note that bucket is empty in most cases.
    # "aw-stopwatch" - active watcher, it is managed by user directly.
    # Make sense measuring duration per unique label from "running=true" to "running=false".
    # data={label: str, running: bool}
    bucket_id = "aw-stopwatch"
    if bucket_id in bucket_ids_to_handle:
        events: List[object] = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
        if events:
            events = [Event(bucket_id, x.id, x.timestamp, x.duration, x.data) for x in events]
            for event in events:  # TODO implement and put before AFK events handling.
                if cur_interval:
                    # Assume events are sorted here.
                    interval = cur_interval.new_after(event)
                else:
                    interval = Interval(event.timestamp, event.timestamp + event.duration)
                    interval.events.append(event)
                cur_interval = interval
            buckets_cnt += 1
            check_and_print_intervals(cur_interval, buckets_cnt, "stopwatch", True)
        else:
            LOG.info("No stopwatch events found.")
        bucket_ids_to_handle.remove(bucket_id)

    # 3) Iterate through remained buckets by given rules.
    # Assume that they are all "passive" watchers reacting on 3d-party application events which leads to:
    # - Events may have bounds out of 'AFK' events therefore don't represent user activity (like "aw-idea").
    # - Event start means some user activity, but event end may be just some timeout or "computer waken up".
    # - Events may have not enough data to describe state (like "aw-window-watcher" behavior on Linux Wayland).
    for bucket_id in bucket_ids_to_handle:
        raw_events = activity_watch_client.get_events(bucket_id, start=start_time, end=end_time)
        if raw_events:
            LOG.info(f"Applying '{bucket_id}' {len(raw_events)} events with.")
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
            events.append(prev_event)  # Add last event.
            # Handle events.
            metrics = apply_events(events, cur_interval, tolerance)
            metrics_strings = (f"{k}: {v}" for k, v in metrics.items())
            LOG.info("In result got %d intervals. Details:\n  %s", cur_interval.get_count(),
                        "\n  ".join(metrics_strings))
            buckets_cnt += 1
            check_and_print_intervals(cur_interval, buckets_cnt, bucket_id, logging.WARN)
        else:
            LOG.info("'%s' bucket doesn't have events in %s..%s.", bucket_id, start_time, end_time)
    return cur_interval
