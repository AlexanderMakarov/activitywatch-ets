#!/usr/bin/env python3
import datetime
import logging
import aw_client
import collections
import dataclasses
from typing import Any, List, Dict, Optional, Tuple, Callable
import re


# Which tolerance to use when comparing events. Mostly from different watchers.
EVENTS_COMPARE_TOLERANCE_SEC = 1
TOLERANCE = datetime.timedelta(0, 1, 0)  # 1 sec
# Timezone to show dates.
CURRENT_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
LOG: logging.Logger = logging.getLogger(__name__)


Event = collections.namedtuple('Event', ['bucket_id', 'timestamp', 'duration', 'data'])
"""
Lightweight representation of ActivityWatcher event without "id" field but with "bucket_it" field to trach source.
"""


def event_data_to_str(event: Event):
    if not event:
        return 'null'
    return str(event.data)


def event_to_str(event: Event):
    return f"{event.timestamp.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{(event.timestamp + event.duration).astimezone(CURRENT_TIMEZONE):%H:%M:%S}("\
           f"{event_data_to_str(event)})"


class Interval:
    """
    Linked list node with start and end time.
    """

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, prev=None, next=None) -> None:
        if start_time >= end_time:
            assert False, f"Wrong interval boundaries - start time {start_time} is after or equal end time"\
                          f" {end_time}, prev={prev}, next={next}"
        self.start_time = start_time
        self.end_time = end_time
        self.set_prev(prev)
        self.set_next(next)
        # Note that events need to add in a custom way, they often inherits from base Interval.
        self.events: List[Event] = []
        self.name = None  # Depends from Interval purpose.

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o , Interval) \
                and self.start_time == __o.start_time \
                and self.end_time == __o.end_time \
                and self.name == __o.name

    def __repr__(self):
        return self.to_str(False)

    def set_prev(self, prev_interval: Interval):
        self.prev = prev_interval
        if prev_interval:
            prev_interval.next = self

    def set_next(self, next_interval: Interval):
        self.next = next_interval
        if next_interval:
            next_interval.prev = self

    def to_str(self, debug=False) -> str:
        """
        Makes string representation.
        :param debug: Flag to add all events information. If `False` then puts only the last one.
        :return: String representation of the interval.
        """
        description = self.name
        if not description and len(self.events) > 0:
            if debug:
                events_str = ";".join((event_data_to_str(x) for x in self.events))
                description = f"{len(self.events)} events={events_str}"
            else:
                description = f"{len(self.events)} events, last={event_data_to_str(self.events[-1])}"
        return f"{self.start_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
               f"..{self.end_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}: {description}"

    def get_duration(self) -> float:
        """
        :return: Duration of interval in seconds.
        """
        return (self.end_time - self.start_time).total_seconds()

    def iterate_next(self, checker: Callable[[Interval], bool] = None) -> Interval:
        """
        Iterates 'next' intervals with checking order of nodes.
        :param checker: Lambda to return True if provided `Interval` is searched one.
        :return: `Interval` where given checker responded with `True`, otherwise last `Interval`.
        """
        if checker and checker(self):
            return self
        interval = self
        while interval.next:
            tmp = interval.next
            # Consistency checks.
            if tmp.start_time <= interval.start_time or tmp.start_time < interval.end_time:
                raise ValueError(f"Wrong 'next' link in '{interval}'->'{tmp}': "
                                 f"{tmp.start_time} expected to be after (greater) {interval.end_time}.")
            # Finish check.
            if checker and checker(tmp):
                return tmp
            interval = tmp
        return interval

    def iterate_prev(self, checker: Callable[[Interval], bool] = None) -> Interval:
        """
        Iterates 'prev' intervals with checking order of nodes.
        :param checker: Lambda to return True if provided `Interval` is searched one.
        :return: `Interval` where given checker responded with `True`, otherwise last `Interval`.
        """
        if checker and checker(self):
            return self
        interval = self
        while interval.prev:
            tmp = interval.prev
            # Consistency checks.
            if tmp.start_time >= interval.start_time or tmp.end_time > interval.start_time:
                raise ValueError(f"Wrong 'prev' link in '{tmp}'<-'{interval}': "
                                 f"{tmp.end_time} expected to be before (lesser) {interval.start_time}.")
            # Finish check.
            if checker and checker(tmp):
                return tmp
            interval = tmp
        return interval

    def get_count(self):
        interval = self.iterate_prev()
        cnt = 1
        while interval := interval.next:
            cnt += 1
        return cnt

    def get_range(self, offset: int=0, num: int=1, from_earliest=False) -> List[Interval]:
        """
        Builds list of surrounding intervals.
        :param offset: Negative (earlier in time) or positive (later) integer to start get intervals from.
        :param num: Number of intervals to get. If negative or 0 then don't limit.
        :param from_earliest: Flag to start iterate from earliest interval.
        :return: List of intervals or empty list if offset on number of intervals is out of bounds.
        """
        interval = self
        if from_earliest:
            interval = self.iterate_prev()
        elif offset != 0:
            if offset > 0:  # Scroll forward.
                i = 1
                while interval.next:
                    interval = interval.next
                    if i >= offset:
                        break
                    i += 1
            else:
                i = -1
                while interval.prev:
                    interval = interval.prev
                    if i <= offset:
                        break
                    i -= 1
        result = []
        if interval:
            result.append(interval)
            i = 1
            while interval := interval.next:
                if i == num:  # If num < 1 then don't limit at all.
                    break
                result.append(interval)
                i += 1
        return result

    def intervals_to_string(self, offset: int=0, num: int=0, from_earliest=True, debug=False) -> str:
        """
        Builds string with description of surrounding intervals.
        :param offset: Negative (earlier in time) or positive (later) integer to start get intervals from.
        :param num: Number of intervals to get. If negative or 0 then don't limit.
        :param from_earliest: Flag to start iterate from earliest interval.
        :param debug: Flag to pass into `to_str()` method.
        :return: List of intervals.
        """
        intervals = self.get_range(offset, num, from_earliest)
        return str(len(intervals)) + " intervals:\n  " + "\n  ".join(x.to_str(debug) for x in intervals)

    def compare_with_time(self, time: datetime.datetime, tolerance: datetime.timedelta, is_start: bool) -> float:
        """
        Compares interval boundaries with specified time.
        :param time: Time to compare interval boundaries with.
        :param tolerance: Precision for "mathes" check.
        :param is_start: Flag to compare with interval start. If `False` then compares with interval end.
        :return: Negative value if time before (earlier) than interval edge, `0` if time matches edge with specified
        tolerance, positive value if time is after (later) than interval edge.
        """
        diff = time - (self.start_time if is_start else self.end_time)
        return 0 if abs(diff) <= tolerance else diff.total_seconds()

    def find_closest(self, date: datetime.datetime, tolerance: datetime.timedelta, by_end_time: bool) -> Interval:
        """
        Finds interval in the linked list with selectively `end_time` or `start_time` closer to the given time.
        First tries to find first interval on the specified time or right before. If there are no such intervals
        then returns closest interval after.
        :param date: Time to search interval closest to.
        :param by_end_time: Flag to search by interval `end_time`. Otherwise searches by `start_time`.
        :return: Interval closest before or on the specified time.
        """
        interval = self
        # First check if date is later than current interval (need search it later intervals).
        if (interval.end_time if by_end_time else interval.start_time) - tolerance <= date:
            return self.iterate_next(lambda x: (x.end_time if by_end_time else x.start_time) + tolerance >= date)
        else:  # Date is before current interval (need search in previous intervals).
            return self.iterate_prev(lambda x: (x.end_time if by_end_time else x.start_time) - tolerance <= date)

    def new_after(self, event: Event) -> Interval:
        """
        Creates new `Interval` from specified event and inserts it after current.
        Doesn't use event start time. Doesn't put 'next' for current event.
        :param event: Event which causes new interval creation.
        :return: Just created interval.
        """
        assert self.next is None, f"'{self}'.new_after is called while 'next' exists: f{self.next}."
        interval = Interval(self.end_time, event.timestamp + event.duration, self, None)
        interval.events.append(event)
        self.set_next(interval)
        return interval

    def separate_new_at_start(self, event: Event, tolerance: datetime.timedelta) -> Interval:
        """
        Separates current `Interval` to 2, with earliest part based on the given event.
        I.e. from [0<-self->3] makes [0<-new->1][1<-self->3]. Does nothing if resulting interval duration shorter than
        given tolerance. Doesn't upadate/set names for both intervals. Doesn't use event start time.
        :param event: `Event` to split current interval with.
        :return: Just created interval or current interval if no actions were performed.
        """
        event_end_time = event.timestamp + event.duration
        if abs(self.start_time - event_end_time) <= tolerance or abs(event_end_time - self.end_time) <= tolerance:
            return self
        interval = Interval(self.start_time, event_end_time, self.prev, self)
        interval.events.extend(self.events)
        interval.events.append(event)
        self.start_time = interval.end_time
        return interval

    def separate_new_at_end(self, event: Event, tolerance: datetime.timedelta) -> Interval:
        """
        Separates current `Interval` to 2, with latest part based on the given event.
        I.e. from [0<-self->3] makes [0<-self->2][2<-new->3]. Does nothing if resulting interval duration shorter than
        given tolerance. Doesn't upadate/set names for both intervals. Doesn't use event end time.
        :param event: `Event` to split current interval with.
        :return: Just created interval or current interval if no actions were performed.
        """
        if abs(self.start_time - event.timestamp) <= tolerance or abs(event.timestamp - self.end_time) <= tolerance:
            return self
        interval = Interval(event.timestamp, self.end_time, self, self.next)
        interval.events.extend(self.events)
        interval.events.append(event)
        self.end_time = interval.start_time
        return interval

    def separate_new_at_middle(self, event: Event, tolerance: datetime.timedelta) -> Interval:
        """
        Separates current `Interval` to 3, with only middle parh based on given event.
        I.e. from [0<-self->3] makes [0<-self->1][1<-new->2][2<-self->3].
        Does nothing if resulting interval duration shorter than given tolerance.
        Doesn't upadate/set names for all resulting intervals.
        :param event: `Event` to split current interval with.
        :return: Just created interval in the middle of initial interval or initial interval if no actions were
        performed.
        """
        # last = self.separate_new_at_end(event, tolerance)
        # return self.separate_new_at_end(event, tolerance)

        event_end_time = event.timestamp + event.duration
        # First separate last part i.e. [0<-self->2][2<-self->3]. Only if it makes sense.
        if abs(self.start_time - event_end_time) > tolerance and abs(event_end_time - self.end_time) > tolerance:
            last_interval = Interval(event_end_time, self.end_time, self, self.next)
            last_interval.events.extend(self.events)
            self.end_time = last_interval.start_time
        # Next separate current interval with new at the end.
        return self.separate_new_at_end(event, tolerance)

    def merge_with_next(self):
        self.end_time = self.next.end_time
        self.events.extend(self.next.events)
        self.set_next(self.next.next)

    def merge_all_adjacent_with_same_name(self):
        interval = self.iterate_prev()
        while interval.next:
            if interval.end_time == interval.next.start_time and interval.name == interval.next.name:
                interval.merge_with_next()
            interval = interval.next


@dataclasses.dataclass
class Rule:
    """
    Structure which represents how to compare related event with others.
    :param priority: Priority of the rule, greater value => more chance to be used among remained.
    Better to use unique priority values, otherwise conflicts will be solved inpredictably.
    Note that default priority for "afk" watcher is 500, for "watchdog" - 1000.
    :param subhandler: `RulesHandler` for the different key of event with its own set of rules.
    It allows to take into consideration several keys of the event.
    :param ignore: Flag to ignore this activity for reports. May be used for "not work" activities.
    """

    priority: int
    subhandler: RulesHandler = None
    ignore: bool = False
    # def __init__(self, priority: int = 1, subhandler: RulesHandler = None,
    #              ignore: bool = False) -> None:
    #     """
    #     Constuctor.
    #     :param priority: Priority of the rule, greater value = more chance to be used among remained.
    #     Better to use unique priority values, otherwise conflicts will be solved inpredictably.
    #     Note that default priority for "afk" watcher is 50, for "watchdog" - 100.
    #     :param subhandler: `RulesHandler` for the different key with its own set of rules to apply only for this rule.
    #     It allows to handle all keys of the event.
    #     :param ignore: Flag to ignore this activity for reports. May be used for "not work" activities.
    #     """
    #     self.priority = priority
    #     self.subhandler = subhandler
    #     self.ignore = ignore


class RulesHandler:

    # TODO add parameter for "to_str" keys set (for firefox 'title' and so on)
    def __init__(self, key: str, rules: Dict[str, Rule], to_str_keys: Optional[List[str]] = None,
                 tolerance: datetime.timedelta=TOLERANCE) -> None:
        """
        Default constructor.
        :param key: Key from ActivityWatch event to choose rules basing on.
        :param rules: Map of key regexp pattern to `Rule` object for handling it.
        :param to_str_keys: List of keys to make `to_str` implementation basing on. If not specified then base
        event key value is used.
        """
        self.key = key
        self.rules: Dict[re.Pattern, Rule] = dict((re.compile(k), v) for k, v in rules.items())
        self.to_str_keys = to_str_keys

    def __repr__(self) -> str:
        return f"RulesHandler(key={self.key}, rules_len={len(self.rules)})"

    def get_rule(self, event: Event) -> Optional[Tuple[Rule, List[str]]]:
        """
        Searches rule for specified event.
        :param event: Event to find rule for.
        :return: `Rule` handling specified event and ordered list of values which point to the rule.
        """
        value = event.data[self.key]
        values = [value]
        rule = next((rule for (regex, rule) in self.rules.items() if regex.match(value)), None)
        if rule and rule.subhandler:
            rule, subhandler_values = rule.subhandler.get_rule(event)
            values.extend(subhandler_values)
        return rule, values

    def get_event_name(self, event: Event) -> Any:
        return event.data[self.key]


def _span_event_down(event: Event, interval: Interval, tolerance: datetime.timedelta) -> Interval:
    """
    Recursively updates event down starting after specified interval.
    :param event: Event to apply.
    :param interval: Interval to start apply after.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :return: Last resulting interval.
    """
    # Recursive implementation makes stack overflow.
    while interval.next:
        interval = interval.next
        ends_diff = interval.compare_with_time(event.timestamp + event.duration, tolerance, False)
        if ends_diff == 0:
            interval.events.append(event)
            # TODO update name
            return interval
        elif ends_diff < 0:
            interval = interval.separate_new_at_start(event, tolerance)
            # TODO update name
            return interval
        else:
            interval.events.append(event)
            # TODO update name
    return interval


def apply_events(events: List[Event], interval: Interval, tolerance: datetime.timedelta) -> Dict[str, int]:
    """
    Applies events to the specified linked list of intervals.
    Expected that this list contains from "afk" and "watchdog" watchers, i.e. events out of bounds of existing
    intervals should be ignored.
    :param events: List of events to apply.
    :param interval: Node of linked list to apply events onto. Out parameter.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :return: Dictionary of metrics obtained on applying events.
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
                # TODO update name
                metrics['cnt_match_interval'] += 1
            elif ends_diff < 0:
                # 5) Event start matches interval and event ends inside it.
                interval = interval.separate_new_at_start(event, tolerance)
                # TODO update name
                metrics['cnt_split_one_interval'] += 1
            else:
                # 7) Event start matches interval and event ends after it.
                interval = _span_event_down(event, interval, tolerance)
                metrics['cnt_split_few_intervals'] += 1
        elif starts_diff > 0:
            if ends_diff == 0:
                # 9) Event starts inside interval and ends matches.
                interval = interval.separate_new_at_end(event, tolerance)
                # TODO update name
                metrics['cnt_split_one_interval'] += 1
            elif ends_diff < 0:
                # 8) Event starts and ends inside interval.
                interval = interval.separate_new_at_middle(event, tolerance)
                # TODO update names for all 3
                metrics['cnt_inside_interval'] += 1
            else:
                # 10) Event starts inside interval and ends after it.
                interval = interval.separate_new_at_end(event, tolerance)
                # TODO update name
                interval = _span_event_down(event, interval, tolerance)
                metrics['cnt_split_few_intervals'] += 1
        else:
            if ends_diff == 0:
                # 3) Event started before "AFK" and ends exactly on first interval end.
                interval.events.append(event)
                # TODO update name
                metrics['cnt_match_interval'] += 1
            elif ends_diff < 0:
                # 2) Event started before "AFK" and ends inside first interval.
                interval = interval.separate_new_at_start(event, tolerance)
                # TODO update name
                metrics['cnt_split_one_interval'] += 1
            else:
                # 4) Event started before "AFK" and ends after first interval.
                interval.events.append(event)
                # TODO update name
                interval = _span_event_down(event, interval, tolerance)
                metrics['cnt_split_few_intervals'] += 1
        metrics['cnt_handled'] += 1
    return metrics


def check_and_print_intervals(interval: Interval, max_events_per_interval: int, last_bucket_name: str,
                              level: logging._Level = logging.INFO) -> None:
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


def report_from_buckets(awc: aw_client.ActivityWatchClient, start_time: datetime.datetime, end_time: datetime.datetime,
                        buckets: List[str], tolerance: datetime.timedelta) -> Interval:
    """
    Gets events from specified buckets, prints report by them and returns linked list of `Interval`-s.
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

    # 1) Handle "aw-watcher-afk" bucket events first to build base intervals to determine activity by.
    # "aw-watcher-afk" - active watcher, it doesn't show "active" when user is not.
    # But is doesn't catch mic on meetings so may produce wrong AFK status.
    # Data contains the only 'status' key with "afk" and "not-afk" values. Make sense extract 100% activities in
    # periods from "not-afk" to "afk" and in remained intervals if other watchers shows "meeting".
    # data={status: [afk, not-afk]}
    bucket_id = next((x for x in bucket_ids_to_handle if x.startswith("aw-watcher-afk")), None)
    if bucket_id:
        events: List[object] = awc.get_events(bucket_id, start=start_time, end=end_time)
        events.sort(key=lambda e: e.timestamp)
        events = [Event(bucket_id, x.timestamp, x.duration, x.data) for x in events]
        for event in events:
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
            cur_interval.name = event.data['status']
        bucket_ids_to_handle.remove(bucket_id)
        buckets_cnt += 1
    else:
        LOG.info("No AFK bucket found. Stopping here - no more events expected.")
        return None
    if cur_interval:
        cur_interval.merge_all_adjacent_with_same_name()  # Remove duplicates.
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
    events: List[object] = awc.get_events(bucket_id, start=start_time, end=end_time)
    if events:
        events = [Event(bucket_id, x.id, x.timestamp, x.duration, x.data) for x in events]
        for event in events:  # TODO implement.
            if cur_interval:
                # Events are sorted here.
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
        raw_events = awc.get_events(bucket_id, start=start_time, end=end_time)
        if raw_events:
            LOG.info(f"Applying '{bucket_id}' {len(raw_events)} events with.")
            # Note that some watchers (like IDEA watcher from few windows) makes events covering each other,
            # i.e. not adjusted. But on each focus change it do generates new event.
            # So first sort all events and cut to make adjusted.
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


@dataclasses.dataclass
class IntervalHandler:
    """
    Structure to represent analyzed interval with attributes required to TODO 
    """

    interval: Interval
    rule: Rule
    event: Event
    values: List[str]
    value: str


def analyze_intervals(interval: Interval, round_to: float, custom_rules: Dict[str, RulesHandler]) -> Dict[str, float]:
    """
    :param interval: Some interval from the full linked list of intervals to analyze.
    :param round_to: Both minimal summary interval length to show and step to align reporting intervals to.
    :param rules: User-specific map of event bucket name prefix to `RulesHandler` to handle data intervals.
    :return Dictionary of report interval description to effort. # TODO return more complicated object.
    """

    # Expected output:
    # A) Worked at intervals: A, B, C
    # B) TOTAL worked
    # C) Unknown = ZZZ
    # D) 2h = window "xxx"
    #   1h = browser "XXX" site
    #   1.75h = IDEA on projectPath
    #   0.25 = meeting in Zoom from XXX to YYY
    #   0.25 = meeting in Slack with ZZZ from XXX to YYY
    #   0.25 = Skype conversations in durations A, B, C
    # E) Gaps between events - to measure activities with turned off laptop (offline meetings, lunches, etc.) 

    # B: It is "aw-watcher-afk" data with not_afk plus "aw-watchdog" events.
    total_not_afk: float = 0.0
    # A: It are intervals where not_afk or '"not_afk": True' or "aw-stopwatch" data.
    aw_active_intervals: List[Tuple[datetime.datetime, datetime.datetime]] = []
    # D: It is A with more specific activity aggregated by this activity.
    tasks: Dict[str, float] = {}
    # Assemble full set of RulesHandler-s from predifined ones and custom.
    bucket_prefix_to_ruleshandler: Dict[str, RulesHandler] = dict(custom_rules)
    if "aw-watcher-afk" not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler["aw-watcher-afk"] = RulesHandler('status', {
            "afk": Rule(500, ignore=True),
            "not-afk": Rule(0)
        })
    if "aw-stopwatch" not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler["aw-stopwatch"] = RulesHandler('label', {
            ".*": Rule(0, subhandler=RulesHandler('running', {
                "true": Rule(1000),
                "false": Rule(0, ignore=True)  # Even if it is the only event in interval it carries no activity.
            }))
        })
    # Iterate all intervals, find out main event and associated rule for each, collect map of "activity" per duration.
    cur_interval: Interval = interval.iterate_prev()  # Go to the first interval.
    activity_counter = collections.Counter()
    # Prepare dummy to use first interval on the very first iteration below.
    cur_interval = Interval(cur_interval.start_time, cur_interval.end_time, None, cur_interval)
    while cur_interval.next:
        cur_interval = cur_interval.next
        # Find out all rules applicable to the interval.
        interval_handler: IntervalHandler = None
        for event in cur_interval.events:
            handler: RulesHandler = next((handler for bucket_prefix, handler in bucket_prefix_to_ruleshandler.items()
                                         if event.bucket_id.startswith(bucket_prefix)), None)
            if not handler:
                LOG.info(f"Can't find handler for {event}")
                continue
            rule, values = handler.get_rule(event)
            if rule:
                # Keep only rule with the highest priority.
                if interval_handler is None or rule.priority > interval_handler.rule.priority:
                    interval_handler = IntervalHandler(cur_interval, rule, event, values, "->".join(values))
        # Decide whether to count this interval or not.
        if interval_handler is None:
            LOG.info(f"Skipping {cur_interval} because it doesn't contain events matching any rule."
                     + " Events:\n  " + "\n  ".join(str(x) for x in cur_interval.events))
            continue
        if interval_handler.rule.ignore:
            LOG.debug(f"Skipping {interval_handler.interval} because of {interval_handler.rule} priority is highest"
                      f" for {interval_handler.event}.")
            continue
        # is_fresh = cur_interval.prev is None or cur_interval.prev.end_time != cur_interval.start_time
        # TODO collect rule+event into buckets.
        activity_counter[interval_handler.value] += interval_handler.interval.get_duration()
    # TODO Apply `round_to` with distributing small buckets into wider ones.
    return activity_counter


# Min duration one activity events sum to show.
MIN_DURATION_SEC = 15 * 60  # 0.25 hours
# Keys matches bucket names start. If there will be few buckets with ID starting from key then all will be handled.
# If few keys match the same bucket then only first RulesHandler will be applied to the bucket events.
RULES = {
    # Passive watcher, always provides value, even if user AFK. But "change value" event 100% shows activity.
    # data={app: str, title: str}.
    "aw-watcher-window": RulesHandler("app", {
        "zoom": Rule(90),
        "Slack": Rule(89, subhandler=RulesHandler("title", {
            # BTW it is not the only case.
            ".*screen share": Rule(89),
        })),
        # Skype doesn't provide info that it is a meeting.
        "Skype": Rule(88),
        "unknown": Rule(0),  # Means that window manager was unable to gather data (for Discord, )
        "flameshot": Rule(5),  # Screenshot tool.
        "jetbrains-idea": Rule(40),  # IDE. TODO ~2 seconds after "afk" watcher events -> are not counted!
        "Double Commander": Rule(35),  # File manager.
        "smplayer": Rule(36),  # Video player.
        "FeatherPad": Rule(37),  # Text editor.
    }),
    # Passive watcher, always provides value, even if user AFK.
    # But "change value" event most probably shows activity (excluding web pages which change title periodically).
    # data={url: str, title: str, audible: bool, incognito: bool, tabCount: int}.
    "aw-watcher-web": RulesHandler("url", {
        "https://vimbox.skyeng.ru/.*": Rule(87, ignore=True),
        "https://gitlab.akvelon.net:9443/.*": Rule(41),
        "https://akvelon.atlassian.net/wiki/.*": Rule(42),
        "https://gitlab.intapp.com/.*": Rule(43),
        "https://wiki.intapp.com/wiki/.*": Rule(44),
    }),
    # Passive watcher, always provides value, even if user AFK. But "change value" event shows activity/focus on.
    # data={file: str, projectPath: str, language: str, editor: const, editorVersion: const, eventType: const}
    "aw-watcher-idea": RulesHandler("file", {}),  # TODO build some rules for IDEA
}


def main():

    # TODO ask date
    # daystart = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
    # dayend = daystart + datetime.timedelta(days=1)
    # TODO support logging levels.

    client = aw_client.ActivityWatchClient("activity_merger")
    buckets = client.get_buckets()
    LOG.info(f"Buckets: {buckets.keys()}")
    interval = report_from_buckets(client, datetime.datetime(2022, 2, 11), datetime.datetime(2022, 2, 12), buckets, TOLERANCE)
    report = analyze_intervals(interval, MIN_DURATION_SEC, RULES)
    LOG.info(report)


def setup_logging():
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.DEBUG, "DEBU")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)-4s: %(message)s',
        datefmt="%H:%M:%S"
    )
    return logging.getLogger()


if __name__ == '__main__':
    LOG = setup_logging()
    main()
