#!/usr/bin/env python3
import datetime
import logging
import aw_client
import collections
import dataclasses
from typing import Any, List, Dict, Optional, Tuple, Callable
import re

REMOVE


# Tolerance to use when comparing events. Events shorter than this value are ignored.
# If duration between start and end of different events if equal or less then they are treated adjacent.
EVENTS_COMPARE_TOLERANCE_TIMEDELTA = datetime.timedelta(0, 1, 0)  # 1 sec
# Default priority of "afk" event. All events with equal or higher priority are treated as "independent"
# and may form separate activities.
AFK_RULE_PRIORITY = 500
# Default priority of "watchdog" watcher, aka maximum priority.
WATCHDOG_RULE_PRIORITY = 1000
# Which activity treat as too long for additional logging.
TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS = datetime.timedelta(2, 0, 0).seconds  # 2 hours
# Timezone to show dates.
CURRENT_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo  # Use system timezone.
# Common logger.
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


def from_start_to_end_to_str(obj) -> str:
    return f"{obj.start_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{obj.end_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"


def seconds_to_int_timedelta(seconds: float) -> str:
    return datetime.timedelta(seconds=int(seconds))


class Interval:
    """
    Interval in time when at least one full `Event` happened. Used to make time-ordered linked list.
    :param start_time: Interval start time.
    :param end_time: Interval end time.
    :param events: List of `Event`-s happened in this interval.
    """

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, prev=None, next=None) -> None:
        """
        Default constructor.
        :param start_time: Interval start time.
        :param end_time: Interval end time.
        :param prev: Previous interval in linked list of time-ordered intervals.
        :param next: Next interval in linked list of time-ordered intervals.
        """
        if start_time >= end_time:
            assert False, f"Wrong interval boundaries - start time {start_time} is after or equal end time"\
                          f" {end_time}, prev={prev}, next={next}"
        self.start_time = start_time
        self.end_time = end_time
        self.set_prev(prev)
        self.set_next(next)
        # Note that events need to add in a custom way, they often inherits from base Interval.
        self.events: List[Event] = []

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o , Interval) \
                and self.start_time == __o.start_time \
                and self.end_time == __o.end_time \

    def __repr__(self):
        return self.to_str(False)

    def set_prev(self, prev_interval: 'Interval') -> None:
        self.prev = prev_interval
        if prev_interval:
            prev_interval.next = self

    def set_next(self, next_interval: 'Interval') -> None:
        self.next = next_interval
        if next_interval:
            next_interval.prev = self

    def to_str(self, debug=False, only_time=False) -> str:
        """
        Makes string representation.
        :param debug: Flag to add all events information. If `False` then puts only the last one.
        :return: String representation of the interval.
        """
        result = from_start_to_end_to_str(self)
        if only_time:
            return f"{result} ({seconds_to_int_timedelta(self.get_duration())}):"
        if len(self.events) > 0:
            if debug:
                events_str = ";".join((event_data_to_str(x) for x in self.events))
                return f"{result}: {len(self.events)} events={events_str}"
            else:
                return f"{result}: {len(self.events)} events, last={event_data_to_str(self.events[-1])}"

    def get_duration(self) -> float:
        """
        :return: Duration of interval in seconds.
        """
        return (self.end_time - self.start_time).total_seconds()

    def iterate_next(self, checker: Callable[['Interval'], bool] = None) -> 'Interval':
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

    def iterate_prev(self, checker: Callable[['Interval'], bool] = None) -> 'Interval':
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

    def get_range(self, offset: int=0, num: int=1, from_earliest=False) -> List['Interval']:
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

    def find_closest(self, date: datetime.datetime, tolerance: datetime.timedelta, by_end_time: bool) -> 'Interval':
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

    def new_after(self, event: Event) -> 'Interval':
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

    def separate_new_at_start(self, event: Event, tolerance: datetime.timedelta) -> 'Interval':
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

    def separate_new_at_end(self, event: Event, tolerance: datetime.timedelta) -> 'Interval':
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

    def separate_new_at_middle(self, event: Event, tolerance: datetime.timedelta) -> 'Interval':
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


@dataclasses.dataclass
class Rule:
    """
    Structure which represents how to compare related event with others.
    :param key_pattern: Regexp pattern to handle intervals of. Make sense only in scope of specific `EventKeyHandler`.
    :param priority: Priority of the rule, greater value => more chance to be used among other rules on interval.
    Better to use unique priority values, otherwise conflicts will be solved unpredictably.
    Note that default priority for "afk" watcher 'afk' state is `AFK_RULE_PRIORITY` (500?), 'not-afk' rule - 1,
    for "watchdog" "enabled" state - `WATCHDOG_RULE_PRIORITY` (1000?). These constants are also used to compound
    intervals into "activities".
    :param subhandler: `EventKeyHandler` for the different key of event with its own set of rules.
    It allows to take into consideration several keys of the event.
    :param to_string: Lambda which produces rule description from "key pattern" regexp `re.Match`. If `None` then uses
    the full match (0 group), if returns `None` then doesn't count.
    :param skip: Flag to skip this activity from reports. May be used for "non working" activities.
    :param is_placeholder: Flag for "service" rules. They are useless for report and usually 0 priority.
    Will cause some hints about necessity of new rule to cover related acitivity.
    :param merge_next: Flag to merge current interval with the next interval rule. Useful for 'new browser tab' like
    activities (i.e. it is impossible to reveal activity from related event but it is part of next event activity).
    """

    key_pattern: str
    priority: int
    subhandler: 'EventKeyHandler' = None
    to_string: Callable[[str], str] = None
    skip: bool = False
    is_placeholder: bool = False
    merge_next: bool = False

    def get_description(self, match: re.Match) -> Optional[str]:
        """
        :return: Description of activity represented by rule or `None` if description should be set by subhandlers.
        """
        return match.group(0) if self.to_string is None else self.to_string(match)

class EventKeyHandler:
    """
    Structure which binds multiple `Rule`-s to one `Event`'s specific key values (by regexp).
    In spite of it supports case with absent key in event data it is better to separate `EventKeyHandler`-s
    per bucket because set of keys differs between differnt bucket events.
    """

    def __init__(self, key: str, rules: List[Rule], to_str_keys: Optional[List[str]] = None) -> None:
        """
        Default constructor.
        :param key: Key from ActivityWatch event to choose rules basing on.
        :param rules: List of `Rule` objects to handle differrent "key" values.
        :param to_str_keys: List of keys to make `to_str` implementation basing on. If not specified then base
        event key value is used.
        """
        self.key = key
        self.rules: Dict[re.Pattern, Rule] = dict((re.compile(rule.key_pattern), rule) for rule in rules)
        self.to_str_keys = to_str_keys

    def __repr__(self) -> str:
        return f"EventKeyHandler(key={self.key}, rules_len={len(self.rules)})"

    def get_rule(self, event: Event) -> Optional[Tuple[Rule, List[str]]]:
        """
        Searches rule for specified event.
        :param event: Event to find rule for.
        :return: `Rule` handling specified event and ordered list of matching key value description-s in order of rule
        handlers which point to the rule.
        """
        value = event.data[self.key]
        descriptions = []
        matched_rule = None
        for (regex, rule) in self.rules.items():
            match = regex.match(value)
            if match:
                description = rule.get_description(match)
                if description:
                    descriptions.append(description)
                matched_rule = rule
                break
        if matched_rule and matched_rule.subhandler:
            matched_rule, subhandler_descriptions = matched_rule.subhandler.get_rule(event)
            descriptions.extend(subhandler_descriptions)
        return matched_rule, descriptions


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


def report_from_buckets(awc: aw_client.ActivityWatchClient, start_time: datetime.datetime, end_time: datetime.datetime,
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
        bucket_ids_to_handle.remove(bucket_id)
        buckets_cnt += 1
    else:
        LOG.info("No AFK bucket found. Stopping here - no more events expected.")
        return None
    if cur_interval:
        # cur_interval.merge_all_adjacent_with_same_data()  # Remove duplicates.
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
        for event in events:  # TODO implement and put before AFK events handling.
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
            # i.e. not adjacent. But on each focus change it do generates new event.
            # So first sort all events and cut to make adjacent.
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
class RuleResult:
    """
    Structure to represent rule output like covered intervals, source event and information why all of them were
    chosen to be connected under one `Rule`.
    :param rule: `Rule` which produced this instance.
    :param event: `Event` choosen by `Rule`.
    :param description: Description of underlying interval.
    :param intervals: List of `Interval`-s covering by this rule.
    :param values: List of `Event` data pieces which pointed on this rule.
    """

    rule: Rule
    event: Event
    description: str
    intervals: List[Interval]
    values: List[str]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rule={self.rule}, description={self.description})"


@dataclasses.dataclass
class Activity:
    """
    One or few `RuleResult`-s separated as independent activity.
    """

    start_time: datetime.datetime
    end_time: datetime.datetime
    rule_results: List[RuleResult]
    description: str
    # Note that 'end_time minus start_time' doesn't work due to possible gaps between intervals.
    duration: float

    def __repr__(self) -> str:
        return f"{seconds_to_int_timedelta((self.duration))} "\
               f"({from_start_to_end_to_str(self)}) {self.description}"


def _increment_metric(metrics: Dict[str, Tuple[int, float]], metric_name: str, interval: Interval):
    metric = metrics.get(metric_name, (0, 0))
    metrics[metric_name] = (metric[0] + 1, metric[1] + interval.get_duration())


def _intervals_duration(intervals: List[Interval]):
    # Note that 'last end minus first start' doesn't work due to possible gaps between intervals.
    return sum(x.get_duration() for x in intervals)


def _find_out_rule_for_interval(
        interval: Interval,
        metrics: Dict[str, Tuple[int, float]],
        bucket_prefix_to_ruleshandler: Dict[str, List[EventKeyHandler]]
    ) -> RuleResult:
    rule_result: RuleResult = None
    # Iterate all events to 
    for event in interval.events:
        # Search `EventKeyHandler` by 2 criteria:
        # 1) event bucket ID starts with handler 'bucket_id',
        # 2) handler key exists in event data.
        handler: EventKeyHandler = None
        for bucket_prefix, bucket_handlers in bucket_prefix_to_ruleshandler.items():
            if event.bucket_id.startswith(bucket_prefix):
                for bucket_handler in bucket_handlers:
                    if bucket_handler.key in event.data:
                        handler = bucket_handler
                        break
        if not handler:
            LOG.info(f"Can't find handler for {event}")
            _increment_metric(metrics, 'events without handlers', interval)
            continue  # It is OK, some events (inside interval) may don't have handlers intentionally.
        # Find rule by event data. Note that it may be sorted out by priority afterwards.
        rule, descriptions = handler.get_rule(event)
        if rule:
            # Keep only rule with the highest priority.
            if rule_result is None or rule.priority > rule_result.rule.priority:
                rule_result = RuleResult(
                    rule, event, "->".join(descriptions), [interval], []
                )
    return rule_result


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
        repr = f"{seconds_to_int_timedelta(self.duration)} with priority={self.priority}"
        if debug:
            events = dict((x.event.timestamp, x.event) for x in self.rule_results)
            events_str = "\n  ".join(
                f"{seconds_to_int_timedelta(x.duration.total_seconds())} {event_data_to_str(x)}"
                for x in events.values()
            )
            return f"{repr}, description='{self.description}' and {len(events)} events:" + "\n  " + events_str
        else:
            return f"{repr} and {len(self.rule_results)} rule results"

    def append(self, rule_result: RuleResult):
        self.rule_results.append(rule_result)
        self.duration += _intervals_duration(rule_result.intervals)
        # If new rule has more priority than all existing in window then update windows priority and description.
        if rule_result.rule.priority >= self.priority:
            self.priority = rule_result.rule.priority
            self.description = rule_result.description

    def to_activity(self) -> Activity:
        start_time = self.rule_results[0].intervals[0].start_time
        end_time = self.rule_results[-1].intervals[-1].end_time
        tmp = set(x.description for x in self.rule_results)
        description = ", ".join(sorted(tmp))
        return Activity(start_time, end_time, list(self.rule_results), description, self.duration)


def _is_new_activity(window: RuleResultsWindow, rule_result: RuleResult, round_to: float) -> bool:
    current_rule = rule_result.rule
    # Don't separate activities for rules with the same priority or same description.
    if current_rule.priority == window.priority and rule_result.description != window.description:
        return False
    # Separate all "indipendent" activities as soon as they appear.
    if current_rule.priority >= AFK_RULE_PRIORITY:
        return rule_result.description != window.description
    # TODO Apply `round_to` with distributing small buckets into wider ones.
    return False


def analyze_intervals(interval: Interval, round_to: float, custom_rules: Dict[str, List[EventKeyHandler]]
        ) -> Tuple[List[Activity], collections.Counter, Dict[str, Tuple[int, float]]]:
    """
    :param interval: Linked list of intervals to analyze.
    :param round_to: Both minimal summary interval length to show and step to align reporting intervals to.
    :param rules: User-specific map of event bucket name prefix to list of `EventKeyHandler`-s to handle data intervals.
    :return: Tuple of:
    1 - List of assembled `Activity`-es.
    2 - `Counter` of interval description-s to theirs durations. Aka `Activity`-es built by naive "equal" strategy.
    3 - Map of metrics to estimate report quality/coverage.
    """
    # Assemble full set of EventKeyHandler-s from predifined ones and custom.
    bucket_prefix_to_ruleshandler: Dict[str, List[EventKeyHandler]] = dict(custom_rules)
    if "aw-watcher-afk" not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler["aw-watcher-afk"] = [EventKeyHandler('status', [
            Rule("afk", AFK_RULE_PRIORITY, skip=True),
            Rule("not-afk", 1, is_placeholder=True)
        ])]
    if "aw-stopwatch" not in bucket_prefix_to_ruleshandler:
        bucket_prefix_to_ruleshandler["aw-stopwatch"] = [EventKeyHandler('label', [
            Rule(".*", 0, subhandler=EventKeyHandler('running', [
                Rule("true", WATCHDOG_RULE_PRIORITY),
                Rule("false", 0, skip=True)  # Even if it is the only event in interval it carries no activity.
            ]))
        ])]
    # Prepare to loop through intervals with searching rules, building report and metrics.
    # Go to the first interval.
    cur_interval: Interval = interval.iterate_prev()
    # Make containers for outputs.
    rules_counter = collections.Counter()
    activities: List[Activity] = []
    metrics = {  # Put default metrics. Next it will be appended by per-rule metrics.
        'total intervals': (0, 0),
        'events without handlers': (0, 0),
        'intervals without rules': (0, 0),
        'intervals merged to next rule': (0, 0),
        'intervals with rule to skip': (0, 0),
        'intervals need to reveal rule for': (0, 0),
    }
    # Prepare dummy `Interval`` to use first interval on the very first iteration below.
    cur_interval = Interval(cur_interval.start_time, cur_interval.end_time, None, cur_interval)
    # Iterate all intervals, find rules for all events in it and choose highest by prioirty to handle interval to
    # build slices, activity_counter, metrics.
    deferred_intervals: List[Interval] = []  # `Interval`-s deffered as "append to next independent rule".
    window: RuleResultsWindow = None
    while cur_interval.next:
        cur_interval = cur_interval.next
        rule_result = _find_out_rule_for_interval(cur_interval, metrics, bucket_prefix_to_ruleshandler)
        # Decide whether to count this interval or not and update metrics.
        if rule_result is None:
            LOG.info(f"Skipping {cur_interval} because it doesn't contain events matching any rule."
                     + " Events:\n  " + "\n  ".join(str(x) for x in cur_interval.events))
            _increment_metric(metrics, 'intervals without rules', cur_interval)
            continue
        # Update per-rule-name metric. It should include all rules (i.e. "skip", "placeholder", etc.).
        _increment_metric(metrics, str(rule_result.rule), cur_interval)
        duration = _intervals_duration(rule_result.intervals)
        # Append deferred intervals if there are such.
        if deferred_intervals is not None:
            rule_result.intervals += deferred_intervals
            _increment_metric(metrics, 'intervals merged to next rule', cur_interval)
            deferred_intervals = []
        # Check if rule says skip interval from the report.
        if rule_result.rule.skip:
            LOG.debug(f"Skipping {duration:.1f} sec {len(rule_result.intervals)}"
                      f" intervals(s) because of {rule_result.rule} priority is highest"
                      f" for {rule_result.event}.")
            _increment_metric(metrics, 'intervals with rule to skip', cur_interval)
            continue
        # Check if rule is a placeholder and provide all information about interval to write appropriate rule for it. 
        if rule_result.rule.is_placeholder:
            LOG.info("Need to reveal rule for interval %s with %d events:\n  %s"
                     % (cur_interval.to_str(only_time=True), len(cur_interval.events),
                        "\n  ".join(str(x) for x in cur_interval.events))
                    )
            _increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
        # is_fresh = cur_interval.prev is None or cur_interval.prev.end_time != cur_interval.start_time
        # Update 'total_intervals' metric.
        _increment_metric(metrics, 'total intervals', cur_interval)
        # Defer interval if need. Note that `activity_counter` shouldn't be touched by this rule.
        if rule_result.rule.merge_next:
            deferred_intervals.append(cur_interval)
            continue
        # Update 'rules counter'.
        rules_counter[rule_result.description] += duration
        # -----------
        is_start_new_window = True  # By default start new window.
        if window is not None:
            # Decide if current `RuleResult` is separate activity from previous ones and need to create `Activity` from
            # items accumulated in `activity_window` so far.
            if _is_new_activity(window, rule_result, round_to):
                if window.duration < round_to:
                    LOG.info(f"On handling {rule_result} separated too small window {window.to_str(True)}")
                    # TODO _increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
                activities.append(window.to_activity())
            else:
                window.append(rule_result)
                # if window.duration > TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS:
                #     LOG.info(f"Too long window {window.to_str(False)} after checking {rule_result}.")
                #     # TODO _increment_metric(metrics, 'intervals need to reveal rule for', cur_interval)
                is_start_new_window = False
        if is_start_new_window:
            window = RuleResultsWindow([rule_result], rule_result.rule.priority, rule_result.description, duration)
    return activities, rules_counter, metrics


# Min duration one activity events sum to show.
MIN_DURATION_SEC = 15 * 60  # 0.25 hours
# Keys matches bucket names start. If there will be few buckets with ID starting from key then all will be handled.
# If few keys match the same bucket then only first EventKeyHandler will be applied to the bucket events.
RULES = {
    # Passive watcher, always provides value, even if user AFK. But "change value" event 100% shows activity.
    # data={app: str, title: str}.
    "aw-watcher-window": [
        EventKeyHandler("app", [
            Rule("zoom", 900, to_string=lambda _: None, subhandler=EventKeyHandler("title", [
                Rule("Zoom Meeting", 900, to_string=lambda _: "Zoom Meeting"),
                Rule(".*", 200, merge_next=True)
            ])),
            Rule("Slack", 890, to_string=lambda _: None, subhandler=EventKeyHandler("title", [
                Rule("Slack \| (.*?) \|.*", 889, to_string=lambda x: f"Slack {x.group(1)}"),
                Rule("(.*) screen share", 890, to_string=lambda x: f"Slack {x.group(1)}"),
            ])),
            # Skype doesn't provide info that it is a meeting.
            Rule("Skype", 880),
            Rule("unknown", 2, merge_next=True),  # Means that OS windows manager was unable to gather data.
            Rule("flameshot", 520),  # Screenshot tool.
            Rule("jetbrains-idea", 40),  # IDE. TODO ~2 seconds after "afk" watcher events -> are not counted!
            Rule("Double Commander", 35),  # File manager.
            Rule("smplayer", 36),  # Video player.
            Rule("FeatherPad", 37),  # Text editor.
            Rule("discord", 38),
        ]),
    ],
    # Passive watcher, always provides value, even if user AFK.
    # But "change value" event most probably shows activity (excluding web pages which change title periodically).
    # data={url: str, title: str, audible: bool, incognito: bool, tabCount: int}.
    "aw-watcher-web": [
        EventKeyHandler("url", [
            Rule("https://(vimbox|student)\.skyeng\.ru/.*", 501, skip=True), # English lesson, may look like AFK.
            Rule("https://gitlab\.akvelon\.net:9443/.*", 41, to_string=lambda _: "Akvelon GitLab"),
            Rule("https://akvelon\.atlassian\.net/wiki/.*", 42, to_string=lambda _: "Akvelon Wiki"),
            Rule("https://gitlab\.intapp\.com/.*", 43, to_string=lambda _: "Intapp GitLab"),
            Rule("https://wiki\.intapp\.com/wiki/.*", 44, to_string=lambda _: "Intapp Wiki"),
            Rule("https://mail\.akvelon\.com/.*", 45, to_string=lambda _: "Akvelon Mail"),
            Rule("https://intapp\.atlassian\.net/browse/.*", 46, to_string=lambda _: "Intapp Jira"),
            Rule("https://intapp\.zoom\.us/.*", 47, to_string=lambda _: "zoom"),
            Rule("https://www\.google\.com/.*", 48, to_string=lambda _: "www.google.com"),
            Rule("about:blank", 520, to_string=lambda _: None, subhandler=EventKeyHandler("title", [
                Rule("intapp\.atlassian\.net/browse/.*", 521, to_string=lambda _: "Intapp Jira"),
                Rule("zoom\.us/j/.*", 521, to_string=lambda _: "zoom"),
                Rule("logs\.(devops|us\.dev\.kube)\.intapp\.com.*", 401, to_string=lambda _: "Intapp Logs"),
                Rule("metrics\.(devops|us\.dev\.kube)\.intapp\.com.*", 400, to_string=lambda _: "Intapp Metrics"),
                Rule("gitlab\.akvelon\.net:9443/.*", 41, to_string=lambda _: "Akvelon GitLab"),
                Rule("New Tab", 42, merge_next=True),
                Rule("wiki\.intapp\.com/wiki/.*", 44, to_string=lambda _: "Intapp Wiki"),
                Rule("intapp\.zoom\.us/.*", 47, to_string=lambda _: "zoom"),
                # Last item as an "uncategorized site".
                Rule("(.+?)/.*", 3, to_string=lambda x: f"Firefox '{x.group(1)}'")
            ])),
            Rule("https://signin\.intapp\.com/", 49, to_string=lambda _: "Intapp SignIn"),
            Rule("https://intapp\.zendesk\.com/.*", 100, to_string=lambda _: "Intapp Zendesk"),
            Rule("https://docs\.google\.com/spreadsheets/.*", 101, to_string=lambda _: "Google Spreadsheets"),
            Rule("https://translate\.google.*", 102, to_string=lambda _: "Google Translate"),
            Rule("https://logs\.(devops|us\.dev\.kube)\.intapp\.com.*", 530, to_string=lambda _: "Intapp Logs"),
            Rule("https://metrics\.(devops|us\.dev\.kube)\.intapp\.com.*", 531, to_string=lambda _: "Intapp Metrics"),
            Rule("file:///.*", 532, to_string=lambda _: "Local file in browser"),
            # Last item as an "uncategorized site".
            Rule("https?://(.+?)/.*", 3, to_string=lambda x: f"Firefox '{x.group(1)}'")
        ]),
    ],
    # Passive watcher, always provides value, even if user AFK. But "change value" event shows activity/focus on.
    # data={file: str, projectPath: str, language: str, editor: const, editorVersion: const, eventType: const}
    # Need to handle only "switch to" intervals because watcher is strange.
    # Also keys are not stable in it, for example 'project' may be absent.
    "aw-watcher-idea": [
        EventKeyHandler("project", [
            Rule(".*", 100, to_string=lambda x: f"IDEA project '{x.group(0)}'")
        ]),
        EventKeyHandler("file", [
            Rule(".*", 100, to_string=lambda x: f"IDEA file '{x.group(0)}'")
        ])
    ],
}


def main():

    # TODO ask date, by default today
    # daystart = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
    # dayend = daystart + datetime.timedelta(days=1)
    # TODO need separate configuration file.
    # TODO need interactive way to merge activities
    # TODO need mixing of Jira/Outlook/watchdog events.

    client = aw_client.ActivityWatchClient("activity_merger.py")
    buckets = client.get_buckets()
    LOG.info(f"Buckets: {buckets.keys()}")
    # Build time-ordered linked list of intervals by provided events.
    interval = report_from_buckets(
        client,
        datetime.datetime(2022, 2, 11),
        datetime.datetime(2022, 2, 12),
        buckets,
        EVENTS_COMPARE_TOLERANCE_TIMEDELTA
    )
    # Convert (assemble) intervals list into activities.
    activities, activity_counter, metrics = analyze_intervals(interval, MIN_DURATION_SEC, RULES)
    # Print metrics as is.
    LOG.info(f"Metrics from intervals analysis ({len(metrics)}):" + "\n  "
             + "\n  ".join(f"{v[0]:4} on {datetime.timedelta(seconds=v[1])} - {k}" for k, v in metrics.items()))
    # Pring only "less than MIN_DURATION_SEC" from "dumb" activities.
    dumb_activities = [
        f"{seconds_to_int_timedelta(v)} {k}" for k, v in activity_counter.most_common() if v >= MIN_DURATION_SEC
    ]
    LOG.info("There were %d equal activities. Longer than %d are:\n  %s"
             % (len(activity_counter), MIN_DURATION_SEC, "\n  ".join(dumb_activities)))
    # Print all activities as is.
    LOG.info("Assembled %d activities:\n  %s" % (len(activities), "\n  ".join(str(x) for x in activities)))


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
