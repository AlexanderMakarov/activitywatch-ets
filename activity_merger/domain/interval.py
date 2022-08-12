import datetime
from typing import List, Callable
from .input_entities import Event
from ..helpers.helpers import *


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
