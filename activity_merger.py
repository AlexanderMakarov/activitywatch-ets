#!/usr/bin/env python3
from __future__ import annotations  # https://stackoverflow.com/a/33533514/1535127
import datetime
import logging
from difflib import restore
import aw_client
import socket
from typing import Any, List, Dict, Tuple, Callable
from aw_transform.filter_period_intersect import _intersecting_eventpairs
from aw_core.models import Event
import re

from numpy import number, void
from tomlkit import integer


# Which tolerance to use when comparing events. Mostly from different watchers.
EVENTS_COMPARE_TOLERANCE_SEC = 1
TOLERANCE = datetime.timedelta(0, 1, 0)  # 1 sec
# Timezone to show dates.
CURRENT_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
LOG: logging.Logger = logging.getLogger(__name__)


def event_data_to_str(event):
    if not event:
        return 'null'
    return str(event.data)


def event_to_str(event):
    return f"{event.timestamp.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{(event.timestamp + event.duration).astimezone(CURRENT_TIMEZONE):%H:%M:%S}("\
           f"{event_data_to_str(event)})"


class Interval:
    """
    Linked list node with start and end time.
    """

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, prev=None,
                 next=None) -> None:
        if start_time > end_time:
            assert False, f"Wrong interval boundaries - start time {start_time} is after end time {end_time},"\
                          f" prev={prev}, next={next}"
        self.start_time = start_time
        self.end_time = end_time
        self.prev = prev
        self.next = next
        # Note that events need to add in a custom way, they often inherits from base Interval.
        self.events = []  # TODO need to flag when event appear first time.
        self.name = None  # Depends from Interval purpose.

    def __repr__(self):
        return self.to_str(False)

    def to_str(self, debug=False):
        description = self.name
        if not description and len(self.events) > 0:
            if debug:
                events_str = ";".join((event_data_to_str(x) for x in self.events))
                description = f"{len(self.events)} events={events_str}"
            else:
                description = f"{len(self.events)} events, last={event_data_to_str(self.events[-1])}"
        return f"{self.start_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
               f"..{self.end_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}: {description}"

    def iterate_next(self, checker: Callable[[Interval], bool] = None) -> Interval:
        """
        Iterates 'next' intervals with protection from wrong order of nodes.
        :param checker: Lambda to retrurn True if required state is reached.
        :return: `Interval` where loop finished by given checker or last interval.
        """
        interval = self
        while interval.next:
            tmp = interval.next
            if tmp.start_time <= interval.start_time:
                raise ValueError(f"Wrong interval.next found: f{tmp} expected to be after current f{interval}.")
            if checker(tmp):
                return tmp
            interval = tmp
        return interval

    def iterate_prev(self, checker: Callable[[Interval], bool] = None) -> Interval:
        """
        Iterates 'prev' intervals with protection from wrong order of nodes.
        :param checker: Lambda to retrurn True if required state is reached.
        :return: `Interval` where loop finished by given checker or first interval.
        """
        interval = self
        while interval.prev:
            tmp = interval.prev
            if tmp.start_time >= interval.start_time:
                raise ValueError(f"Wrong interval.prev found: f{tmp} expected to be before current f{interval}.")
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

    def intervals_to_strings(self, offset: int=0, count: int=0, from_earliest=True, debug=False):
        interval = self
        if from_earliest:
            interval = self.iterate_prev()
        elif offset != 0:
            i = 0
            if offset > 0:
                while interval := interval.next:
                    if i >= offset:
                        break
                    i += 1
            else:
                while interval := interval.prev:
                    if i >= offset:
                        break
                    i += 1
        result = []
        if interval:
            result.append(interval.to_str(debug))
            i = 1
            while interval := interval.next:
                if i == count:  # If count < 1 then don't limit at all.
                    break
                result.append(interval.to_str(debug))
                i += 1
        return result

    def intervals_to_string(self, offset: int=0, count: int=0, from_earliest=True, debug=False) -> str:
        intervals_in_strings = self.intervals_to_strings(offset, count, from_earliest, debug)
        return str(len(intervals_in_strings)) + " intervals:\n  " + "\n  ".join(intervals_in_strings)

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

    def is_match_start_time(self, time: datetime.datetime, tolerance: datetime.timedelta) -> bool:
        return abs(self.start_time - time) <= tolerance

    def is_match_end_time(self, time: datetime.datetime, tolerance: datetime.timedelta) -> bool:
        return abs(self.end_time - time) <= tolerance

    def is_time_inside(self, time: datetime.datetime, tolerance: datetime.timedelta) -> bool:
        return self.start_time - tolerance <= time and self.end_time + tolerance >= time

    def is_event_inside(self, event: Event, tolerance: datetime.timedelta) -> bool:
        return self.start_time - tolerance <= event.timestamp\
               and self.end_time + tolerance >= event.timestamp + event.duration

    def is_time_after_and_not_adjacent(self, time: datetime.datetime, tolerance: datetime.timedelta) -> bool:
        return self.end_time + tolerance < time

    def find_closest(self, date: datetime.datetime, tolerance: datetime.timedelta, by_end_time: bool) -> Interval:
        """
        Finds interval with selectively `end_time` or `start_time` <= (i.e. earlier than) specified time.
        :param date: Time to search closest interval.
        :param by_end_time: Flag to search interval `end_time`. Otherwise searches by `start_time`.
        :return: Interval before or `None` if all intervals are after specified time.
        """
        interval = self
        # First check if date is later than current interval (need search it later intervals).
        if (interval.end_time if by_end_time else interval.start_time) - tolerance < date:
            return self.iterate_next(lambda x: (x.end_time if by_end_time else x.start_time) + tolerance >= date)
        else:  # Date is before current interval (need search in previous intervals).
            return self.iterate_prev(lambda x: (x.end_time if by_end_time else x.start_time) - tolerance < date)

    def new_after(self, event: Event) -> Interval:
        """
        Creates new `Interval` from specified event and inserts it after current.
        Doesn't use event start time.
        :param event: Event which causes new interval creation.
        :return: Just created interval.
        """
        interval = Interval(self.end_time, event.timestamp + event.duration, self)
        interval.events.append(event)
        if self.next:
            self.next.prev = interval
        self.next = interval
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
        self.prev = interval
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
        self.next = interval
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
        event_end_time = event.timestamp + event.duration
        # First separate last part i.e. [0<-self->2][2<-self->3]. Only if it makes sense.
        if abs(self.start_time - event_end_time) > tolerance and abs(event_end_time - self.end_time) > tolerance:
            last_interval = Interval(event_end_time, self.end_time, self, self.next)
            last_interval.events.extend(self.events)
            self.next = last_interval
        # Next separate current interval with new at the end.
        return self.separate_new_at_end(event, tolerance)

    def _merge_with_next(self):
        self.end_time = self.next.end_time
        self.events.extend(self.next.events)
        self.next = self.next.next

    def merge_all_adjacent_with_same_name(self):
        interval = self.iterate_prev()
        while interval.next:
            if interval.end_time == interval.next.start_time and interval.name == interval.next.name:
                interval._merge_with_next()
            interval = interval.next


class Rule:
    def __init__(self, not_afk: bool = False, subrules: RulesHandler = None,
                 ignore: bool = False) -> None:
        """
        Constuctor.
        :param not_afk: Flag to convert "afk" state to "not_afk".
        :param subrules: Applies rules for different key in 'data' of event.
        :param ignore: Flag to ignore this activity for reports.
        """
        self.not_afk = not_afk
        self.subrules = subrules
        self.ignore = ignore


class RulesHandler:
    def __init__(self, key: str, rules: Dict[str, Rule], tolerance: datetime.timedelta=TOLERANCE) -> None:
        self.key = key
        self.tolerance = tolerance
        self.rules = dict((re.compile(k), v) for k, v in rules.items())

    def __repr__(self) -> str:
        return f"RulesHandler(key={self.key}, rules_len={len(self.rules)})"

    def get_event_name(self, event) -> Any:
        return event.data[self.key]

    def _span_event_on_few_intervals(self, event: Event, interval: Interval) -> Interval:
        """
        Recursively applies given event down to linked list of intervals.
        Assumes that event is started before or at the start of the given interval.
        :param event: Event to apply.
        :param interval: Interval which starts before or exactly on this event.
        :return: Interval on which event applying is over.
        """
        if interval.is_time_inside(event.timestamp + event.duration, self.tolerance):
            # Event ends inside this interval. Therefore stop recursion.
            interval = interval.separate_new_at_start(event, self.tolerance)
            # TODO update name
            return interval
        else:
            # Event covers current interval completely.
            interval.events.append(event)
            # TODO update name
            if not interval.next:
                LOG.debug(f"  Skipping part of {event_to_str(event)} event spanning after 'afk' watcher events "
                      "- computer didn't work those time.")
                return interval
            return self._span_event_on_few_intervals(event, interval.next)

    def apply_events(self, events: List[Event], interval: Interval) -> Dict[str, int]:
        """
        Applies events to the specified linked list of intervals.
        Expected that this list contains from "afk" and "watchdog" watchers, i.e. events out of bounds of existing
        intervals should be ignored.
        :param events: List of events to apply.
        :param interval: Node of linked list to apply events onto. Out parameter.
        :return: Dictionary of metrics obtained on applying events.
        """
        # Metrics to measure each bucket importance.
        metrics = {
            'cnt_skipped_too_short': 0,
            'cnt_skipped_out_of_afk': 0,
            'cnt_handled': 0,
            'cnt_match_interval': 0,
            'cnt_inside_interval': 0,
            'cnt_split_one_interval': 0,
            'cnt_split_few_intervals': 0,
        }
        for event in events:
            # Skip too short events.
            if event.duration < self.tolerance:
                metrics['cnt_skipped_too_short'] += 1
                continue
            # Find previous interval to merge into.
            interval = interval.find_closest(event.timestamp, self.tolerance, by_end_time=False)
            # If event happend before all previous events/intervals then skip it.
            if not interval:
                LOG.debug("  Skipping event %s happened before 'AFK' events "
                          "- user didn't work those time.", event_to_str(event))
                metrics['cnt_skipped_out_of_afk'] += 1
                continue
            # Compare closest interval boundaries with event boundaries and update intervals linked list.
            # Following cases are possible:
            # [-----]    <- interval boundaries, next will be event ones
            # -----------------------------------------------------------
            # [--]  |        <- split interval with new on the start
            # [-----]        <- just add event to interval
            # [-------]      <- span event on few intervals
            # | [-] |        <- split interval on 3
            # | [---]        <- split interval with new on the end
            # | [-----]      <- split current interval with new on the end and span event on few intervals
            # |     [--]     <- skip event as happend in AFK time
            # |     |  [---] <- skip event as happend in AFK time
            compare_event_start_with_interval_end = interval.compare_with_time(event.timestamp, self.tolerance, False)
            # If event is after/later closest found interval then it is "out of AFK" so skip it (both 'skip' cases).
            if compare_event_start_with_interval_end >= 0:
                LOG.debug("  Skipping %s event happened after 'AFK' events "
                          "- user didn't work those time.", event_to_str(event))
                metrics['cnt_skipped_out_of_afk'] += 1
                continue
            # Event have to contribute into intervals at least partially. So do it.
            compare_starts = interval.compare_with_time(event.timestamp, self.tolerance, True)
            compare_ends = interval.compare_with_time(event.timestamp + event.duration, self.tolerance, False)
            if compare_starts == 0:
                if compare_ends < 0:
                    interval = interval.separate_new_at_start(event, self.tolerance)
                    # TODO update name
                    metrics['cnt_split_one_interval'] += 1
                elif compare_ends == 0:
                    interval.events.append(event)
                    metrics['cnt_match_interval'] += 1
                    # TODO update name
                else:
                    interval = self._span_event_on_few_intervals(event, interval)
                    metrics['cnt_split_few_intervals'] += 1
            elif compare_ends < 0:
                # TODO makes negative intervals
                LOG.info("before %s %s", event_to_str(event), interval.intervals_to_string(offset=-1, count=3,
                                                                                           from_earliest=False))
                interval = interval.separate_new_at_middle(event, self.tolerance)
                LOG.info("after %s %s", event_to_str(event), interval.intervals_to_string(offset=-2, count=5,
                                                                                          from_earliest=False))
                # TODO update names for all 3
                metrics['cnt_inside_interval'] += 1
            elif compare_ends == 0:
                interval = interval.separate_new_at_end(event, self.tolerance)
                # TODO update name
                metrics['cnt_split_one_interval'] += 1
            else:
                interval = interval.separate_new_at_end(event, self.tolerance)
                if interval.next:
                    interval = self._span_event_on_few_intervals(event, interval)
                    metrics['cnt_split_few_intervals'] += 1
                else:
                    metrics['cnt_split_one_interval'] += 1
            metrics['cnt_handled'] += 1
        return metrics


def report_from_buckets(awc: aw_client.ActivityWatchClient, start_time: datetime.datetime, end_time: datetime.datetime,
                        buckets: List[str], rules: Dict[str, RulesHandler], tolerance: datetime.timedelta) -> Interval:
    """
    Gets events from specified buckets, prints report by them and returns linked list of `Interval`-s.
    :param awc: ActivityWatch client to use.
    :param start_time: Start time for the report.
    :param end_time: End time for the report.
    :param buckets: List of buckets to report events from.
    :param rules: User-specific map of rules to make report with.
    :param tolerance: Tolerance to match events boundaries to intervals.
    :return: Last touched `Interval` from linked list built from events.
    """

    # Expected output:
    # A) Worked at intervals: A, B, C - Total ZZ
    # B) TOTAL not afk = Z
    # C) Unknown = ZZZ
    # D) 2h = window "xxx"
    #   1h = browser "XXX" site
    #   1.75h = IDEA on projectPath
    #   0.25 = meeting in Zoom from XXX to YYY
    #   0.25 = meeting in Slack with ZZZ from XXX to YYY
    #   0.25 = Skype conversations in durations A, B, C
    cur_interval: Interval = None

    # 1) Find out first "aw-watcher-afk" bucket to find out "not_afk" intervals as a main intervals to determine
    # activity by (we still need in "afk" cases for '"not_afk": True').
    # "aw-watcher-afk" - active watcher, it doesn't show "active" when user is not.
    # But is doesn't catch mic on meetings so may produce wrong AFK status.
    # Data contains the only 'status' key with "afk" and "not-afk" values. Make sense extract 100% activities in
    # periods from "not-afk" to "afk" and in remained intervals if other watchers shows "meeting".
    # data={status: [afk, not-afk]}
    bucket_id = next((x for x in buckets.keys() if x.startswith("aw-watcher-afk")), None)
    if bucket_id:
        events: List[Event] = awc.get_events(bucket_id, start=start_time, end=end_time)
        events = sorted(events, key=lambda e: e.timestamp)
        for event in events:
            if not cur_interval:
                cur_interval = Interval(event.timestamp, event.timestamp + event.duration)
                cur_interval.events.append(event)
            else:
                interval_to_put_after = cur_interval.find_closest(event.timestamp, tolerance, by_end_time=True)
                if interval_to_put_after:
                    # If there is adjacent interval before then create new interval based on event.
                    cur_interval = interval_to_put_after.new_after(event)
                else:
                    # If no adjacent intervals then create new interval with event time bounds after cur_interval.
                    # Note that event are sorted so cur_interval is expected to be before in time.
                    tmp = Interval(event.timestamp, event.timestamp + event.duration, cur_interval, None)
                    tmp.events.append(event)
                    cur_interval.prev = tmp
                    cur_interval = tmp
            cur_interval.name = event.data['status']
    else:
        LOG.info("No AFK bucket found. Stopping here - no more events expected.")
        return None
    if cur_interval:
        cur_interval.merge_all_adjacent_with_same_name()  # Remove duplicates.
        LOG.info("By AFK found %s", cur_interval.intervals_to_string(debug=True))
    else:
        LOG.info("No AFK events found in %s..%s. Stopping here - no more events expected.", start_time, end_time)
        return None

    # 2) "aw-stopwatch" events are highest priority. Just create intervals. Note that it may be an empty bucket.
    # Create intervals above of previous intervals because stopwatch intervals are highest priority.
    # "aw-stopwatch" - active watcher, it is managed by user intentionally so most priority.
    # Make sense measure duration per unique label from "running=true" to "running=false".
    # data={label: str, running: bool}
    events: List[Event] = awc.get_events("aw-stopwatch", start=start_time, end=end_time)
    if events:
        for event in events:  # TODO implement.
            if cur_interval:
                # Events are sorted here.
                interval = cur_interval.new_after(event)
            else:
                interval = Interval(event.timestamp, event.timestamp + event.duration)
                interval.events.append(event)
            cur_interval = interval
        LOG.info("After stopwatch got %s", cur_interval.intervals_to_string())
    else:
        LOG.info("No stopwatch events found.")

    # 3) Iterate through remained buckets
    for bucket_id in buckets.keys():
        rule = next((x for x in rules.keys() if bucket_id.startswith), None)
        if rule is None:
            LOG.info("Skipping '%s' bucket as not described in rules.", bucket_id)
            continue
        else:
            rules_handler = next((v for k, v in rules.items() if bucket_id.startswith(k)), None)
            if rules_handler:
                rules_handler.tolerance = tolerance  # TODO handle it more beautiful.
                events = awc.get_events(bucket_id, start=start_time, end=end_time)
                if events:
                    LOG.info("Handling '%s' %d events with %s", bucket_id, len(events), rules_handler)
                    metrics = rules_handler.apply_events(events, cur_interval)
                    metrics_strings = (f"{k}: {v}" for k, v in metrics.items())
                    LOG.info("In result got %d intervals. Details:\n  %s", cur_interval.get_count(),
                             "\n  ".join(metrics_strings))
                else:
                    LOG.info("'%s' bucket doesn't have events in %s..%s.", bucket_id, start_time, end_time)

    # 4) Gather and print metrics.
    # B: It is simple "aw-watcher-afk" data with not_afk.
    total_not_afk: float = 0.0
    # A: It are intervals where not_afk or '"not_afk": True' or "aw-stopwatch" data.
    aw_active_intervals: List[Tuple[datetime.datetime, datetime.datetime]] = []
    # D: It is A with more specific activity aggregated by this activity.
    tasks: Dict[str, float] = {}
    # TODO implement
    return cur_interval


# Min duration one activity events sum to show.
MIN_DURATION_SEC = 15 * 60  # 0.25 hours
# Keys matches bucket names start. If there will be few buckets with ID starting from key then all will be handled.
# If few keys match the same bucket then only first RulesHandler will be applied to the bucket events.
RULES = {
    # Passive watcher, always provides value, even if user AFK. But "change value" event 100% shows activity.
    # data={app: str, title: str}.
    "aw-watcher-window": RulesHandler("app", {
        "zoom": Rule(not_afk=True),
        "Slack": Rule(subrules=RulesHandler("title", {
            # BTW it is not the only case.
            ".*screen share": Rule(not_afk=True),
        })),
        # Skype doesn't provide info that it is a meeting.
        "Skype": Rule(not_afk=True),
        "unknown": Rule(),  # Means that window manager was unable to gather data. Discord,
        "flameshot": Rule(),  # Screenshot tool.
        "jetbrains-idea": Rule(),  # IDE. TODO ~2 seconds after "afk" watcher events -> are not counted!
        "Double Commander": Rule(),  # File manager.
        "smplayer": Rule(),  # Video player.
        "FeatherPad": Rule(),  # Text editor.
    }),
    # Passive watcher, always provides value, even if user AFK.
    # But "change value" event most probably shows activity (excluding web pages which change title periodically).
    # data={url: str, title: str, audible: bool, incognito: bool, tabCount: int}.
    "aw-watcher-web": RulesHandler("url", {
        "https://vimbox.skyeng.ru/.*": Rule(not_afk=True, ignore=True),
        "https://gitlab.akvelon.net:9443/.*": Rule(),
        "https://akvelon.atlassian.net/wiki/.*": Rule(),
        "https://gitlab.intapp.com/.*": Rule(),
        "https://wiki.intapp.com/wiki/.*": Rule(),
    }),
    # Passive watcher, always provides value, even if user AFK. But "change value" event 100% shows activity.
    # data={file: str, projectPath: str, language: str, editor: const, editorVersion: const, eventType: const}
    "aw-watcher-idea": RulesHandler("file", {}),
}


def main():

    # TODO ask date
    # daystart = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
    # dayend = daystart + datetime.timedelta(days=1)
    # TODO support logging levels.

    awc = aw_client.ActivityWatchClient("activity_merger")
    buckets = awc.get_buckets()
    LOG.info(f"Buckets: {buckets.keys()}")
    report_from_buckets(awc, datetime.datetime(2022, 2, 11), datetime.datetime(2022, 2, 12), buckets, RULES, TOLERANCE)


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
