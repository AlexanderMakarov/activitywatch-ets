#!/usr/bin/env python3
from __future__ import annotations  # https://stackoverflow.com/a/33533514/1535127
import datetime
from difflib import restore
import aw_client
import socket
from typing import Any, List, Dict, Tuple
from aw_transform.filter_period_intersect import _intersecting_eventpairs
from aw_core.models import Event
import re

from numpy import number


local_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo


def event_to_str(event):
    if not event:
        return 'null'
    return str(event.data)


class Interval:
    """
    Linked list node with start and end time.
    """

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, prev=None,
                 next=None) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.prev = prev
        self.next = next
        # Note that events need to add in a custom way, they often inherits from base Interval.
        self.events = []
        self.name = None  # Depends from Interval purpose.

    def print_interval(self):
        description = self.name
        if not description and len(self.events) > 0:
            description = f"{len(self.events)} events, last={event_to_str(self.events[-1])}"
        print(f"{self.start_time.astimezone(local_timezone):%H:%M:%S}"
              f"..{self.end_time.astimezone(local_timezone):%H:%M:%S}: {description}")

    def get_earliest(self) -> Interval:
        interval = self
        while interval.prev:
            interval = interval.prev
        return interval

    def print_intervals(self, from_earliest=True):
        interval = self.get_earliest() if from_earliest else self
        if interval:
            interval.print_interval()
            while interval := interval.next:
                interval.print_interval()

    def is_match_start_time(self, time: datetime.datetime, tolerance_sec: float) -> bool:
        return abs(self.start_time - time) <= tolerance_sec

    def is_match_end_time(self, time: datetime.datetime, tolerance_sec: float) -> bool:
        return abs(self.end_time - time) <= tolerance_sec

    def find_closest(self, date: datetime.datetime, by_end_time: bool) -> Interval:
        """
        Finds interval with either `end_time` or `start_time` <= specified time.
        :param date: Time to search interval before.
        :param by_end_time: Flag to search interval `end_time`. Otherwise searches by `start_time`.
        :return: Interval before or `None` if all intervals are after specified time.
        """
        interval = self
        # First check if interval later or before.
        if (interval.start_time if by_end_time else interval.end_time) < date:
            while tmp := interval.next:
                if (tmp.start_time if by_end_time else tmp.end_time) >= date:
                    return interval
                interval = tmp  # Shift to interval later as possible result.
        else:  # Search closest in intervals before.
            while tmp := interval.prev:
                if (tmp.start_time if by_end_time else tmp.end_time) < date:
                    return tmp
        return None  # Date is before first interval in the list.

    def new_after(self, event: Event) -> Interval:
        """
        Creates new Interval from specified event and inserts it after current.
        :return: Just created interval.
        """
        interval = Interval(event.timestamp, event.timestamp + event.duration, self)
        interval.events.append(event)
        if self.next:
            self.next.prev = interval
        self.next = interval
        return interval

    def separate_from_start(self, event) -> Interval:
        interval = Interval(event.timestamp, event.timestamp + event.duration, self.prev, self)
        interval.events.extend(self.events)
        interval.events.append(event)
        self.start_time = interval.end_time
        self.prev = interval
        return interval

    def separate_from_end(self, event) -> Interval:
        interval = Interval(event.timestamp, event.timestamp + event.duration, self, self.next)
        interval.events.extend(self.events)
        interval.events.append(event)
        self.end_time = interval.start_time
        self.next = interval
        return interval

    def _merge_with_next(self):
        self.end_time = self.next.end_time
        self.events.extend(self.next.events)
        self.next = self.next.next

    def merge_all_adjacent_with_same_name(self):
        interval = self.get_earliest()
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
    def __init__(self, key: str, rules: Dict[str, Rule]) -> None:
        self.key = key
        self.rules = dict((re.compile(k), v) for k, v in rules.items())

    def get_event_name(self, event) -> Any:
        return event.data[self.key]

    def apply_events(events: List[Event], cur_interval: Interval, tolerance_sec: float = 1.0):
        # Better start from earliest interval because events are sorted usually.
        cur_interval = cur_interval.get_earliest()
        for event in events:
            # Find previous interval to merge into.
            interval = cur_interval.find_closest(event.timestamp, by_end_time=False)
            if not interval:
                print(f"  Skipping {event_to_str(event)} event happened out of 'afk' watcher "
                      "- computer didn't work this time.")
                continue
            # There are 3 cases during merging events by time (with tolerance):
            # 1) Event matches previous interval. Rare case.
            #    Just add it to the interval and change its name.
            # 2) Event matches only start_time or end_time of the interval.
            #    Separate interval on 2 in this case.
            # 3) Event covers 2 or more intervals, often partially.
            #    Mix of 2 cases above.
            if interval.is_match_start_time(event.timestamp, tolerance_sec):
                if interval.is_match_end_time(event.timestamp + event.duration, tolerance_sec):
                    interval.events.append(event)
                else:
                    interval = interval.separate_from_start(event)
            elif interval.is_match_end_time(event.timestamp + event.duration, tolerance_sec):
                interval = interval.separate_from_end(event)
            else:
                pass  # TODO implement
            # Update 'name' on new event(s)


def report_from_buckets(awc, start_time: datetime.datetime, end_time: datetime.datetime, buckets, rules) -> Interval:
    """
    Gets events from specified buckets, prints report by them and returns linked list of `Interval`-s.
    :param awc: ActivityWatch client to use.
    :param start_time: Start time for the report.
    :param end_time: End time for the report.
    :param buckets: List of buckets to report events from.
    :param rules: User-specific map of rules to make report with.
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
        events: List[Event] = awc.get_events(
            bucket_id, start=start_time, end=end_time)
        for event in events:
            if not cur_interval:
                cur_interval = Interval(event.timestamp, event.timestamp + event.duration)
                cur_interval.name = event.data['status']
                cur_interval.events.append(event)
            else:
                interval_to_put_after = cur_interval.find_closest(event.timestamp, by_end_time=True)
                if interval_to_put_after:
                    interval_to_put_after.new_after(event)
                else:
                    interval = Interval(event.timestamp, event.timestamp + event.duration, None, cur_interval)
                interval.events.append(event)
                interval.name = event.data['status']
                cur_interval.prev = interval
                cur_interval = interval
    else:
        print(f"No AFK bucket found. Stopping here - no more events expected.")
        return None
    if cur_interval:
        cur_interval.merge_all_adjacent_with_same_name()  # Remove duplicates.
        print("By AFK found:")
        cur_interval.print_intervals()
    else:
        print(f"No AFK events found in {start_time}..{end_time}. Stopping here - no more events expected.")
        return None

    # 2) "aw-stopwatch" events are highest priority. Just create intervals. Note that it may be an empty bucket.
    # Create intervals above of previous intervals because stopwatch intervals are highest priority.
    # "aw-stopwatch" - active watcher, it is managed by user intentionally so most priority.
    # Make sense measure duration per unique label from "running=true" to "running=false".
    # data={label: str, running: bool}
    events: List[Event] = awc.get_events("aw-stopwatch", start=start_time, end=end_time)
    if events:
        for event in events:  # TODO change
            if cur_interval:
                # Events are sorted here.
                interval = cur_interval.new_after(event)
            else:
                interval = Interval(event.timestamp, event.timestamp + event.duration)
                interval.events.append(event)
            cur_interval = interval
        print("With stopwatch found:")
        cur_interval.print_intervals()
    else:
        print("No stopwatch events found.")

    # 3) iterate through remained buckets
    for bucket_id in buckets.keys():
        rule = next((x for x in rules.keys() if bucket_id.startswith), None)
        if rule is None:
            print(f"Skipping '{bucket_id}' bucket as not described in RULES.")
            continue
        else:
            rules_handler = next((v for k, v in RULES.items() if bucket_id.startswith(k)), None)
            if rules_handler:
                events = awc.get_events(bucket_id, start=start_time, end=end_time)
                if events:
                    print(f"Handling '{bucket_id}' {len(events)} events with {rules_handler}")
                    rules_handler.apply_events(events, cur_interval)  # TODO pass tolerance
                else:
                    print(f"'{bucket_id} bucket doesn't have events in {start_time}..{end_time}.")

    # B: It is simple "aw-watcher-afk" data with not_afk.
    total_not_afk: float = 0.0
    # A: It are intervals where not_afk or '"not_afk": True' or "aw-stopwatch" data.
    aw_active_intervals: List[Tuple[datetime.datetime, datetime.datetime]] = []
    # D: It is A with more specific activity aggregated by this activity.
    tasks: Dict[str, float] = {}
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
        "jetbrains-idea": Rule(),  # IDE.
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

    awc = aw_client.ActivityWatchClient("activity_merger")
    buckets = awc.get_buckets()
    print(f"Buckets: {buckets.keys()}")
    report_from_buckets(awc, datetime.datetime(2022, 2, 11),
                        datetime.datetime(2022, 2, 12), buckets, RULES)


if __name__ == '__main__':
    main()
