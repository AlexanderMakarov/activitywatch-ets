#!/usr/bin/env python3
import datetime
from difflib import restore
import aw_client
import socket
from typing import Any, List, Dict, Tuple
from aw_transform.filter_period_intersect import _intersecting_eventpairs
from aw_core.models import Event


MIN_DURATION_MINUTES = 15  # 0.25 hours
RULES = {
    # Passive watcher, always provides value, even if user AFK. But "change value" event 100% shows activity.
    # data={app: str, title: str}.
    "aw-watcher-window": {
        "_key": "app",
        "zoom": {"not_afk": True},
        "Slack": {"subrules": {  # Allows make rules for different keys.
            "title": {
                "*.screen share": {"regex": True, "not_afk": True}  # BTW it is not the only case.
            },
        }},
        "Skype": {"not_afk": True},  # Skype doesn't provide info that it is a meeting.
        "unknown": {},  # Means that window manager was unable to gather data. Discord,
        "flameshot": {},  # Screenshot tool.
        "jetbrains-idea": {},  # IDE.
        "Double Commander": {},  # File manager.
        "smplayer": {},  # Video player.
        "FeatherPad": {},  # Text editor.
    },
    # Passive watcher, always provides value, even if user AFK.
    # But "change value" event most probably shows activity (excluding web pages which change title periodically).
    # data={url: str, title: str, audible: bool, incognito: bool, tabCount: int}.
    "aw-watcher-web": {
        "_key": "url",
        "https://vimbox.skyeng.ru/.*": {"regex": True, "not_afk": True, "ignore": True},
        "https://gitlab.akvelon.net:9443/.*": {"regex": True},
        "https://akvelon.atlassian.net/wiki/.*": {"regex": True},
        "https://gitlab.intapp.com/.*": {"regex": True},
        "https://wiki.intapp.com/wiki/.*": {"regex": True},
    },
    # Passive watcher, always provides value, even if user AFK. But "change value" event 100% shows activity.
    # data={file: str, projectPath: str, language: str, editor: const, editorVersion: const, eventType: const}
    "aw-watcher-idea": {
        "_key": "file",
    }
}

# "aw-stopwatch" - active watcher, it is managed by user intentionally so most priority.
# Make sense measure duration per unique label from "running=true" to "running=false".
# data={label: str, running: bool}
# "aw-watcher-afk" - active watcher, it doesn't show "active" when user is not.
# But is doesn't catch mic on meetings so may produce wrong AFK status.
# Data contains the only 'status' key with "afk" and "not-afk" values. Make sense extract 100% activities in
# periods from "not-afk" to "afk" and in remained intervals if other watchers shows "meeting".
# data={status: [afk, not-afk]}


def sort_bucket_by_rules(bucket: List[Event], intervals: List[Any]):
    pass


def event_to_str(event):
    if not event:
        return 'null'
    return str(event.data)


class Interval:
    """
    Linked list node with start and end time.
    """

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, prev = None,
                 next = None) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.prev = prev
        self.next = next
        self.events = []  # Note that events need to add in a custom way, they often inherits from base Interval.
        self.name = None  # Depends from Interval purpose.

    def print_interval(self):
        description = self.name
        if not description and len(self.events) > 0:
            description = f"{len(self.events)} events, last={event_to_str(self.events[-1])}"
        print(f"{self.start_time}..{self.end_time}: {description}\n")

    def print_intervals(self, from_first=True):
        start = self
        if from_first:
            while start := self.prev:
                pass
        if start:
            start.print_interval()
        while interval := start.next:
            interval.print_interval()

    def new_after(self, event: Event) -> Interval:
        """
        Creates new Interval from specified event and inserts it after current.
        :return: Just created interval.
        """
        interval = Interval(event.start, event.end, self)
        interval.events.append(event)
        if self.next:
            self.next.prev = interval
        self.next = interval
        return interval

    def find_closest_interval_before(self, date: datetime.datetime):
        result = self
        if result.end_time < date:  # Search closest in intervals later.
            while tmp := result.next:
                if tmp.end_time > date:
                    return result
                result = tmp  # Shift to interval later as possible result.
        else:  # Search closest in intervals before.
            while tmp := result.prev:
                if tmp.end_time < date:
                    return tmp
        return None  # Date is before first interval in the list.


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

    # 1) Iterate few "aw-watcher-afk" buckets to find out "not_afk" intervals as a main intervals to determine
    # activity by (we still need in "afk" cases for '"not_afk": True').
    # Evict previous "afk"-s (consider case when 1 PC reports "afk" while you are working on the 2 PC).
    for bucket_id in buckets.keys():
        if bucket_id.startswith("aw-watcher-afk"):
            events: List[Event] = awc.get_events(bucket_id, start=start_time, end=end_time)
            for event in events:
                if not cur_interval:  # TODO merge from few PC-s
                    cur_interval = Interval(event.start, event.end)
                    cur_interval.events.append(event)
                else:
                    interval_to_put_after = cur_interval.find_closest_interval_before(event.start)
                    if interval_to_put_after:
                        interval_to_put_after.new_after(event)
                    else:
                        interval = Interval(event.start, event.end, None, cur_interval)
                    interval.events.append(event)
                    cur_interval.prev = interval
                    cur_interval = interval
    if cur_interval:
        print("By AFK found:")
        cur_interval.print_intervals()
    else:
        print(f"No AFK events found in {start_time}..{end_time}. Stopping here - no more events expected.")
        return None

    # 2) "aw-stopwatch" events are highest priority. Just create intervals. Note that it may be an empty bucket.
    # Create intervals above of previous intervals because stopwatch intervals are highest priority.
    events: List[Event] = awc.get_events("aw-stopwatch", start=start_time, end=end_time)
    if events:
        for event in events:  # TODO change
            if cur_interval:
                interval = cur_interval.new_after(event)  # Events are sorted here.
            else:
                interval = Interval(event.start, event.end)
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
            print(f"Skipping {bucket_id} bucket as not supported in RULES.")
            bucket_rules = next((x for x in RULES.keys() if bucket_id.startswith(x)), None)
            if bucket_rules:
                pass

    # B: It is simple "aw-watcher-afk" data with not_afk.
    total_not_afk: float = 0.0
    # A: It are intervals where not_afk or '"not_afk": True' or "aw-stopwatch" data.
    aw_active_intervals: List[Tuple[datetime.datetime, datetime.datetime]] = []
    # D: It is A with more specific activity aggregated by this activity.
    tasks: Dict[str, float] = {}
    return cur_interval


def main():

    # TODO ask date
    # daystart = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
    # dayend = daystart + datetime.timedelta(days=1)

    awc = aw_client.ActivityWatchClient("activity_merger")
    buckets = awc.get_buckets()
    print(f"Buckets: {buckets.keys()}")
    report_from_buckets(awc, datetime.datetime(2022, 2, 11), datetime.datetime(2022, 2, 12), buckets, RULES)


if __name__ == '__main__':
    main()
