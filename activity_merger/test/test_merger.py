import unittest
import datetime
import logging
from unittest.mock import MagicMock, patch, call
from typing import Any, List, Dict, Tuple
from parameterized import parameterized
from aw_core import Event as AWEvent

from . import build_datetime, build_timedelta

from ..domain.input_entities import Event
from ..domain.interval import Interval
from ..domain.metrics import Metrics, Metric
from ..domain.merger import apply_events, report_from_buckets, sort_merge_convert_raw_events
from ..config.config import LOG


def build_intervals(data: List[Tuple[int, int, List[Event]]]) -> Interval:
    """
    Build linked list of intervals.
    :param data: List of tuples (start, duration, list of events).
    :return: Last created `Interval`.
    """
    interval = None
    for (start, duration, events) in data:
        if not interval:
            interval = Interval(build_datetime(start, day=1), build_datetime(start + duration, day=1))
        else:
            tmp = Interval(build_datetime(start, day=1), build_datetime(start + duration, day=1), interval)
            interval.next = tmp
            interval = tmp
        interval.events.extend(events)
    return interval


def build_AWEvent(timestamp_hours: int, duration_hours: int, data: Any = None) -> AWEvent:
    return AWEvent(timestamp=datetime.datetime(2023, 1, 1, timestamp_hours, tzinfo=datetime.timezone.utc),
                   duration=datetime.timedelta(hours=duration_hours), data=data)


def build_Event(timestamp_hours: int, duration_hours: int, data: Any = None) -> Event:
    return Event("b1", datetime.datetime(2023, 1, 1, timestamp_hours, tzinfo=datetime.timezone.utc),
                 datetime.timedelta(hours=duration_hours), data)


LOG = logging.getLogger("activity_merger.config.config")
start_time = build_datetime(1, day=1)
end_time = build_datetime(2, day=1)
AFK_BUCKET_ID = "aw-watcher-afk-foo"
AFK_BUCKET_ID2 = "aw-watcher-afk-foo2"
SOME_BUCKET_ID = "some_bucket_id"
SOME_BUCKET_ID2 = "some_bucket_id2"
awc = MagicMock()
timedelta_0 = build_timedelta(0)
timedelta_1 = build_timedelta(1, True)
timedelta_2 = build_timedelta(2, True)
timedelta_3 = build_timedelta(3, True)
timedelta_4 = build_timedelta(4, True)
timedelta_5 = build_timedelta(5, True)
timedelta_6 = build_timedelta(6, True)
timedelta_8 = build_timedelta(8, True)
event_1l0 = Event(AFK_BUCKET_ID, start_time, timedelta_0, 'event_1l0')
event_1 = Event(AFK_BUCKET_ID, start_time, timedelta_1, 'event_1')
event_1_bucket2 = Event(AFK_BUCKET_ID2, start_time, timedelta_1, 'event_1_bucket2')
inerval_for_1_event = build_intervals([(1, 1, [event_1])])
event_2 = Event(AFK_BUCKET_ID, build_datetime(2, day=1), timedelta_1, 'event_2')
event_2l0 = Event(AFK_BUCKET_ID, build_datetime(2, day=1), timedelta_0, 'event_2l0')
event_2_bucket2 = Event(AFK_BUCKET_ID2, build_datetime(2, day=1), timedelta_1, 'event_2_bucket2')
event_3 = Event(AFK_BUCKET_ID, build_datetime(3, day=1), timedelta_1, 'event_3')
event_3_bucket2 = Event(AFK_BUCKET_ID2, build_datetime(3, day=1), timedelta_1, 'event_3_bucket2')
interval_for_3_consecutive_events = build_intervals([(1, 1, [event_1]), (2, 1, [event_2]), (3, 1, [event_3])])
event_4 = Event(AFK_BUCKET_ID, build_datetime(4, day=1), timedelta_1, 'event_4')
event_4_bucket2 = Event(AFK_BUCKET_ID2, build_datetime(4, day=1), timedelta_1, 'event_4_bucket2')
event_5 = Event(AFK_BUCKET_ID, build_datetime(5, day=1), timedelta_1, 'event_5')
event_6 = Event(AFK_BUCKET_ID, build_datetime(6, day=1), timedelta_1, 'event_6')
event_1l2 = Event(AFK_BUCKET_ID, start_time, timedelta_2, 'event_1l2')
event_1l3 = Event(AFK_BUCKET_ID, start_time, timedelta_3, 'event_1l3')
event_2l2_data_from_1l2 = Event(AFK_BUCKET_ID, build_datetime(2, day=1), timedelta_2, event_1l2.data)
event_1l3_data_from_1l2 = Event(AFK_BUCKET_ID, start_time, timedelta_3, event_1l2.data)
event_1l4 = Event(AFK_BUCKET_ID, start_time, timedelta_4, 'event_1l4')
event_2l2 = Event(AFK_BUCKET_ID, build_datetime(2, day=1), timedelta_2, 'event_2l2')
event_1l8 = Event(AFK_BUCKET_ID, start_time, timedelta_8, 'event_1l8')

sevent_1 = Event(SOME_BUCKET_ID, start_time, timedelta_1, 'sevent_1')
sevent_2 = Event(SOME_BUCKET_ID, build_datetime(2, day=1), timedelta_1, 'sevent_2')
sevent_1l2 = Event(SOME_BUCKET_ID, start_time, timedelta_2, 'sevent_1l2')
sevent_1l3 = Event(SOME_BUCKET_ID, start_time, timedelta_3, 'sevent_1l3')
sevent_1l6 = Event(SOME_BUCKET_ID, start_time, timedelta_6, 'sevent_1l6')
sevent_2l2 = Event(SOME_BUCKET_ID, build_datetime(2, day=1), timedelta_2, 'sevent_2l2')
sevent_3 = Event(SOME_BUCKET_ID, build_datetime(3, day=1), timedelta_1, 'sevent_3')
sevent_3l2 = Event(SOME_BUCKET_ID, build_datetime(3, day=1), timedelta_2, 'sevent_3l2')
sevent_5l2 = Event(SOME_BUCKET_ID, build_datetime(5, day=1), timedelta_2, 'sevent_5l2')

sevent_3l2_bucket2 = Event(SOME_BUCKET_ID2, build_datetime(3, day=1), timedelta_2, 'sevent_3l2_bucket2')

class TestMerger(unittest.TestCase):

    @parameterized.expand([
        (
            "first event",
            [event_1],
            None,
            timedelta_0,
            False,
            build_intervals([(1, 1, [event_1])]),
            {'cnt_new_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "adjacent event",
            [event_2],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            False,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2])]),
            {'cnt_new_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "event far after",
            [event_3],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            False,
            build_intervals([(1, 1, [event_1]), (3, 1, [event_3])]),
            {'cnt_new_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping event same start time but smaller",
            [event_2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            True,
            build_intervals([(1, 2, [event_1l2])]),
            {'cnt_split_one_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlaping event same start and ent time",
            [event_1],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            True,
            build_intervals([(1, 1, [event_1, event_1])]),
            {'cnt_match_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping event same start time but bigger",
            [event_1l2],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            True,
            build_intervals([(1, 1, [event_1, event_1l2]), (2, 1, [event_1l2])]),
            {'cnt_split_few_intervals': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping event same end time",
            [event_2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            True,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2])]),
            {'cnt_split_one_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping event start time inside and end time outside",
            [event_2l2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            True,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2l2]), (3, 1, [event_2l2])]),
            {'cnt_split_few_intervals': 1, 'cnt_handled_events': 1},
        ),
    ])
    def test_apply_events_stopwatch(self, test_name: str, events: List[Event], interval: Interval,
            tolerance: datetime.timedelta, is_should_fail: bool,
            expected_interval: Interval, expected_metrics: Dict[str, int]):
        # Act
        actual_interval: Interval
        actual_metrics: Dict[str, int]
        if is_should_fail:
            with self.assertRaises(ValueError):
                actual_interval, actual_metrics = apply_events(events, interval, tolerance, True, True)
            return
        else:
            actual_interval, actual_metrics = apply_events(events, interval, tolerance, True, True)
        # Assert
        err_msg = f"'{test_name}' case failed."
        if expected_interval:
            self.assertListEqual(actual_interval.get_range(-10, 10), expected_interval.get_range(-10, 10), err_msg)
        else:
            self.assertIsNone(actual_interval, err_msg)
        self.assertDictEqual(
            {k: v for (k, v) in actual_metrics.items() if v > 0},
            {k: v for (k, v) in expected_metrics.items() if v > 0},
            err_msg
        )

    @parameterized.expand([
        (
            "first event",
            [event_1],
            None,
            timedelta_0,
            build_intervals([(1, 1, [event_1])]),
            {'cnt_new_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "adjacent event",
            [event_2],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2])]),
            {'cnt_new_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "event far after",
            [event_3],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1]), (3, 1, [event_3])]),
            {'cnt_new_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping with interval same start time but smaller",
            [event_2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2])]),
            {'cnt_split_one_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlaping with interval same start and ent time",
            [event_1],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1, event_1])]),
            {'cnt_match_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping with interval same start time but bigger",
            [event_1l2],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1, event_1l2]), (2, 1, [event_1l2])]),
            {'cnt_split_few_intervals': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping with interval same end time",
            [event_2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2])]),
            {'cnt_split_one_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping with interval start time inside and end time outside",
            [event_2l2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2l2]), (3, 1, [event_2l2])]),
            {'cnt_split_few_intervals': 1, 'cnt_handled_events': 1},
        ),
    ])
    def test_apply_events_afk(self, test_name: str, events: List[Event], interval: Interval,
            tolerance: datetime.timedelta,
            expected_interval: Interval, expected_metrics: Dict[str, int]):
        # Act
        actual_interval: Interval
        actual_metrics: Dict[str, int]
        actual_interval, actual_metrics = apply_events(events, interval, tolerance, True, False)
        # Assert
        err_msg = f"'{test_name}' case failed."
        if expected_interval:
            self.assertListEqual(actual_interval.get_range(-10, 10), expected_interval.get_range(-10, 10), err_msg)
        else:
            self.assertIsNone(actual_interval, err_msg)
        self.assertDictEqual(
            {k: v for (k, v) in actual_metrics.items() if v > 0},
            {k: v for (k, v) in expected_metrics.items() if v > 0},
            err_msg
        )

    @parameterized.expand([
        (
            "No buckets",
            [],
            {},
            timedelta_0,
            None,
            None,
            [],
            [("No buckets except AFK and/or Stopwatch found. Stopping here - no more events expected.",)]
        ),
        (
            "No AFK events",
            [[]],
            {AFK_BUCKET_ID: None},
            timedelta_0,
            None,
            None,
            [AFK_BUCKET_ID],
            [("No buckets except AFK and/or Stopwatch found. Stopping here - no more events expected.",)]
        ),
        (
            "1 AFK 0-length event",
            [[event_1l0]],
            {AFK_BUCKET_ID: None},
            timedelta_0,
            None,
            None,
            [AFK_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 0, ''),
                ("No AFK events found in %s..%s. Stopping here - no more events expected.", start_time, end_time)
            ]
        ),
        (
            "1 AFK event",
            [[event_1], []],
            {AFK_BUCKET_ID: None},
            timedelta_0,
            None,
            inerval_for_1_event,
            [AFK_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1')
            ]
        ),
        (
            "2 consecutive AFK events same bucket",
            [[event_1, event_2], []],
            {AFK_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2])]),
            [AFK_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2')
            ]
        ),
        (
            "2 similar AFK events same bucket",
            [[event_1, event_1]],
            {AFK_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1])]),
            [AFK_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1,
                 'cnt_new_interval: 1\n  cnt_handled_events: 1')
            ]
        ),
        (
            "2 similar AFK events different buckets",
            [[event_1], [event_1_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1, event_1_bucket2])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 1,
                 'cnt_new_interval: 1\n  cnt_handled_events: 2\n  cnt_match_interval: 1')
            ]
        ),
        (
            "2 consecutive AFK events different buckets",
            [[event_1], [event_2_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2_bucket2])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2')
            ]
        ),
        (
            "2 consecutive AFK events different buckets opposite order",
            [[event_2], [event_1_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2')
            ]
        ),
        (
            "4 consecutive AFK events one 0-length different buckets",
            [[event_1l2, event_2l0, event_2l2_data_from_1l2], [event_2_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l3_data_from_1l2]), (2, 1, [event_1l3_data_from_1l2, event_2_bucket2]),
                             (3, 1, [event_1l3_data_from_1l2])]),
            [AFK_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 3,
                 'cnt_new_interval: 1\n  cnt_handled_events: 2\n  cnt_inside_interval: 1')
            ]
        ),
        (
            "AFK events 1 and 3 in first bucket, 2 in second",
            [[event_1, event_3], [event_2_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2_bucket2]), (3, 1, [event_3])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 3, 'cnt_new_interval: 3\n  cnt_handled_events: 3')
            ]
        ),
        (
            "AFK events 2 in first bucket, 1 and 3 in second",
            [[event_2], [event_1_bucket2, event_3_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2]), (3, 1, [event_3_bucket2])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 3, 'cnt_new_interval: 3\n  cnt_handled_events: 3')
            ]
        ),
        (
            "AFK events 2 in first bucket, 1 and 4 in second",
            [[event_2], [event_1_bucket2, event_4_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2]), (4, 1, [event_4_bucket2])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 3, 'cnt_new_interval: 3\n  cnt_handled_events: 3')
            ]
        ),
        (
            "2 overlaping AFK events same bucket and data",
            [[event_1l2, event_2l2_data_from_1l2]],
            {AFK_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 3, [event_1l3_data_from_1l2])]),
            [AFK_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1,
                 'cnt_new_interval: 1\n  cnt_handled_events: 1')
            ]
        ),
        (
            "AFK events overlaping in buckets, 1 overlaps 1l2 plus far after 4",
            [[event_1l2], [event_1_bucket2, event_4_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l2, event_1_bucket2]), (2, 1, [event_1l2]),
                             (4, 1, [event_4_bucket2])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 3,
                 'cnt_new_interval: 2\n  cnt_handled_events: 3\n  cnt_split_one_interval: 1')
            ]
        ),
        (
            "AFK events overlaping in buckets, 1l4 overlaps 1 and 3",
            [[event_1l4], [event_1_bucket2, event_3_bucket2]],
            {AFK_BUCKET_ID: None, AFK_BUCKET_ID2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l4, event_1_bucket2]), (2, 1, [event_1l4]),
                             (3, 1, [event_1l4, event_3_bucket2]), (4, 1, [event_1l4])]),
            [AFK_BUCKET_ID, AFK_BUCKET_ID2],
            [
                ('  In result got %d intervals. Details:\n  %s', 4,
                 'cnt_new_interval: 1\n  cnt_handled_events: 3\n  cnt_inside_interval: 1\n  cnt_split_one_interval: 1')
            ]
        ),
        (
            "Some empty bucket",
            [[event_1], []],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("'%s' bucket doesn't have events in %s..%s.", SOME_BUCKET_ID, start_time, build_datetime(2, day=1)),
            ]
        ),
        (
            "Some event matches AFK",
            [[event_1], [sevent_1]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1, sevent_1])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_handled_events: 1\n  cnt_match_interval: 1'),
            ]
        ),
        (
            "Some event before AFK",
            [[event_2], [sevent_1]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_skipped_before_afk: 1'),
            ]
        ),
        (
            "Some event after AFK",
            [[event_1], [sevent_2]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_skipped_after_afk: 1'),
            ]
        ),
        (
            "Some event overlaps AFK's start",
            [[event_2l2], [sevent_1l2]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2l2, sevent_1l2]), (3, 1, [event_2l2])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
                ('  In result got %d intervals. Details:\n  %s', 2,
                 'cnt_handled_events: 1\n  cnt_split_one_interval: 1'),
            ]
        ),
        (
            "Some event overlaps AFK's end",
            [[event_1l2], [sevent_2l2]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, sevent_2l2])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
                ('  In result got %d intervals. Details:\n  %s', 2,
                 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
            ]
        ),
        (
            "Some event overlaps AFK's both sides",
            [[event_2], [sevent_1l3]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2, sevent_1l3])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
                ('  In result got %d intervals. Details:\n  %s', 1,
                 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
            ]
        ),
        (
            "Some event overlaps 2 AFK's both sides",
            [[event_2, event_4], [sevent_1l6]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2, sevent_1l6]), (4, 1, [event_4, sevent_1l6])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
                ('  In result got %d intervals. Details:\n  %s', 2,
                 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
            ]
        ),
        (
            "Some event is 2nd, inside AFK and there is next AFK",
            [[event_1l4, event_5], [sevent_1, sevent_3]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l4, sevent_1]), (2, 1, [event_1l4]), (3, 1, [event_1l4, sevent_3]),
                             (4, 1, [event_1l4]), (5, 1, [event_5])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 2),
                ('  In result got %d intervals. Details:\n  %s', 5,
                 'cnt_handled_events: 2\n  cnt_inside_interval: 1\n  cnt_split_one_interval: 1'),
            ]
        ),
        # (  # TODO add support of case below, for Stopwatch overlaps
        #     "Some events starts right after 1 AFK's and ovelaps with next",
        #     [[event_1, event_3], [sevent_2l2]],
        #     {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1, sevent_1l2]), (3, 1, [event_3, sevent_2l2])]),
        #     [AFK_BUCKET_ID, SOME_BUCKET_ID],
        #     [
        #         ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
        #         ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
        #         ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_handled_events: 1\n  cnt_split_one_interval: 1'),
        #     ]
        # ),
        (
            "2 some events inside 2 AFK's with different border",
            [[event_1, event_2l2], [sevent_1l2, sevent_3l2]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1, sevent_1l2]), (2, 1, [event_2l2, sevent_1l2]),
                             (3, 1, [event_2l2, sevent_3l2])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 2),
                ('  In result got %d intervals. Details:\n  %s', 3,
                 'cnt_handled_events: 2\n  cnt_split_few_intervals: 2'),
            ]
        ),
        (
            "Tolerance 1, Some event is 2nd, completely inside AFK and not adjacent",
            [[event_1l8], [sevent_2, sevent_5l2]],
            {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None},
            timedelta_1,
            None,
            build_intervals([(1, 2, [event_1l8, sevent_2]), (3, 2, [event_1l8]), (5, 2, [event_1l8, sevent_5l2]),
                             (7, 2, [event_1l8])]),
            [AFK_BUCKET_ID, SOME_BUCKET_ID],
            [
                ('  In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 2),
                ('  In result got %d intervals. Details:\n  %s', 4,
                 'cnt_handled_events: 2\n  cnt_inside_interval: 1\n  cnt_split_one_interval: 1'),
            ]
        ),
        # (  # TODO add support of case below - event starts, for Stopwatch overlaps
        #     "Few some events overlaps 2 AFK's both sides",
        #     [[event_2, event_4], [sevent_1l6], [sevent_3l2_bucket2]],
        #     {AFK_BUCKET_ID: None, SOME_BUCKET_ID: None, SOME_BUCKET_ID2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(2, 1, [event_2, sevent_1l6]), (4, 1, [event_4, sevent_1l6, sevent_3l2_bucket2])]),
        #     [AFK_BUCKET_ID, SOME_BUCKET_ID, SOME_BUCKET_ID2],
        #     [
        #         ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
        #         ("Applying '%s' bucket %d events:", SOME_BUCKET_ID, 1),
        #         ('  In result got %d intervals. Details:\n  %s', 2, 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
        #     ]
        # ),
    ])
    @patch.object(LOG, "info", MagicMock())
    # @patch.object(merger, "check_and_print_intervals", MagicMock())
    def test_report_from_buckets(self, test_name: str, get_events_lists_results: List[Event],
            buckets: Dict[str, object], tolerance: datetime.timedelta, expected_error_message: str,
            expected_interval: Interval, expected_buckets_called: List[str], expected_logs: List[str]):
        # Arrange
        awc.get_events.reset()
        awc.get_events.side_effect = get_events_lists_results
        LOG.info.reset_mock()
        # Act
        actual: Interval
        if expected_error_message:
            self.assertRaisesRegex(ValueError, expected_error_message,
                                   report_from_buckets, awc, start_time, end_time, buckets, tolerance)
            return
        else:
            actual = report_from_buckets(awc, start_time, end_time, buckets, tolerance)
        # Assert
        err_msg = f"'{test_name}' case failed."
        if expected_interval:
            self.assertListEqual(actual.get_range(-10, num=10), expected_interval.get_range(-10, num=10), err_msg)
        else:
            self.assertIsNone(actual, err_msg)
        awc.get_events.assert_has_calls([call(x, start=start_time, end=end_time) for x in expected_buckets_called])
        if expected_logs:
            LOG.info.assert_has_calls([call(*x) for x in expected_logs])
        else:
            LOG.info.assert_not_called()


    @parameterized.expand([
        (
            "adjucent_events_same_data",
            [build_AWEvent(1, 1, "foo"), build_AWEvent(2, 1, "foo")],
            [build_Event(1, 2, "foo")],
            {
                'raw events': Metric(2, 7200),
                'b1 merged the same data events': Metric(1, 3600.0),
                'events to handle': Metric(1, 7200.0),
            }
        ),
        (
            "adjucent_events_different_data",
            [build_AWEvent(1, 1, "foo"), build_AWEvent(2, 1, "bar")],
            [build_Event(1, 1, "foo"), build_Event(2, 1, "bar")],
            {
                'raw events': Metric(2, 7200.0),
                'events to handle': Metric(2, 7200.0),
            }
        ),
        (
            "events_far_away_same_data_and_mixed",
            [build_AWEvent(8, 1, "foo"), build_AWEvent(1, 1, "foo"), build_AWEvent(4, 1, "foo")],
            [build_Event(1, 1, "foo"), build_Event(4, 1, "foo"), build_Event(8, 1, "foo")],
            {
                'raw events': Metric(3, 10800.0),
                'events to handle': Metric(3, 10800.0),
            }
        ),
        (
            "events_far_away_different_data_and_mixed",
            [build_AWEvent(8, 1, "foo"), build_AWEvent(1, 1, "bar"), build_AWEvent(4, 1, "dom")],
            [build_Event(1, 1, "bar"), build_Event(4, 1, "dom"), build_Event(8, 1, "foo")],
            {
                'raw events': Metric(3, 10800.0),
                'events to handle': Metric(3, 10800.0),
            }
        ),
        (
            "same_timestamp_first_shorter_same_data",
            [build_AWEvent(1, 1, "foo"), build_AWEvent(1, 2, "foo")],
            [build_Event(1, 2, "foo")],
            {
                'b1 merged the same data events': Metric(cnt=1, duration=3600.0),
                'raw events': Metric(2, 10800.0),
                'events to handle': Metric(1, 7200.0),
            }
        ),
        (
            "same_timestamp_second_shorter_same_data",
            [build_AWEvent(1, 2, "foo"), build_AWEvent(1, 1, "foo")],
            [build_Event(1, 2, "foo")],
            {
                'b1 merged the same data events': Metric(cnt=1, duration=3600.0),
                'raw events': Metric(2, 10800),
                'events to handle': Metric(1, 7200),
            }
        ),
        # overlapping_by_heads_different_data
        # overlapping_by_tails_same_data
        # overlapping_by_tails_different_data
        # overlapping_completely_same_data
        # overlapping_completely_different_data
    ])
    def test_sort_merge_convert_raw_events(self, test_name: str, raw_events: List[AWEvent],
                                           expected_events: List[Event], expected_metrics: Dict[str, Metric]):
        self.maxDiff = None
        metrics = Metrics({})
        # Act
        result : List[Event] = sort_merge_convert_raw_events(raw_events, "b1", datetime.timedelta(0), metrics)
        # Assert
        err_msg = f"'{test_name}' case failed with wrong "
        self.assertEqual(result, expected_events, err_msg + "events")
        self.assertDictEqual(
            {k: v for (k, v) in metrics.metrics.items() if v.cnt > 0},
            {k: v for (k, v) in expected_metrics.items() if v.cnt > 0},
            err_msg
        )
