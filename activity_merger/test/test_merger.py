import unittest
import datetime
import logging
from unittest.mock import MagicMock, Mock, patch, call
from parameterized import parameterized
from typing import List, Dict, Tuple

from . import build_datetime, build_timedelta

from ..helpers.helpers import event_to_str
from ..domain.input_entities import Event
from ..domain.interval import Interval
from ..domain import merger
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


LOG = logging.getLogger("activity_merger.config.config")
start_time = build_datetime(1, day=1)
end_time = build_datetime(2, day=1)
afk_bucket_id = "aw-watcher-afk-foo"
afk_bucket_id2 = "aw-watcher-afk-foo2"
some_bucket_id = "some_bucket_id"
some_bucket_id2 = "some_bucket_id2"
awc = MagicMock()
timedelta_0 = build_timedelta(0)
timedelta_1 = build_timedelta(1, True)
timedelta_2 = build_timedelta(2, True)
timedelta_3 = build_timedelta(3, True)
timedelta_4 = build_timedelta(4, True)
timedelta_6 = build_timedelta(6, True)
event_1l0 = Event(afk_bucket_id, start_time, timedelta_0, 'event_1l0')
event_1 = Event(afk_bucket_id, start_time, timedelta_1, 'event_1')
event_1_bucket2 = Event(afk_bucket_id2, start_time, timedelta_1, 'event_1_bucket2')
inerval_for_1_event = build_intervals([(1, 1, [event_1])])
event_2 = Event(afk_bucket_id, build_datetime(2, day=1), timedelta_1, 'event_2')
event_2l0 = Event(afk_bucket_id, build_datetime(2, day=1), timedelta_0, 'event_2l0')
event_2_bucket2 = Event(afk_bucket_id2, build_datetime(2, day=1), timedelta_1, 'event_2_bucket2')
event_3 = Event(afk_bucket_id, build_datetime(3, day=1), timedelta_1, 'event_3')
event_3_bucket2 = Event(afk_bucket_id2, build_datetime(3, day=1), timedelta_1, 'event_3_bucket2')
interval_for_3_consecutive_events = build_intervals([(1, 1, [event_1]), (2, 1, [event_2]), (3, 1, [event_3])])
event_4 = Event(afk_bucket_id, build_datetime(4, day=1), timedelta_1, 'event_4')
event_4_bucket2 = Event(afk_bucket_id2, build_datetime(4, day=1), timedelta_1, 'event_4_bucket2')
event_1l2 = Event(afk_bucket_id, start_time, timedelta_2, 'event_1l2')
event_2l2_data_from_1l2 = Event(afk_bucket_id, build_datetime(2, day=1), timedelta_2, event_1l2.data)
event_1l3_data_from_1l2 = Event(afk_bucket_id, start_time, timedelta_3, event_1l2.data)
event_1l4 = Event(afk_bucket_id, start_time, timedelta_4, 'event_1l4')
event_2l2 = Event(afk_bucket_id, build_datetime(2, day=1), timedelta_2, 'event_2l2')

sevent_1 = Event(some_bucket_id, start_time, timedelta_1, 'sevent_1')
sevent_2 = Event(some_bucket_id, build_datetime(2, day=1), timedelta_1, 'sevent_2')
sevent_1l2 = Event(some_bucket_id, start_time, timedelta_2, 'sevent_1l2')
sevent_1l3 = Event(some_bucket_id, start_time, timedelta_3, 'sevent_1l3')
sevent_1l6 = Event(some_bucket_id, start_time, timedelta_6, 'sevent_1l6')
sevent_2l2 = Event(some_bucket_id, build_datetime(2, day=1), timedelta_2, 'sevent_2l2')
sevent_3l2 = Event(some_bucket_id, build_datetime(3, day=1), timedelta_2, 'sevent_3l2')

sevent_3l2_bucket2 = Event(some_bucket_id2, build_datetime(3, day=1), timedelta_2, 'sevent_3l2_bucket2')

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
                actual_interval, actual_metrics = merger.apply_events(events, interval, tolerance, True, True)
            return
        else:
            actual_interval, actual_metrics = merger.apply_events(events, interval, tolerance, True, True)
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
        actual_interval, actual_metrics = merger.apply_events(events, interval, tolerance, True, False)
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
            {afk_bucket_id: None},
            timedelta_0,
            None,
            None,
            [afk_bucket_id],
            [("No buckets except AFK and/or Stopwatch found. Stopping here - no more events expected.",)]
        ),
        (
            "1 AFK 0-length event",
            [[event_1l0]],
            {afk_bucket_id: None},
            timedelta_0,
            None,
            None,
            [afk_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 0, ''),
                ("No AFK events found in %s..%s. Stopping here - no more events expected.", start_time, end_time)
            ]
        ),
        (
            "1 AFK event",
            [[event_1], []],
            {afk_bucket_id: None},
            timedelta_0,
            None,
            inerval_for_1_event,
            [afk_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1')
            ]
        ),
        (
            "2 consecutive AFK events same bucket",
            [[event_1, event_2], []],
            {afk_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2])]),
            [afk_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2')
            ]
        ),
        (
            "2 similar AFK events same bucket",
            [[event_1, event_1]],
            {afk_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1])]),
            [afk_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1,
                 'cnt_new_interval: 1\n  cnt_handled_events: 1')
            ]
        ),
        (
            "2 similar AFK events different buckets",
            [[event_1], [event_1_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1, event_1_bucket2])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 1,
                 'cnt_new_interval: 1\n  cnt_handled_events: 2\n  cnt_match_interval: 1')
            ]
        ),
        (
            "2 consecutive AFK events different buckets",
            [[event_1], [event_2_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2_bucket2])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2')
            ]
        ),
        (
            "2 consecutive AFK events different buckets opposite order",
            [[event_2], [event_1_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2')
            ]
        ),
        (
            "4 consecutive AFK events one 0-length different buckets",
            [[event_1l2, event_2l0, event_2l2_data_from_1l2], [event_2_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l3_data_from_1l2]), (2, 1, [event_1l3_data_from_1l2, event_2_bucket2]),
                             (3, 1, [event_1l3_data_from_1l2])]),
            [afk_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 3,
                 'cnt_new_interval: 1\n  cnt_handled_events: 2\n  cnt_inside_interval: 1')
            ]
        ),
        (
            "AFK events 1 and 3 in first bucket, 2 in second",
            [[event_1, event_3], [event_2_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1]), (2, 1, [event_2_bucket2]), (3, 1, [event_3])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 3, 'cnt_new_interval: 3\n  cnt_handled_events: 3')
            ]
        ),
        (
            "AFK events 2 in first bucket, 1 and 3 in second",
            [[event_2], [event_1_bucket2, event_3_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2]), (3, 1, [event_3_bucket2])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 3, 'cnt_new_interval: 3\n  cnt_handled_events: 3')
            ]
        ),
        (
            "AFK events 2 in first bucket, 1 and 4 in second",
            [[event_2], [event_1_bucket2, event_4_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2]), (4, 1, [event_4_bucket2])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 3, 'cnt_new_interval: 3\n  cnt_handled_events: 3')
            ]
        ),
        (
            "2 overlaping AFK events same bucket and data",
            [[event_1l2, event_2l2_data_from_1l2]],
            {afk_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 3, [event_1l3_data_from_1l2])]),
            [afk_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1,
                 'cnt_new_interval: 1\n  cnt_handled_events: 1')
            ]
        ),
        (
            "AFK events overlaping in buckets, 1 overlaps 1l2 plus far after 4",
            [[event_1l2], [event_1_bucket2, event_4_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l2, event_1_bucket2]), (2, 1, [event_1l2]),
                             (4, 1, [event_4_bucket2])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 3,
                 'cnt_new_interval: 2\n  cnt_handled_events: 3\n  cnt_split_one_interval: 1')
            ]
        ),
        (
            "AFK events overlaping in buckets, 1l4 overlaps 1 and 3",
            [[event_1l4], [event_1_bucket2, event_3_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l4, event_1_bucket2]), (2, 1, [event_1l4]),
                             (3, 1, [event_1l4, event_3_bucket2]), (4, 1, [event_1l4])]),
            [afk_bucket_id, afk_bucket_id2],
            [
                ('In result got %d intervals. Details:\n  %s', 4,
                 'cnt_new_interval: 1\n  cnt_handled_events: 3\n  cnt_inside_interval: 1\n  cnt_split_one_interval: 1')
            ]
        ),
        (
            "Some empty bucket",
            [[event_1], []],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("'%s' bucket doesn't have events in %s..%s.", some_bucket_id, start_time, build_datetime(2, day=1)),
            ]
        ),
        (
            "Some event matches AFK",
            [[event_1], [sevent_1]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1, sevent_1])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 1),
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_handled_events: 1\n  cnt_match_interval: 1'),
            ]
        ),
        (
            "Some event before AFK",
            [[event_2], [sevent_1]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 1),
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_skipped_before_afk: 1'),
            ]
        ),
        (
            "Some event after AFK",
            [[event_1], [sevent_2]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 1),
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_skipped_after_afk: 1'),
            ]
        ),
        (
            "Some event overlaps AFK's start",
            [[event_2l2], [sevent_1l2]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2l2, sevent_1l2]), (3, 1, [event_2l2])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 1),
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_handled_events: 1\n  cnt_split_one_interval: 1'),
            ]
        ),
        (
            "Some event overlaps AFK's end",
            [[event_1l2], [sevent_2l2]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, sevent_2l2])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 1),
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
            ]
        ),
        (
            "Some event overlaps AFK's both sides",
            [[event_2], [sevent_1l3]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2, sevent_1l3])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_new_interval: 1\n  cnt_handled_events: 1'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 1),
                ('In result got %d intervals. Details:\n  %s', 1, 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
            ]
        ),
        (
            "Some event overlaps 2 AFK's both sides",
            [[event_2, event_4], [sevent_1l6]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(2, 1, [event_2, sevent_1l6]), (4, 1, [event_4, sevent_1l6])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 1),
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
            ]
        ),
        # (  # TODO add support of case below
        #     "Some events starts right after 1 AFK's and ovelaps with next",
        #     [[event_1, event_3], [sevent_2l2]],
        #     {afk_bucket_id: None, some_bucket_id: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1, sevent_1l2]), (3, 1, [event_3, sevent_2l2])]),
        #     [afk_bucket_id, some_bucket_id],
        #     [
        #         ('In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
        #         ("Applying '%s' bucket %s events.", some_bucket_id, 1),
        #         ('In result got %d intervals. Details:\n  %s', 2, 'cnt_handled_events: 1\n  cnt_split_one_interval: 1'),
        #     ]
        # ),
        (
            "2 some events inside 2 AFK's with other borders",
            [[event_1, event_2l2], [sevent_1l2, sevent_3l2]],
            {afk_bucket_id: None, some_bucket_id: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1, sevent_1l2]), (2, 1, [event_2l2, sevent_1l2]),
                             (3, 1, [event_2l2, sevent_3l2])]),
            [afk_bucket_id, some_bucket_id],
            [
                ('In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
                ("Applying '%s' bucket %s events.", some_bucket_id, 2),
                ('In result got %d intervals. Details:\n  %s', 3, 'cnt_handled_events: 2\n  cnt_split_few_intervals: 2'),
            ]
        ),
        # (  # TODO add support of case below - event starts
        #     "Few some events overlaps 2 AFK's both sides",
        #     [[event_2, event_4], [sevent_1l6], [sevent_3l2_bucket2]],
        #     {afk_bucket_id: None, some_bucket_id: None, some_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(2, 1, [event_2, sevent_1l6]), (4, 1, [event_4, sevent_1l6, sevent_3l2_bucket2])]),
        #     [afk_bucket_id, some_bucket_id, some_bucket_id2],
        #     [
        #         ('In result got %d intervals. Details:\n  %s', 2, 'cnt_new_interval: 2\n  cnt_handled_events: 2'),
        #         ("Applying '%s' bucket %s events.", some_bucket_id, 1),
        #         ('In result got %d intervals. Details:\n  %s', 2, 'cnt_handled_events: 1\n  cnt_split_few_intervals: 1'),
        #     ]
        # ),
    ])
    @patch.object(LOG, "info", MagicMock())
    @patch.object(merger, "check_and_print_intervals", MagicMock())
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
                                   merger.report_from_buckets, awc, start_time, end_time, buckets, tolerance)
            return
        else:
            actual = merger.report_from_buckets(awc, start_time, end_time, buckets, tolerance)
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


if __name__ == '__main__':
    print("Run 'test.py' from root folder.")
    exit(1)
