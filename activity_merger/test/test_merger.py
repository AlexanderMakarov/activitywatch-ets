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
awc = MagicMock()
timedelta_0 = build_timedelta(0)
event_0_length = Event(afk_bucket_id, start_time, timedelta_0, 'event_0_length')
event_1 = Event(afk_bucket_id, start_time, build_timedelta(1, True), 'event_1')
event_1_bucket2 = Event(afk_bucket_id2, start_time, build_timedelta(1, True), 'event_1_bucket2')
inerval_for_1_event = build_intervals([(1, 1, [event_1])])
event_2 = Event(afk_bucket_id, build_datetime(2, day=1), build_timedelta(1, True), 'event_2')
event_2_bucket2 = Event(afk_bucket_id2, build_datetime(2, day=1), build_timedelta(1, True), 'event_2_bucket2')
event_3 = Event(afk_bucket_id, build_datetime(3, day=1), build_timedelta(1, True), 'event_3')
event_3_bucket2 = Event(afk_bucket_id2, build_datetime(3, day=1), build_timedelta(1, True), 'event_3_bucket2')
interval_for_3_consecutive_events = build_intervals([(1, 1, [event_1]), (2, 1, [event_2]), (3, 1, [event_3])])
event_4 = Event(afk_bucket_id, build_datetime(4, day=1), build_timedelta(1, True), 'event_4')
event_4_bucket2 = Event(afk_bucket_id2, build_datetime(4, day=1), build_timedelta(1, True), 'event_4_bucket2')
event_1l2 = Event(afk_bucket_id, build_datetime(1, day=1), build_timedelta(2, True), 'event_1l2')
event_1l2_1 = Event(afk_bucket_id, build_datetime(1, day=1), build_timedelta(1, True), 'event_1l2_1')
event_1l2_2 = Event(afk_bucket_id, build_datetime(2, day=1), build_timedelta(1, True), 'event_1l2_2')
event_1l4 = Event(afk_bucket_id, build_datetime(1, day=1), build_timedelta(4, True), 'event_1l4')
event_2l2 = Event(afk_bucket_id, build_datetime(2, day=1), build_timedelta(2, True), 'event_2l2')


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
    def TODOtest_apply_events_stopwatch(self, test_name: str, events: List[Event], interval: Interval,
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
            self.assertListEqual(actual_interval.get_range(0), expected_interval.get_range(0), err_msg)
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
            "overlapping event same start time but smaller",
            [event_2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2])]),
            {'cnt_split_one_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlaping event same start and ent time",
            [event_1],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1, event_1])]),
            {'cnt_match_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping event same start time but bigger",
            [event_1l2],
            build_intervals([(1, 1, [event_1])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1, event_1l2]), (2, 1, [event_1l2])]),
            {'cnt_split_few_intervals': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping event same end time",
            [event_2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2])]),
            {'cnt_split_one_interval': 1, 'cnt_handled_events': 1},
        ),
        (
            "overlapping event start time inside and end time outside",
            [event_2l2],
            build_intervals([(1, 2, [event_1l2])]),
            timedelta_0,
            build_intervals([(1, 1, [event_1l2]), (2, 1, [event_1l2, event_2l2]), (3, 1, [event_2l2])]),
            {'cnt_split_few_intervals': 1, 'cnt_handled_events': 1},
        ),
    ])
    def TODOtest_apply_events_afk(self, test_name: str, events: List[Event], interval: Interval,
            tolerance: datetime.timedelta,
            expected_interval: Interval, expected_metrics: Dict[str, int]):
        # Act
        actual_interval: Interval
        actual_metrics: Dict[str, int]
        actual_interval, actual_metrics = merger.apply_events(events, interval, tolerance, True, False)
        # Assert
        err_msg = f"'{test_name}' case failed."
        if expected_interval:
            self.assertListEqual(actual_interval.get_range(0), expected_interval.get_range(0), err_msg)
        else:
            self.assertIsNone(actual_interval, err_msg)
        self.assertDictEqual(
            {k: v for (k, v) in actual_metrics.items() if v > 0},
            {k: v for (k, v) in expected_metrics.items() if v > 0},
            err_msg
        )

    @parameterized.expand([
        # (
        #     "No buckets",
        #     [],
        #     {},
        #     timedelta_0,
        #     None,
        #     None,
        #     [],
        #     [("No buckets except AFK and/or Stopwatch found. Stopping here - no more events expected.",)]
        # ),
        # (
        #     "No AFK events",
        #     [[]],
        #     {afk_bucket_id: None},
        #     timedelta_0,
        #     None,
        #     None,
        #     [afk_bucket_id],
        #     [("No buckets except AFK and/or Stopwatch found. Stopping here - no more events expected.",)]
        # ),
        # (
        #     "1 AFK 0-length event",
        #     [[event_0_length]],
        #     {afk_bucket_id: None},
        #     timedelta_0,
        #     "Wrong interval boundaries - start time .+ is after or equal end time .+, prev=None, next=None",
        #     None,
        #     [afk_bucket_id],
        #     [
        #         ("JFYI: Skipped 0-duration AFK event at %s in '%s' bucket.", event_to_str(event_0_length), afk_bucket_id),
        #         ("No AFK events found in %s..%s. Stopping here - no more events expected.", start_time, end_time)
        #     ]
        # ),
        # (
        #     "1 AFK event",
        #     [[event_1], []],
        #     {afk_bucket_id: None},
        #     timedelta_0,
        #     None,
        #     inerval_for_1_event,
        #     [afk_bucket_id],
        #     []
        # ),
        # (
        #     "2 consecutive AFK events same bucket",
        #     [[event_1, event_2], []],
        #     {afk_bucket_id: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1]), (2, 1, [event_2])]),
        #     [afk_bucket_id],
        #     []
        # ),
        # (
        #     "2 similar AFK events different buckets",
        #     [[event_1], [event_1_bucket2]],
        #     {afk_bucket_id: None, afk_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1, event_1_bucket2])]),
        #     [afk_bucket_id, afk_bucket_id2],
        #     []
        # ),
        # (
        #     "2 consecutive AFK events different buckets",
        #     [[event_1], [event_2_bucket2]],
        #     {afk_bucket_id: None, afk_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1]), (2, 1, [event_2_bucket2])]),
        #     [afk_bucket_id, afk_bucket_id2],
        #     []
        # ),
        # (
        #     "2 consecutive AFK events different buckets opposite order",
        #     [[event_2], [event_1_bucket2]],
        #     {afk_bucket_id: None, afk_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2])]),
        #     [afk_bucket_id, afk_bucket_id2],
        #     []
        # ),
        # (
        #     "AFK events 1 and 3 in first bucket, 2 in second",
        #     [[event_1, event_3], [event_2_bucket2]],
        #     {afk_bucket_id: None, afk_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1]), (2, 1, [event_2_bucket2]), (3, 1, [event_3])]),
        #     [afk_bucket_id, afk_bucket_id2],
        #     []
        # ),
        # (
        #     "AFK events 2 in first bucket, 1 and 3 in second",
        #     [[event_2], [event_1_bucket2, event_3_bucket2]],
        #     {afk_bucket_id: None, afk_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2]), (3, 1, [event_3_bucket2])]),
        #     [afk_bucket_id, afk_bucket_id2],
        #     []
        # ),
        # (
        #     "AFK events 2 in first bucket, 1 and 4 in second",
        #     [[event_2], [event_1_bucket2, event_4_bucket2]],
        #     {afk_bucket_id: None, afk_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1_bucket2]), (2, 1, [event_2]), (4, 1, [event_4_bucket2])]),
        #     [afk_bucket_id, afk_bucket_id2],
        #     []
        # ),
        # (
        #     "AFK events overlaping in buckets, 1 overlaps 1l2 plus far after 4",
        #     [[event_1l2], [event_1_bucket2, event_4_bucket2]],
        #     {afk_bucket_id: None, afk_bucket_id2: None},
        #     timedelta_0,
        #     None,
        #     build_intervals([(1, 1, [event_1l2_1, event_1_bucket2]), (2, 1, [event_1l2_2]),
        #                      (4, 1, [event_4_bucket2])]),
        #     [afk_bucket_id, afk_bucket_id2],
        #     []
        # ),
        (
            "AFK events overlaping in buckets, 1l4 overlaps 1 and 3",
            [[event_1l4], [event_1_bucket2, event_3_bucket2]],
            {afk_bucket_id: None, afk_bucket_id2: None},
            timedelta_0,
            None,
            build_intervals([(1, 1, [event_1l4, event_1_bucket2]), (2, 1, [event_1l4]),
                             (3, 1, [event_1l4, event_3_bucket2]), (4, 1, [event_1l4])]),
            [afk_bucket_id, afk_bucket_id2],
            []
        ),
    ])
    @patch.object(LOG, "info", MagicMock())
    @patch.object(merger, "check_and_print_intervals", MagicMock())
    def test_report_from_buckets(self, test_name: str, get_events_lists_results: List[Event],
            buckets: Dict[str, object], tolerance: datetime.timedelta, expected_error_message: str,
            expected_interval: Interval, expected_buckets_called: List[str], expected_logs: List[str]):
        # Arrange
        awc.get_events.reset()
        awc.get_events.side_effect = get_events_lists_results
        # Act
        actual: Interval
        if expected_error_message:
            self.assertRaisesRegex(ValueError, expected_error_message,
                                   merger.report_from_buckets, awc, start_time, end_time, buckets, tolerance)
            # self.assertEquals(cm.exception.message, "sdfasd")
            return
        else:
            actual = merger.report_from_buckets(awc, start_time, end_time, buckets, tolerance)
        # Assert
        err_msg = f"'{test_name}' case failed."
        if expected_interval:
            self.assertListEqual(actual.get_range(0), expected_interval.get_range(0), err_msg)
        else:
            self.assertIsNone(actual, err_msg)
        awc.get_events.assert_has_calls([call(x, start=start_time, end=end_time) for x in expected_buckets_called])
        LOG.info.assert_has_calls([call(*x) for x in expected_logs])


if __name__ == '__main__':
    print("Run 'test.py' from root folder.")
    exit(1)
