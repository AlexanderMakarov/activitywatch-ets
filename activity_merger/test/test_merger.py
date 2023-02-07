import unittest
import datetime
import logging
from unittest.mock import MagicMock, Mock, patch, call
from parameterized import parameterized
from aw_core.models import Event
from typing import List, Dict, Tuple

from . import build_datetime, build_timedelta

from ..helpers.helpers import event_to_str
from ..domain.interval import Interval
from ..domain import merger
from ..config.config import LOG


def build_intervals(data: List[Tuple[int, int, List[Event]]]) -> Interval:
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
awc = MagicMock()
event_0_length = Event('', start_time, build_timedelta(0), None)
event_1 = Event('', start_time, build_timedelta(1, True), None)
inerval_for_1_event = build_intervals([(1, 1, [event_1])])
event_2 = Event('', build_datetime(2, day=1), build_timedelta(1, True), None)
inerval_for_2_consequtive_events = build_intervals([(1, 1, [event_1]), (2, 1, [event_2])])

class TestMerger(unittest.TestCase):
    @parameterized.expand([
        # (
        #     "No buckets",
        #     [],
        #     {},
        #     0.0,
        #     None,
        #     [],
        #     [("No AFK buckets found. Stopping here - no more events expected.",)]
        # ),
        # (
        #     "No AFK events",
        #     [[]],
        #     {afk_bucket_id: None},
        #     0.0,
        #     None,
        #     [afk_bucket_id],
        #     [("No AFK buckets found. Stopping here - no more events expected.",)]
        # ),
        # (
        #     "1 AFK 0-length event",
        #     [[event_0_length]],
        #     {afk_bucket_id: None},
        #     0.0,
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
        #     0.0,
        #     inerval_for_1_event,
        #     [afk_bucket_id],
        #     []
        # ),
        (
            "2 AFK events same bucket",
            [[event_1, event_2], []],
            {afk_bucket_id: None},
            0.0,
            inerval_for_2_consequtive_events,
            [afk_bucket_id],
            []
        ),
    ])
    @patch.object(LOG, "info", MagicMock())
    @patch.object(merger, "check_and_print_intervals", MagicMock())
    def test_report_from_buckets(self, test_name: str, get_events_lists_results: List[Event], buckets: Dict[str, object],
                                 tolerance: datetime.timedelta,
                                 expected_interval: Interval, expected_buckets_called: List[str], expected_logs: List[str]):
        # Arrange
        awc.get_events.reset()
        awc.get_events.side_effect = get_events_lists_results
        # Act
        actual: Interval = merger.report_from_buckets(awc, start_time, end_time, buckets, tolerance)
        # Assert
        err_msg = f"'{test_name}' case failed."
        if expected_interval:
            self.assertListEqual(actual.get_range(), expected_interval.get_range(), err_msg)
        else:
            self.assertIsNone(actual, err_msg)
        awc.get_events.assert_has_calls([call(x, start=start_time, end=end_time) for x in expected_buckets_called])
        LOG.info.assert_has_calls([call(*x) for x in expected_logs])

    # @parameterized.expand([
    #     (
    #         "Simple the only interval",
    #         build_intervals_linked_list([
    #             (1, True, 1)
    #         ]),
    #         1
    #     ),
    #     (
    #         "The same interval",
    #         build_intervals_linked_list([
    #             (1, False, 1),
    #             (4, True, 1),
    #             (6, False, 1),
    #         ]),
    #         4
    #     ),
    #     (
    #         "Exact Interval right before",
    #         build_intervals_linked_list([
    #             (4, False, 1),
    #             (6, True, 1),
    #             (7, False, 1),
    #         ]),
    #         4
    #     ),
    #     (
    #         "Exact Interval right after",
    #         build_intervals_linked_list([
    #             (1, False, 1),
    #             (2, True, 1),
    #             (4, False, 1),
    #         ]),
    #         4
    #     ),
    #     (
    #         "Exact Interval far after",
    #         build_intervals_linked_list([
    #             (2, True, 1),
    #             (3, False, 1),
    #             (4, False, 1),
    #             (5, False, 1),
    #         ]),
    #         4
    #     ),
    #     (
    #         "Exact Interval far before",
    #         build_intervals_linked_list([
    #             (3, False, 1),
    #             (4, False, 1),
    #             (6, False, 1),
    #             (7, True, 1),
    #         ]),
    #         4
    #     ),
    # ])
    # def test_find_closest_by_end(self, test_name, interval: Interval, expected_start_seed):
    #     target = _build_datetime(5)
    #     actual: Interval = interval.find_closest(target, datetime.timedelta(0), True)
    #     expected = _build_datetime(expected_start_seed)
    #     self.assertEqual(actual.start_time, expected, f"'{test_name}' case failed.")

    # @parameterized.expand([
    #     (
    #         "Event at middle",
    #         build_intervals_linked_list([
    #             (3, True, 5),
    #         ]),
    #         Event(1, _build_datetime(5), _build_timedelta(1)),
    #         build_intervals_linked_list([
    #             (3, True, 2),
    #             (5, False, 1),
    #             (6, False, 2),
    #         ]),
    #     ),
    #     (
    #         "Event start equal interval start",
    #         build_intervals_linked_list([
    #             (5, True, 5),
    #         ]),
    #         Event(1, _build_datetime(5), _build_timedelta(1)),
    #         build_intervals_linked_list([
    #             (5, True, 1),
    #             (6, False, 4),
    #         ]),
    #     ),
    #     (
    #         "Event end equal interval end",
    #         build_intervals_linked_list([
    #             (4, True, 2),
    #         ]),
    #         Event(1, _build_datetime(5), _build_timedelta(1)),
    #         build_intervals_linked_list([
    #             (4, True, 1),
    #             (5, False, 1),
    #         ]),
    #     ),
    # ])
    # def test_separate_new_at_middle(self, test_name: str, interval: Interval, event: Event,
    #         expected_interval_offset_2_num_4: Interval):
    #     actual: Interval = interval.separate_new_at_middle(event, datetime.timedelta(0))
    #     self.assertListEqual(actual.get_range(-2, 4), expected_interval_offset_2_num_4.get_range(-2, 4),
    #             f"'{test_name}' case failed.")


if __name__ == '__main__':
    print("Run 'test.py' from root folder.")
    exit(1)
