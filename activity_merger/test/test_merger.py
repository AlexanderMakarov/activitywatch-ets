import unittest
import datetime
import logging
from unittest.mock import MagicMock, patch
from parameterized import parameterized
from aw_core.models import Event
from typing import List, Dict
from . import build_datetime, build_timedelta, build_intervals_linked_list

from ..domain.interval import Interval
from ..domain.merger import report_from_buckets
from ..config.config import LOG


LOG = logging.getLogger(__name__)
start_time = build_datetime(1)
end_time = build_datetime(2)


class TestMerger(unittest.TestCase):
    @parameterized.expand([
        (
            "Simple one event",
            [Event('', start_time, build_timedelta(0), None)],
            {"b1": None},
            0.0,
            build_intervals_linked_list([
                (1, True, 1)
            ]),
            ["b1"],
            [""]
        ),
    ])
    @patch.object(LOG, "warning", MagicMock())
    def test_report_from_buckets(self, test_name: str, get_events_results: List[Event], buckets: Dict[str, object],
                                 tolerance: datetime.timedelta,
                                 expected_interval: Interval, expected_buckets_called: List[str], expected_logs: List[str]):
        # Arrange
        awc: MagicMock = MagicMock(
            **{'get_events.return_value': get_events_results})
        # Act
        actual: Interval = report_from_buckets(awc, start_time, end_time, buckets, tolerance)
        # Assert
        self.assertListEqual(actual.get_range(), expected_interval.get_range(), f"'{test_name}' case failed.")
        awc.get_events.assert_has_calls(
            [(x, start_time, end_time) for x in expected_buckets_called])
        LOG.info.assert_has_calls(expected_logs)

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
