import unittest
import datetime
from parameterized import parameterized
from aw_core.models import Event
from . import build_datetime, build_timedelta, build_intervals_linked_list

from ..domain.interval import Interval


class TestInterval(unittest.TestCase):

    @parameterized.expand([
        (
            "Simple the only interval",
            build_intervals_linked_list([
                (1, True, 1)
            ]),
            1
        ),
        (
            "The same interval",
            build_intervals_linked_list([
                (1, False, 1),
                (5, True, 1),
                (6, False, 1)
            ]),
            5
        ),
        (
            "Exact Interval right before",
            build_intervals_linked_list([
                (5, False, 1),
                (6, True, 1),
                (7, False, 1)
            ]),
            5
        ),
        (
            "Exact Interval right after",
            build_intervals_linked_list([
                (3, False, 1),
                (4, True, 1),
                (5, False, 1)
            ]),
            5
        ),
        (
            "Exact Interval far after",
            build_intervals_linked_list([
                (3, True, 1),
                (4, False, 1),
                (5, False, 1),
                (6, False, 1),
            ]),
            5
        ),
        (
            "Exact Interval far before",
            build_intervals_linked_list([
                (4, False, 1),
                (5, False, 1),
                (6, False, 1),
                (7, True, 1),
            ]),
            5
        ),
    ])
    def test_find_closest_by_start(self, test_name, interval, expected_start_seed):
        target = build_datetime(5)
        actual: Interval = interval.find_closest(target, datetime.timedelta(0), False)
        expected = build_datetime(expected_start_seed)
        self.assertEqual(actual.start_time, expected, f"'{test_name}' case failed.")

    @parameterized.expand([
        (
            "Simple the only interval",
            build_intervals_linked_list([
                (1, True, 1)
            ]),
            1
        ),
        (
            "The same interval",
            build_intervals_linked_list([
                (1, False, 1),
                (4, True, 1),
                (6, False, 1),
            ]),
            4
        ),
        (
            "Exact Interval right before",
            build_intervals_linked_list([
                (4, False, 1),
                (6, True, 1),
                (7, False, 1),
            ]),
            4
        ),
        (
            "Exact Interval right after",
            build_intervals_linked_list([
                (1, False, 1),
                (2, True, 1),
                (4, False, 1),
            ]),
            4
        ),
        (
            "Exact Interval far after",
            build_intervals_linked_list([
                (2, True, 1),
                (3, False, 1),
                (4, False, 1),
                (5, False, 1),
            ]),
            4
        ),
        (
            "Exact Interval far before",
            build_intervals_linked_list([
                (3, False, 1),
                (4, False, 1),
                (6, False, 1),
                (7, True, 1),
            ]),
            4
        ),
    ])
    def test_find_closest_by_end(self, test_name, interval: Interval, expected_start_seed):
        target = build_datetime(5)
        actual: Interval = interval.find_closest(target, datetime.timedelta(0), True)
        expected = build_datetime(expected_start_seed)
        self.assertEqual(actual.start_time, expected, f"'{test_name}' case failed.")

    @parameterized.expand([
        (
            "Event at middle",
            build_intervals_linked_list([
                (3, True, 5),
            ]),
            Event(1, build_datetime(5), build_timedelta(1)),
            build_intervals_linked_list([
                (3, True, 2),
                (5, False, 1),
                (6, False, 2),
            ]),
        ),
        (
            "Event start equal interval start",
            build_intervals_linked_list([
                (5, True, 5),
            ]),
            Event(1, build_datetime(5), build_timedelta(1)),
            build_intervals_linked_list([
                (5, True, 1),
                (6, False, 4),
            ]),
        ),
        (
            "Event end equal interval end",
            build_intervals_linked_list([
                (4, True, 2),
            ]),
            Event(1, build_datetime(5), build_timedelta(1)),
            build_intervals_linked_list([
                (4, True, 1),
                (5, False, 1),
            ]),
        ),
    ])
    def test_separate_new_at_middle(self, test_name: str, interval: Interval, event: Event,
            expected_interval_offset_2_num_4: Interval):
        actual: Interval = interval.separate_new_at_middle(event, datetime.timedelta(0))
        self.assertListEqual(actual.get_range(-2, 4), expected_interval_offset_2_num_4.get_range(-2, 4),
                f"'{test_name}' case failed.")


if __name__ == '__main__':
    print("Run 'test.py' from root folder.")
    exit(1)
