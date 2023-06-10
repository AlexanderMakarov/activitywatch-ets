import unittest
import datetime
from parameterized import parameterized
from aw_core.models import Event

from . import build_datetime, build_timedelta, build_intervals_linked_list
from ..domain.interval import Interval


event_3l5 = Event(1, build_datetime(2), build_timedelta(5), "event_3l5")
event_3l2 = Event(2, build_datetime(3), build_timedelta(2), "event_3l2")
event_5l1 = Event(3, build_datetime(5), build_timedelta(1), "event_5l1")
event_5l5 = Event(4, build_datetime(5), build_timedelta(5), "event_5l5")


class TestInterval(unittest.TestCase):

    @parameterized.expand([
        # Search for date 5. (name, interval, expected start date of returned interval)
        (
            "The only interval before",
            build_intervals_linked_list([
                (1, True, 1, [event_3l5])
            ]),
            1
        ),
        (
            "The only interval after",
            build_intervals_linked_list([
                (6, True, 1, [event_3l5])
            ]),
            6
        ),
        (
            "Start of current interval",
            build_intervals_linked_list([
                (1, False, 1, [event_3l5]),
                (5, True, 1, [event_3l5]),
                (6, False, 1, [event_3l5])
            ]),
            5
        ),
        (
            "End of current interval",
            build_intervals_linked_list([
                (1, False, 1, [event_3l5]),
                (4, True, 1, [event_3l5]),
                (5, False, 1, [event_3l5])
            ]),
            5  # Should prefer interval with 'start'.
        ),
        (
            "End of interval right before",
            build_intervals_linked_list([
                (3, False, 1, [event_3l5]),
                (4, False, 1, [event_3l5]),
                (6, True, 1, [event_3l5]),
                (7, False, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Start of interval far before",
            build_intervals_linked_list([
                (4, False, 1, [event_3l5]),
                (5, False, 1, [event_3l5]),
                (6, False, 1, [event_3l5]),
                (7, False, 1, [event_3l5]),
                (8, True, 1, [event_3l5]),
            ]),
            5
        ),
        (
            "End of interval right after",
            build_intervals_linked_list([
                (3, True, 1, [event_3l5]),
                (4, False, 1, [event_3l5]),
                (5, False, 1, [event_3l5])
            ]),
            5  # Should prefer interval with 'start'.
        ),
        (
            "Start of interval far after",
            build_intervals_linked_list([
                (3, True, 1, [event_3l5]),
                (5, False, 1, [event_3l5]),
                (6, False, 1, [event_3l5]),
            ]),
            5
        ),
        (
            "Middle of current interval",
            build_intervals_linked_list([
                (3, False, 1, [event_3l5]),
                (4, True, 2, [event_3l5]),
                (6, False, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Middle of interval right before",
            build_intervals_linked_list([
                (3, False, 1, [event_3l5]),
                (4, False, 2, [event_3l5]),
                (6, True, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Middle of interval far before",
            build_intervals_linked_list([
                (2, False, 1, [event_3l5]),
                (4, False, 2, [event_3l5]),
                (6, False, 1, [event_3l5]),
                (7, True, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Middle of interval fight after",
            build_intervals_linked_list([
                (3, True, 1, [event_3l5]),
                (4, False, 2, [event_3l5]),
                (6, False, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Middle of interval far after",
            build_intervals_linked_list([
                (2, True, 1, [event_3l5]),
                (3, False, 1, [event_3l5]),
                (4, False, 2, [event_3l5]),
                (6, False, 1, [event_3l5]),
            ]),
            4
        ),
    ])
    def test_find_closest(self, test_name, interval, expected_start_seed):
        # Arrange
        target = build_datetime(5)
        expected = build_datetime(expected_start_seed)
        # Act
        actual: Interval = interval.find_closest(target, datetime.timedelta(0))
        # Assert
        self.assertEqual(actual.start_time, expected, f"'{test_name}' case failed.")

    @parameterized.expand([
        (
            "Event at middle",
            build_intervals_linked_list([
                (3, True, 5, [event_3l5]),
            ]),
            event_5l1,
            build_intervals_linked_list([
                (3, True, 2, [event_3l5]),
                (5, False, 1, [event_3l5, event_5l1]),
                (6, False, 2, [event_3l5]),
            ]),
        ),
        (
            "Event start equal interval start",
            build_intervals_linked_list([
                (5, True, 5, [event_5l5]),
            ]),
            event_5l1,
            build_intervals_linked_list([
                (5, True, 1, [event_5l5, event_5l1]),
                (6, False, 4, [event_5l5]),
            ]),
        ),
        (
            "Event end equal interval end",
            build_intervals_linked_list([
                (4, True, 2, [event_5l5]),
            ]),
            event_5l1,
            build_intervals_linked_list([
                (4, True, 1, [event_5l5]),
                (5, False, 1, [event_5l5, event_5l1]),
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
