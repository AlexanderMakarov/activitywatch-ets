import unittest
import datetime
from typing import List, Tuple
from parameterized import parameterized
from aw_core.models import Event

from . import build_datetime, build_timedelta
from ..domain.interval import Interval


def build_intervals_linked_list(data: List[Tuple[int, bool, int, List[Event]]]) -> Interval:
    """
    Builds intervals linked list from the list of tuples. Doesn't check parameters.
    :param data: List of tuples (day of start, flag to return `Interval` from the function, duration, list of events).
    :param in_hours: Flag to build intervals in hours. By-default in days.
    :return: Chosen interval.
    """
    result = None
    previous = None
    for (seed, is_target, duration, events) in data:
        if not previous:
            previous = Interval(build_datetime(seed), build_datetime(seed + duration))
            previous.events = events
        else:
            tmp = Interval(build_datetime(seed), build_datetime(seed + duration), previous)
            tmp.events = events
            previous.next = tmp
            previous = tmp
        if is_target:
            assert result is None, f"Wrong parameters - '{seed}' interval is marked as result but is not first."
            result = previous
    return result


event_3l5 = Event(1, build_datetime(2), build_timedelta(5), "event_3l5")
event_3l2 = Event(2, build_datetime(3), build_timedelta(2), "event_3l2")
event_5l1 = Event(3, build_datetime(5), build_timedelta(1), "event_5l1")
event_5l5 = Event(4, build_datetime(5), build_timedelta(5), "event_5l5")


class TestInterval(unittest.TestCase):

    @parameterized.expand([
        (
            "Simple the only interval",
            build_intervals_linked_list([
                (1, True, 1, [event_3l5])
            ]),
            1
        ),
        (
            "The same interval",
            build_intervals_linked_list([
                (1, False, 1, [event_3l5]),
                (5, True, 1, [event_3l5]),
                (6, False, 1, [event_3l5])
            ]),
            5
        ),
        (
            "Exact Interval right before",
            build_intervals_linked_list([
                (5, False, 1, [event_3l5]),
                (6, True, 1, [event_3l5]),
                (7, False, 1, [event_3l5])
            ]),
            5
        ),
        (
            "Exact Interval right after",
            build_intervals_linked_list([
                (3, False, 1, [event_3l5]),
                (4, True, 1, [event_3l5]),
                (5, False, 1, [event_3l5])
            ]),
            5
        ),
        (
            "Exact Interval far after",
            build_intervals_linked_list([
                (3, True, 1, [event_3l5]),
                (4, False, 1, [event_3l5]),
                (5, False, 1, [event_3l5]),
                (6, False, 1, [event_3l5]),
            ]),
            5
        ),
        (
            "Exact Interval far before",
            build_intervals_linked_list([
                (4, False, 1, [event_3l5]),
                (5, False, 1, [event_3l5]),
                (6, False, 1, [event_3l5]),
                (7, True, 1, [event_3l5]),
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
                (1, True, 1, [event_3l5])
            ]),
            1
        ),
        (
            "The same interval",
            build_intervals_linked_list([
                (1, False, 1, [event_3l5]),
                (4, True, 1, [event_3l5]),
                (6, False, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Exact Interval right before",
            build_intervals_linked_list([
                (4, False, 1, [event_3l5]),
                (6, True, 1, [event_3l5]),
                (7, False, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Exact Interval right after",
            build_intervals_linked_list([
                (1, False, 1, [event_3l5]),
                (3, True, 1, [event_3l5]),
                (4, False, 1, [event_3l5]),
                (5, False, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Exact Interval far after",
            build_intervals_linked_list([
                (2, True, 1, [event_3l5]),
                (3, False, 1, [event_3l5]),
                (4, False, 1, [event_3l5]),
                (5, False, 1, [event_3l5]),
            ]),
            4
        ),
        (
            "Exact Interval far before",
            build_intervals_linked_list([
                (3, False, 1, [event_3l5]),
                (4, False, 1, [event_3l5]),
                (6, False, 1, [event_3l5]),
                (7, True, 1, [event_3l5]),
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
