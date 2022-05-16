import unittest
import datetime
from parameterized import parameterized
from activity_merger import Interval
from typing import List, Tuple


def _build_datetime(seed: int) -> datetime.datetime:
    return datetime.datetime(2000, 1, seed, seed, 0, 0)


def build_intervals_linked_list(data: List[Tuple[int, int, bool]]) -> Interval:
    result = None
    previous = None
    for (seed, is_target) in data:
        if not previous:
            previous = Interval(_build_datetime(seed), _build_datetime(seed + 1))
        else:
            tmp = Interval(_build_datetime(seed), _build_datetime(seed + 1), previous)
            previous.next = tmp
            previous = tmp
        if is_target:
            result = previous
    return result


class TestInterval(unittest.TestCase):

    @parameterized.expand([
        (
            "Simple the only interval",
            build_intervals_linked_list([
                (1, True)
            ]),
            1
        ),
        (
            "The same interval",
            build_intervals_linked_list([
                (1, False),
                (5, True),
                (6, False)
            ]),
            5
        ),
        (
            "Exact Interval right before",
            build_intervals_linked_list([
                (5, False),
                (6, True),
                (7, False)
            ]),
            5
        ),
        (
            "Exact Interval right after",
            build_intervals_linked_list([
                (3, False),
                (4, True),
                (5, False)
            ]),
            5
        ),
        (
            "Exact Interval far after",
            build_intervals_linked_list([
                (3, True),
                (4, False),
                (5, False)
            ]),
            5
        ),
        (
            "Exact Interval far before",
            build_intervals_linked_list([
                (5, False),
                (6, False),
                (7, True)
            ]),
            5
        ),
    ])
    def test_find_closest_by_start(self, test_name, interval, expected_start_seed):
        target = _build_datetime(5)
        actual: Interval = interval.find_closest(target, datetime.timedelta(0), False)
        expected = _build_datetime(expected_start_seed)
        self.assertEqual(actual.start_time, expected, f"'{test_name}' case failed.")


if __name__ == '__main__':
    unittest.main()
