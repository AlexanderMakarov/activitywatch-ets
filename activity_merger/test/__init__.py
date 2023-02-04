"""Tests for the repository."""
import datetime
from typing import List, Tuple
from ..domain.interval import Interval


def build_datetime(seed: int) -> datetime.datetime:
    return datetime.datetime(2000, 1, seed, seed, 0, 0).astimezone(datetime.timezone.utc)


def build_timedelta(seed: int) -> datetime.timedelta:
    return build_datetime(seed + 1) - build_datetime(1)


def build_intervals_linked_list(data: List[Tuple[int, bool, int]]) -> Interval:
    """
    Builds intervals linked list from the list of tuples. Doesn't check parameters.
    :param data: List of tuples (day of start, flag to return `Interval` from the function, duration).
    :return: Chosen interval.
    """
    result = None
    previous = None
    for (seed, is_target, duration) in data:
        if not previous:
            previous = Interval(build_datetime(seed), build_datetime(seed + duration))
        else:
            tmp = Interval(build_datetime(seed), build_datetime(seed + duration), previous)
            previous.next = tmp
            previous = tmp
        if is_target:
            assert result is None, f"Wrong parameters - '{seed}' interval is marked as result but is not first."
            result = previous
    return result
