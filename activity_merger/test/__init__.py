"""Tests for the repository."""
import datetime
from typing import List, Tuple
from ..domain.interval import Interval


def build_datetime(seed: int, day=None) -> datetime.datetime:
    """
    Builds UTC datetime for Jan 2000 with given day and hour. Remained attributes are 0.
    :param seed: Hour and a day if is not specified.
    :param day: Day to set, if not specified then equal to seed.
    :return: Datetime for Jan 2000 with given day and hour.
    """
    return datetime.datetime(2000, 1, day if day else seed, seed, 0, 0).astimezone(datetime.timezone.utc)


def build_timedelta(seed: int, in_hours=False) -> datetime.timedelta:
    """
    Builds timedelta between specified datetimes in days+hours or just in hours.
    :param seed: difference in days+hours or just in hours.
    :param in_hours: Flag to make difference only in hours. By default makes difference in days and hours.
    :return: Timedelta with specified difference.
    """
    return build_datetime(seed + 1, 1 if in_hours else None) - build_datetime(1, 1 if in_hours else None)


def build_intervals_linked_list(data: List[Tuple[int, bool, int]]) -> Interval:
    """
    Builds intervals linked list from the list of tuples. Doesn't check parameters.
    :param data: List of tuples (day of start, flag to return `Interval` from the function, duration).
    :param in_hours: Flag to build intervals in hours. By-default in days.
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
