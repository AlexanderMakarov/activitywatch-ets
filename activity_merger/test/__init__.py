"""Tests for the repository."""
import datetime

from ..domain.input_entities import Event


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
