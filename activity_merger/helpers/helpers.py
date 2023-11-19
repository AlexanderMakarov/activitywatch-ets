import argparse
import datetime
import logging
import os

from ..config.config import CURRENT_TIMEZONE


def setup_logging() -> logging.Logger:
    """
    Configures 'logging' package and retuns new logger.
    Sets logging level to environment "LOGLEVEL" value or with "INFO".
    """
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.DEBUG, "DEBU")  # To be 4 chars length as another ones.
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s.%(msecs)03d %(levelname)-4s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger()


def datetime_to_time_str(date: datetime.datetime) -> str:
    date = date if date.tzinfo == CURRENT_TIMEZONE else date.astimezone(CURRENT_TIMEZONE)
    return f"{date:%H:%M:%S}"


def from_start_to_end_to_str(start: datetime.datetime, end: datetime.datetime) -> str:
    return f"{datetime_to_time_str(start)}..{datetime_to_time_str(end)}"


def seconds_to_timedelta(seconds: float) -> datetime.timedelta:
    return datetime.timedelta(seconds=int(seconds))


def valid_date(date_str) -> datetime.datetime:  # https://stackoverflow.com/a/25470943
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").astimezone()
    except ValueError as err:
        msg = "not a valid date: {0!r}".format(date_str)
        raise argparse.ArgumentTypeError(msg) from err


def ensure_datetime(obj):  # https://stackoverflow.com/a/29840081/1535127
    """
    Takes a date or a datetime as input, outputs a datetime.
    :param d: Datetime or date.
    :return: Always datetime in current time zone.
    """
    if isinstance(obj, datetime.datetime):
        return obj
    return datetime.datetime(obj.year, obj.month, obj.day).astimezone(CURRENT_TIMEZONE)
