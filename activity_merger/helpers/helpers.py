import datetime
import logging
import argparse
from ..domain.input_entities import Event
from ..config.config import CURRENT_TIMEZONE


def setup_logging():
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.DEBUG, "DEBU")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)-4s: %(message)s',
        datefmt="%H:%M:%S"
    )
    return logging.getLogger()


def event_data_to_str(event: Event):
    if not event:
        return 'null'
    return str(event.data)


def event_to_str(event: Event):
    return f"{event.timestamp.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{(event.timestamp + event.duration).astimezone(CURRENT_TIMEZONE):%H:%M:%S}("\
           f"{event_data_to_str(event)})"


def from_start_to_end_to_str(obj) -> str:
    return f"{obj.start_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{obj.end_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"


def seconds_to_int_timedelta(seconds: float) -> str:
    return datetime.timedelta(seconds=int(seconds))


def valid_date(s):  # https://stackoverflow.com/a/25470943
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)
