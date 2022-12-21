import datetime
import logging
import argparse
import os
from typing import List
from ..domain.input_entities import Event
from ..config.config import CURRENT_TIMEZONE
import aw_client
import aw_core.models as awmodels


def setup_logging() -> logging.Logger:
    """
    Configures 'logging' package and retuns new logger.
    Sets logging level to environment "LOGLEVEL" value or with "INFO".
    """
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.DEBUG, "DEBU")  # To be 4 chars length as another ones.
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO").upper(),
        format='%(asctime)s.%(msecs)03d %(levelname)-4s: %(message)s',
        datefmt="%H:%M:%S"
    )
    return logging.getLogger()


def event_data_to_str(event: Event) -> str:
    if not event:
        return 'null'
    return str(event.data)


def event_to_str(event: Event) -> str:
    return f"{event.timestamp.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{(event.timestamp + event.duration).astimezone(CURRENT_TIMEZONE):%H:%M:%S}("\
           f"{event_data_to_str(event)})"


def from_start_to_end_to_str(obj) -> str:
    return f"{obj.start_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{obj.end_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"


def seconds_to_int_timedelta(seconds: float) -> str:
    return datetime.timedelta(seconds=int(seconds))


def valid_date(s) -> datetime.datetime:  # https://stackoverflow.com/a/25470943
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").astimezone()
    except ValueError as e:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg) from e


def ensure_datetime(d):  # https://stackoverflow.com/a/29840081/1535127
    """
    Takes a date or a datetime as input, outputs a datetime.
    :param d: Datetime or date.
    :return: Always datetime in current time zone.
    """
    if isinstance(d, datetime.datetime):
        return d
    return datetime.datetime(d.year, d.month, d.day).astimezone(CURRENT_TIMEZONE)


def upload_events(events: List[Event], aw_client_name: str, event_type: str, bucket_id: str, is_replace: bool = False):
    """
    Takes list of `Event`-s, converts them into ActivityWatch events, creates new ActivityWatch client, removes bucket
    if need, creates bucket and uploads events into it.
    :param events: List of events to upload.
    :param aw_client_name: Name of client for ActivityWatch.
    :param event_type: Type of event to set for the bucket. Means nothing.
    :param bucket_id: Name of bucket to put events into or recreate.
    :param is_replace: Flag to force remove bucket first.
    """
    # Convert into ActivityWatch clients.
    aw_events = [awmodels.Event(timestamp=x.timestamp, duration=x.duration, data=x.data) for x in events]
    # Build client, check than bucket is created and insert events.
    client = aw_client.ActivityWatchClient(aw_client_name)
    if is_replace:
        client.delete_bucket(bucket_id, True)
    client.create_bucket(bucket_id, event_type=event_type)  # Will return 304 if bucket exists.
    client.insert_events(bucket_id, aw_events)  # Actually returns None.
