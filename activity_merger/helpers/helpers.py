import datetime
import logging
import argparse
import os
from typing import List
import aw_client
import aw_core.models as awmodels
from ..domain.input_entities import Event
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


def event_data_to_str(event: Event) -> str:
    if not event:
        return "null"
    return str(event.data)


def datetime_to_time_str(date: datetime.datetime) -> str:
    date = date if date.tzinfo == CURRENT_TIMEZONE else date.astimezone(CURRENT_TIMEZONE)
    return f"{date:%H:%M:%S}"


def event_to_str(event: Event) -> str:
    return (
        f"{datetime_to_time_str(event.timestamp)}"
        f"..{datetime_to_time_str(event.timestamp + event.duration)}("
        f"{event_data_to_str(event)})"
    )


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


def delete_buckets(bucket_ids: List[str], client: aw_client.ActivityWatchClient) -> str:
    """
    Forcely removes ActivityWatch buckets without producing extra logs.
    :param bucket_ids: List of bucket ID-s to remove.
    :param client: ActivityWatch client to use.
    :return: Human-friendly string with applied actions.
    """
    result = []
    try:
        buckets = client.get_buckets()
        for bucket_id in bucket_ids:
            if bucket_id in buckets.keys():
                client.delete_bucket(bucket_id, True)
                result.append("Deleted '" + bucket_id + "' bucket. ")
    except Exception as err:
        return "Wasn't able connect to ActivityWatch client because: " + str(err)
    return "".join(result)


def upload_events(
    events: List[Event],
    event_type: str,
    bucket_id: str,
    is_replace: bool = False,
    aw_client_name: str = "upload_events",
    client: aw_client.ActivityWatchClient = None,
) -> str:
    """
    Takes list of `Event`-s, converts them into ActivityWatch events, creates new ActivityWatch client, removes bucket
    if need, creates bucket and uploads events into it.
    :param events: List of events to upload.
    :param event_type: Type of event to set for the bucket. Means nothing.
    :param bucket_id: Name of bucket to put events into or recreate.
    :param is_replace: Flag to force remove bucket first.
    :param aw_client_name: Name of client for ActivityWatch. Not needed if client is specified externally.
    :param client: ActivityWatch client to use. By-default new client will be created.
    :return: String with representation of implemented actions.
    """
    # Convert into ActivityWatch clients.
    aw_events = [awmodels.Event(timestamp=x.timestamp, duration=x.duration, data=x.data) for x in events]
    result = ""
    # Build client, check than bucket is created and insert events.
    if not client:
        client = aw_client.ActivityWatchClient(aw_client_name)
    if is_replace:
        result += delete_buckets(bucket_id, client)
    client.create_bucket(bucket_id, event_type=event_type)  # Will return 304 if bucket exists.
    client.insert_events(bucket_id, aw_events)  # Actually returns None.
    result += f"Uploaded {len(aw_events)} events into local ActivityWatch '{bucket_id}' bucket."
    return result
