from typing import List

import aw_client
import aw_core.models as awmodels

from activity_merger.helpers.helpers import datetime_to_time_str, from_start_to_end_to_str, seconds_to_timedelta

from ..domain.input_entities import ActivityByStrategy, Event


def event_data_to_str(event: Event) -> str:
    """
    Converts an event data to a string.
    """
    if not event:
        return "{}"
    return str(event.data)


def event_to_str(event: Event) -> str:
    """
    Converts an event to a string.
    """
    return (
        f"{datetime_to_time_str(event.timestamp)}"
        f"..{datetime_to_time_str(event.timestamp + event.duration)}("
        f"{event_data_to_str(event)})"
    )


def activity_by_strategy_to_str(activitybs: ActivityByStrategy) -> str:
    """
    Converts `ActivityByStrategy` to a string.
    """
    return (
        f"{activitybs.id:>4}: {seconds_to_timedelta(activitybs.duration())} x{activitybs.density:.2f},"
        f" {from_start_to_end_to_str(activitybs.suggested_start_time, activitybs.suggested_end_time)}"
        f" (min {from_start_to_end_to_str(activitybs.max_start_time, activitybs.min_end_time)}),"
        f" {len(activitybs.events):>3} {activitybs.strategy.name} events grouped by {activitybs.grouping_data}."
    )


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
