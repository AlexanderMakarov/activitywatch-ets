#!/usr/bin/env python3
import datetime
import os
import argparse
import json
import aw_client
from activity_merger.helpers.helpers import setup_logging, valid_date


FILE_NAME_PREFIX = os.path.basename(__file__) + "_events_"
FILE_NAME_SUFFIX = ".json"


LOG = setup_logging()


def main():
    parser = argparse.ArgumentParser(
        description="Export all buckets events from AcivityWatch for one given date."
        " Expected to be used for migrating chunks of data from one machine to another for experiments."
    )
    parser.add_argument(
        "date", nargs="?", type=valid_date, help="Date to analyze AcivityWatch events in format 'YYYY-mm-dd'."
    )
    args = parser.parse_args()
    events_date: datetime.datetime = args.date
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    buckets: dict = client.get_buckets()
    result = {"buckets": dict()}  # Prepare structure which ActivityWatch produces on its "export all" endpoint.
    LOG.info("Get %d buckets, trying to get events on %s...", len(buckets), events_date)
    for bucket in buckets.values():
        end_date = events_date + datetime.timedelta(days=1)
        bucket_id = bucket["id"]
        events = client.get_events(bucket_id, start=events_date, end=end_date)
        LOG.info("Got %d events from '%s' bucket on %s", len(events), bucket_id, events_date)
        if events:
            for event in events:  # Modify events to "importable" version.
                event.pop("id")
                event["duration"] = event.duration.total_seconds()
            bucket["events"] = events
            result["buckets"][bucket_id] = bucket
    file_name = FILE_NAME_PREFIX + str(events_date) + FILE_NAME_SUFFIX
    with open(file_name, "w") as f:
        json.dump(result, f, indent=None, default=str)
    LOG.info(
        "Dumped events into '%s' nearby. Use 'Raw Data' tab in ActivityWatch UI to export this data."
        " In case of 'peewee.IntegrityError: UNIQUE constraint failed: bucketmodel.id' error in aw-server logs"
        " please consider removing buckets which you are going to import.",
        file_name,
    )


if __name__ == "__main__":
    main()
