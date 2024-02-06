#!/usr/bin/env python3
import argparse
import datetime
import json
import os

import aw_client

from activity_merger.domain.input_entities import Event
from activity_merger.helpers.event_helpers import upload_events, delete_bucket_events_for_interval
from activity_merger.helpers.helpers import setup_logging, valid_date

FILE_NAME_PREFIX = os.path.basename(__file__) + "_events_"
FILE_NAME_SUFFIX = ".json"
LOG = setup_logging()


def main():
    parser = argparse.ArgumentParser(
        description="Imports on AcivityWatch server given JSON file with multiple buckets events."
        " Doesn't fail with 'UNIQUE constraint failed: bucketmodel.id' in server logs if bucket exists "
        "unlike UI 'Import and export buckets' option."
        " Also corrects ID (if possible) for buckets without hostname suffix to don't mix computers."
        " Uses JSON files downloaded from AcivityWatch ('Export bucket as JSON' option on bucket) "
        "or created by 'aw_export_one_day.py' script in this repo."
        " Expected to be used for migrating chunks of data from one machine to another for experiments."
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=argparse.FileType("r", encoding="UTF-8"),
        help="Path to JSON file with events to import.",
    )
    parser.add_argument(
        "-c",
        "--pre-clean-day",
        dest="pre_clean_day",
        type=valid_date,
        help="Date in YYYY-MM-DD format to remove events for that date for all buckets which are imported.",
    )
    parser.add_argument(
        "-r",
        "--replace",
        dest="is_replace_buckets",
        action="store_true",
        help="Flag to delete all buckets first. USE WITH CAUTION! Removes all buckets found in file, for all times.",
    )
    args = parser.parse_args()
    data = json.load(args.file)
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    valid_hostname = "unknown"
    for bucket in data["buckets"].values():
        bucket_id = bucket["id"]
        # Correct "hostname" property. For example 'aw-watcher-web-firefox' usually is created without a hostname.
        hostname = bucket["hostname"]
        if hostname != "unknown":
            valid_hostname = hostname
        if valid_hostname not in bucket_id:
            bucket_id = f"{bucket_id}_{valid_hostname}"
        # Build list of events for this bucket.
        events = [Event(bucket_id, x["timestamp"], x["duration"], dict(x["data"])) for x in bucket["events"]]
        if args.pre_clean_day:
            end_date = args.pre_clean_day + datetime.timedelta(days=1)
            delete_bucket_events_for_interval(client, bucket_id, args.pre_clean_day, end_date)
        LOG.info(upload_events(events, bucket["type"], bucket_id, args.is_replace_buckets, bucket["client"], client))
    LOG.info("Done")


if __name__ == "__main__":
    main()
