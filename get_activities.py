#!/usr/bin/env python3
import datetime
import logging
import argparse
import aw_client

from activity_merger.config.config import LOG, EVENTS_COMPARE_TOLERANCE_TIMEDELTA, MIN_DURATION_SEC, RULES
from activity_merger.helpers.helpers import seconds_to_int_timedelta
from activity_merger.domain.merger import report_from_buckets
from activity_merger.domain.analyzer import analyze_intervals


def valid_date(s):  # https://stackoverflow.com/a/25470943
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Calls ActivityWatcher for all available events on specified date, "
                    "merges all events by specified rules into linked list of some activities and"
                    "then separates this list into 'ready to import' actvities."
    )
    parser.add_argument('date', nargs='?', type=valid_date, default=datetime.datetime.now().date(),
                        help="Date to build activities for in format 'YYYY-mm-dd'. By-default today.")
    args = parser.parse_args()
    # TODO need interactive way to merge activities
    # TODO need mixing of Jira/Outlook/watchdog events.

    client = aw_client.ActivityWatchClient("activity_merger.py")
    buckets = client.get_buckets()
    LOG.info(f"Buckets: {buckets.keys()}")
    # Build time-ordered linked list of intervals by provided events.
    interval = report_from_buckets(
        client,
        # TODO args.date, args.date + datetime.timedelta(days=1),
        datetime.datetime(2022, 2, 11),
        datetime.datetime(2022, 2, 12),
        buckets,
        EVENTS_COMPARE_TOLERANCE_TIMEDELTA
    )
    # Convert (assemble) intervals list into activities.
    activities, activity_counter, metrics = analyze_intervals(interval, MIN_DURATION_SEC, RULES)
    # Print metrics as is.
    LOG.info(f"Metrics from intervals analysis ({len(metrics)}):" + "\n  "
             + "\n  ".join(f"{v[0]:4} on {datetime.timedelta(seconds=v[1])} - {k}" for k, v in metrics.items()))
    # Pring only "less than MIN_DURATION_SEC" from "dumb" activities.
    dumb_activities = [
        f"{seconds_to_int_timedelta(v)} {k}" for k, v in activity_counter.most_common() if v >= MIN_DURATION_SEC
    ]
    LOG.info("There were %d equal activities. Longer than %d are:\n  %s"
             % (len(activity_counter), MIN_DURATION_SEC, "\n  ".join(dumb_activities)))
    # Print all activities as is.
    LOG.info("Assembled %d activities:\n  %s" % (len(activities), "\n  ".join(str(x) for x in activities)))


def setup_logging():
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.DEBUG, "DEBU")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)-4s: %(message)s',
        datefmt="%H:%M:%S"
    )
    return logging.getLogger()


if __name__ == '__main__':
    LOG = setup_logging()
    main()
