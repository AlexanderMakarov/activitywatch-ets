#!/usr/bin/env python3
import datetime
import os
import argparse
import aw_client
from typing import List

from activity_merger.config.config import LOG, EVENTS_COMPARE_TOLERANCE_TIMEDELTA, MIN_DURATION_SEC, RULES
from activity_merger.helpers.helpers import setup_logging, seconds_to_int_timedelta, valid_date
from activity_merger.domain.merger import report_from_buckets
from activity_merger.domain.analyzer import analyze_intervals, ProblemReporter
from activity_merger.domain.output_entities import Activity


def convert_aw_events_to_activities(events_date: datetime.datetime, ignore_hints: List[str]) -> List[Activity]:
    """
    Gets all ActivityWatch events for the specified date, builds linked list of intervals from them,
    analyzes intervals, converts them into combined activities by specified (and fine-tuned per person) rules,
    prints them into output.
    :param events_date: Date to get events on.
    :param ignore_hints: List of problems to disable in logs.
    """
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    try:
        buckets = client.get_buckets()
    except Exception as e:
        LOG.exception("Can't connect to ActivityWatcher. Please check that it is enabled on localhost: %s", e,
                      exc_info=True)
        exit(1)
    LOG.info("Buckets: [%s]", ", ".join(buckets.keys()))
    # Build time-ordered linked list of intervals by provided events.
    interval = report_from_buckets(client, events_date.date(), events_date.date() + datetime.timedelta(days=1),
        buckets, EVENTS_COMPARE_TOLERANCE_TIMEDELTA)
    if interval is None:
        LOG.warning("Can't find events/intervals for %s. Doing nothing.", events_date.date())
        return []
    # Convert (assemble) intervals list into activities.
    activities, activity_counter, metrics = analyze_intervals(interval, MIN_DURATION_SEC, RULES, ignore_hints)
    # Print metrics as is.
    sorted_metric_entries = sorted(metrics.items(), key=lambda x: x[1][1], reverse=True)
    LOG.info("Metrics from intervals analysis (%s):\n  %s",
             len(metrics),
             "\n  ".join(f"{x[1][0]:4} on {datetime.timedelta(seconds=x[1][1])} - {x[0]}" for x in sorted_metric_entries))
    # Print "less than MIN_DURATION_SEC" values from 'activity_counter'.
    dumb_activities = [
        f"{seconds_to_int_timedelta(v)} {k}" for k, v in activity_counter.most_common() if v >= MIN_DURATION_SEC
    ]
    LOG.info("There were %d 'equal' activities with %d longer than %d seconds:\n  %s",
             len(activity_counter), len(dumb_activities), MIN_DURATION_SEC, "\n  ".join(dumb_activities))
    # Print resulting activities as is. Order is important here.
    LOG.info("Assembled %d activities:\n  %s", len(activities), "\n  ".join(str(x) for x in activities))
    return activities


def main():
    parser = argparse.ArgumentParser(
        description="Calls AcivityWatch for all available events on specified date, "
                    "merges all events by specified rules into linked list of 'intervals' and"
                    "then separates this list into 'ready to import' actvities."
    )
    parser.add_argument('date', nargs='?', type=valid_date,
                        help="Date to analyze AcivityWatch events in format 'YYYY-mm-dd'. By-default is today. "
                             "If omit here but set 'back days' argument then date is calculated as today - back_days.")
    parser.add_argument('-b', '--back-days', type=int,
                        help="How many days back search events on. I.e. '1' value means 'search for yesterday.")
    parser.add_argument('-i', '--ignore-hints', nargs='*', default=[],
                        help="Hints to ignore in report. Helps filter log messages about rule mistakes. "
                             f"Supported values in importance order: {ProblemReporter.SUPPORTED_PROBLEMS}. "
                             "For example, to understand what need to setup for yourself with default config, use "
                             "'./get_activities.py 2022-12-31 -i TOO_SPECIFIC_RULE TOO_WIDE_RULE' and after it stop "
                             "to report issues remove '-i' part.")
    args = parser.parse_args()
    events_date = args.date if args.date else datetime.datetime.today().astimezone()
    if args.back_days and args.back_days > 0:
        events_date = (events_date - datetime.timedelta(days=args.back_days))
    # TODO need interactive way to merge activities
    convert_aw_events_to_activities(events_date, args.ignore_hints)


if __name__ == '__main__':
    LOG = setup_logging()
    main()
