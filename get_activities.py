#!/usr/bin/env python3
import datetime
import os
import argparse
import aw_client
from typing import List, Set, Tuple

from activity_merger.config.config import LOG, EVENTS_COMPARE_TOLERANCE_TIMEDELTA, MIN_DURATION_SEC,\
                                          DEBUG_BUCKETS_IMPORTER_NAME, DEBUG_BUCKET_PREFIX, STRATEGIES
from activity_merger.domain.interval import Interval
from activity_merger.domain.metrics import Metrics
from activity_merger.domain.strategies import ActivitiesByStrategy
from activity_merger.helpers.helpers import setup_logging, valid_date, upload_events, delete_buckets
from activity_merger.domain.merger import report_from_buckets, analyze_buckets
from activity_merger.domain.analyzer import analyze_intervals, ProblemReporter, analyze_activities_per_strategy
from activity_merger.domain.output_entities import AnalyzerResult


def delete_debug_buckets(client: aw_client.ActivityWatchClient) -> List[str]:
    result = []
    try:
        buckets = client.get_buckets()
        for bucket_id in buckets:
            if bucket_id.startswith(DEBUG_BUCKET_PREFIX):
                client.delete_bucket(bucket_id, True)
                result.append(bucket_id)
    except Exception as ex:
        LOG.exception("Can't connect to ActivityWatcher. Please check that it is enabled on localhost: %s", ex,
                      exc_info=True)
        exit(1)
    return result


def get_interval(events_date: datetime.datetime, client: aw_client.ActivityWatchClient) -> Interval:
    """
    Connects to ActivityWatch, gets list of buckets and builds linked list of `Interval`-s representing events.
    :param events_date: Date to get events for.
    :param client: ActivityWatch client to use.
    :return: Linked list of `Interval`-s.
    """
    try:
        # Remove debug buckets because they may become sources of events.
        LOG.info("Deleted [%s] debug buckets.", ", ".join(delete_debug_buckets(client)))
        # Get existing buckets.
        buckets = client.get_buckets()
    except Exception as ex:
        LOG.exception("Can't connect to ActivityWatcher. Please check that it is enabled on localhost: %s", ex,
                      exc_info=True)
        exit(1)
    LOG.info("Buckets to analyze: [%s]", ", ".join(buckets.keys()))
    # Build time-ordered linked list of intervals by provided events.
    return report_from_buckets(client, events_date.date(), events_date.date() + datetime.timedelta(days=1),
                               buckets, EVENTS_COMPARE_TOLERANCE_TIMEDELTA)


def get_activities_by_strategy(events_date: datetime.datetime, client: aw_client.ActivityWatchClient)\
        -> Tuple[List[ActivitiesByStrategy], Metrics]:
    """
    Connects to ActivityWatch, gets list of buckets and applies all strategies on all events in them.
    :param events_date: Date to get events for.
    :param client: ActivityWatch client to use.
    :return: Linked list of `Interval`-s.
    """
    try:
        # Remove debug buckets because they may become sources of events.
        LOG.info("Deleted [%s] debug buckets.", ", ".join(delete_debug_buckets(client)))
        # Get existing buckets.
        buckets = client.get_buckets()
    except Exception as ex:
        LOG.exception("Can't connect to ActivityWatcher. Please check that it is enabled on localhost: %s", ex,
                      exc_info=True)
        exit(1)
    LOG.info("Buckets to analyze: [%s]", ", ".join(buckets.keys()))
    return analyze_buckets(
        client, events_date.date(), events_date.date() + datetime.timedelta(days=1), client.get_buckets(), STRATEGIES,
        EVENTS_COMPARE_TOLERANCE_TIMEDELTA
    )


def reload_debug_buckets(analyzer_result: AnalyzerResult, client: aw_client.ActivityWatchClient):
    """
    Uploads events representing analyzer debug information into ActivityWatch "debug" buckets. Removes these
    bucket preliminary.
    :param analyzer_result: Result to get data from.
    :param client: ActivityWatch client to use.
    """
    delete_debug_buckets(client)
    for bucket_id, events in analyzer_result.debug_dict.items():
        LOG.info(upload_events(events, DEBUG_BUCKETS_IMPORTER_NAME, bucket_id, client=client))


def convert_aw_events_to_activities(events_date: datetime.datetime, ignore_substrings: List[str],
                                    is_import_debug_buckets: bool) -> AnalyzerResult:
    """
    Gets all ActivityWatch events for the specified date, builds linked list of intervals from them,
    analyzes intervals, converts them into combined activities by specified (and fine-tuned per person) rules,
    prints them into output.
    :param events_date: Date to get events on.
    :param ignore_substrings: List of substrings to ignore metrics with them in logs.
    :param is_import_debug_buckets: Flag to assemble and import into ActivityWatch debugging information as events for
    "debugging" buckets.
    :return: `AnalyzerResult` object or `None` if no intervals to analyze were found.
    """
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    LOG.info("Starting to build activities per strategy...")
    activities_by_strategy, metrics = get_activities_by_strategy(events_date, client)
    metrics_strings = list(metrics.to_strings(ignore_with_substrings=ignore_substrings))
    LOG.info("Analyzed all buckets separately:\n  %s", "\n  ".join(metrics_strings))
    LOG.info("Got following activities-per-strategy:\n%s",
             "\n".join(x.to_string(ignore_metrics_by_substrings=ignore_substrings) for x in activities_by_strategy))

    analyzer_result = analyze_activities_per_strategy(activities_by_strategy, is_import_debug_buckets)
    LOG.info(analyzer_result.to_str(ignore_metrics_by_substrings=ignore_substrings))
    if is_import_debug_buckets:
        reload_debug_buckets(analyzer_result, client)
    return analyzer_result


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
    parser.add_argument('-i', '--ignore-substrings', nargs='*', default=[],
                        help="Substrings to ignore metrics or other logs with. Helps reduce log messages."
                             " Just add substring from logs you don't want to see."
                             " Supports multiple arguments therefore should be the last flag in the command line."
                             ' Example of usage (are filtered logs containing "events with data" and "prefix b"):'
                             ' `./get_activities.py -b 1 -i "events with data" "prefix b"`')
    parser.add_argument('-d', '--debug-buckets', dest='is_import_debug_buckets', action='store_true',
                        help="Flag to import debugging buckets into ActivityWatch which allows to debug analyzing"
                             f" behavior. All such bucket ID's starts with' {DEBUG_BUCKET_PREFIX}' string."
                             " They are very handy on http://localhost:5600/#/timeline page "
                             " (in ActivityWatch v0.12.1 need to refresh browser page to see them)."
                             " Note that these debugging buckets are pre-removed (for the whole time)"
                             " and aren't machine-specific. Also they may be quite heavy for UI to render.")
    args = parser.parse_args()
    events_date = args.date if args.date else datetime.datetime.today().astimezone()
    if args.back_days and args.back_days > 0:
        events_date = (events_date - datetime.timedelta(days=args.back_days))
    convert_aw_events_to_activities(events_date, list(args.ignore_substrings), args.is_import_debug_buckets)


if __name__ == '__main__':
    LOG = setup_logging()
    main()
