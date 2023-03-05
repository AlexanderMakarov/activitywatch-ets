#!/usr/bin/env python3
import datetime
import os
import argparse
import aw_client
from typing import List

from activity_merger.config.config import LOG, EVENTS_COMPARE_TOLERANCE_TIMEDELTA, MIN_DURATION_SEC, RULES,\
                                          DEBUG_BUCKETS_IMPORTER_NAME, BUCKET_DEBUG_RAW_RULE_RESULTS,\
                                          BUCKET_DEBUG_FINAL_RULE_RESULTS, BUCKET_DEBUG_ACTIVITES
from activity_merger.domain.interval import Interval
from activity_merger.helpers.helpers import setup_logging, seconds_to_int_timedelta, valid_date, upload_events
from activity_merger.domain.merger import report_from_buckets
from activity_merger.domain.analyzer import analyze_intervals, ProblemReporter, ANALYZE_MODE_ACTIVITIES,\
                                            ANALYZE_MODE_DEBUG
from activity_merger.domain.output_entities import AnalyzerResult


def get_interval(events_date: datetime.datetime, client: aw_client.ActivityWatchClient) -> Interval:
    """
    Connects to ActivityWatch, gets list of buckets and builds linked list of `Interval`-s representing events.
    :param events_date: Date to get events for.
    :param client: ActivityWatch client to use.
    :return: Linked list of `Interval`-s.
    """
    try:
        # Remove debug buckets because they may become sources of events.
        client.delete_bucket(BUCKET_DEBUG_RAW_RULE_RESULTS, True)
        client.delete_bucket(BUCKET_DEBUG_FINAL_RULE_RESULTS, True)
        client.delete_bucket(BUCKET_DEBUG_ACTIVITES, True)
        # Get existing buckets.
        buckets = client.get_buckets()
    except Exception as ex:
        LOG.exception("Can't connect to ActivityWatcher. Please check that it is enabled on localhost: %s", ex,
                      exc_info=True)
        exit(1)
    LOG.info("Buckets: [%s]", ", ".join(buckets.keys()))
    # Build time-ordered linked list of intervals by provided events.
    return report_from_buckets(client, events_date.date(), events_date.date() + datetime.timedelta(days=1),
                               buckets, EVENTS_COMPARE_TOLERANCE_TIMEDELTA)


def upload_debug_buckets(analyzer_result: AnalyzerResult, client: aw_client.ActivityWatchClient):
    """
    Uploads events representing analyzer debug information into ActivityWatch "debug" buckets. Removes these
    bucket preliminary.
    :param analyzer_result: Result to get data from.
    :param client: ActivityWatch client to use.
    """
    if analyzer_result.raw_rule_result_debug_events:
        LOG.info(upload_events(analyzer_result.raw_rule_result_debug_events, DEBUG_BUCKETS_IMPORTER_NAME,
                               BUCKET_DEBUG_RAW_RULE_RESULTS, True, client=client))
    if analyzer_result.final_rule_result_debug_events:
        LOG.info(upload_events(analyzer_result.final_rule_result_debug_events, DEBUG_BUCKETS_IMPORTER_NAME,
                               BUCKET_DEBUG_FINAL_RULE_RESULTS, True, client=client))
    if analyzer_result.activity_debug_events:
        LOG.info(upload_events(analyzer_result.activity_debug_events, DEBUG_BUCKETS_IMPORTER_NAME,
                               BUCKET_DEBUG_ACTIVITES, True, client=client))


def print_analyzer_result(analyzer_result: AnalyzerResult):
    """
    Prints 'AnalyzerResult' content as INFO logs.
    :param analyzer_result: Object to describe data in.
    """
    sorted_metric_entries = sorted(analyzer_result.metrics.items(), key=lambda x: x[1][1], reverse=True)
    LOG.info("Metrics from intervals analysis (%s):\n  %s",
             len(analyzer_result.metrics),
             "\n  ".join(f"{x[1][0]:4} on {datetime.timedelta(seconds=x[1][1])} - {x[0]}"
                         for x in sorted_metric_entries))
    # Print "less than MIN_DURATION_SEC" values from 'activity_counter'.
    dumb_activities = [f"{seconds_to_int_timedelta(v)} {k}"
                       for k, v in analyzer_result.rule_results_counter.most_common() if v >= MIN_DURATION_SEC]
    LOG.info("There were %d 'equal' activities with %d longer than %d seconds:\n  %s",
             len(analyzer_result.rule_results_counter), len(dumb_activities), MIN_DURATION_SEC,
             "\n  ".join(dumb_activities))
    # Print resulting activities as is. Order is important here.
    LOG.info("Assembled %d activities:\n  %s", len(analyzer_result.activities),
             "\n  ".join(str(x) for x in analyzer_result.activities))


def convert_aw_events_to_activities(events_date: datetime.datetime, ignore_hints: List[str],
                                    is_import_debug_buckets: bool) -> AnalyzerResult:
    """
    Gets all ActivityWatch events for the specified date, builds linked list of intervals from them,
    analyzes intervals, converts them into combined activities by specified (and fine-tuned per person) rules,
    prints them into output.
    :param events_date: Date to get events on.
    :param ignore_hints: List of problems to disable in logs.
    :param is_import_debug_buckets: Flag to assemble and import into ActivityWatch debugging information as events for
    "debugging" buckets.
    :return: 'AnalyzerResult' object or 'None' if no intervals to analyze were found.
    """
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    # Build time-ordered linked list of intervals by provided events.
    interval = get_interval(events_date, client)
    if interval is None:
        LOG.warning("Can't find events/intervals for %s. Doing nothing.", events_date.date())
        return None
    # Convert (analyze) intervals list into activities.
    analyzer_result: AnalyzerResult = analyze_intervals(
        interval, MIN_DURATION_SEC, RULES, ignore_hints,
        ANALYZE_MODE_DEBUG if is_import_debug_buckets else ANALYZE_MODE_ACTIVITIES
    )
    print_analyzer_result(analyzer_result)
    if is_import_debug_buckets:
        upload_debug_buckets(analyzer_result, client)
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
    parser.add_argument('-i', '--ignore-hints', nargs='*', default=[],
                        help="Hints to ignore in report. Helps filter log messages about rule mistakes. "
                             f"Supported values in importance order: {ProblemReporter.SUPPORTED_PROBLEMS}. "
                             "For example, to understand what need to setup for yourself with default config, use "
                             "'./get_activities.py 2022-12-31 -i TOO_SPECIFIC_RULE TOO_WIDE_RULE' and after it stop "
                             "to report issues remove '-i' part.")
    parser.add_argument('-d', '--debug-buckets', dest='is_import_debug_buckets', action='store_true',
                        help="Flag to import debugging buckets into ActivityWatch which allows represent rules "
                             "behavior. They are very handy on http://localhost:5600/#/timeline page "
                             "(in ActivityWatch v0.12.1 need to refresh browser page to get them). "
                             " Note that these debugging buckets are pre-removed (for the whole time) "
                             "and aren't machine-specific. Also they are quite heavy for UI."
                             f"'{BUCKET_DEBUG_RAW_RULE_RESULTS}' bucket contains list of raw 'RuleResult'-s. "
                             "I.e. rule found for each interval, before applying 'skip' or 'placeholder' features."
                             f"'{BUCKET_DEBUG_FINAL_RULE_RESULTS}' bucket contains list of final 'RuleResult'-s."
                             "I.e. rule for intervals which will contribute into final report."
                             f"'{BUCKET_DEBUG_ACTIVITES}' bucket contains list of resulting activities.")
    args = parser.parse_args()
    events_date = args.date if args.date else datetime.datetime.today().astimezone()
    if args.back_days and args.back_days > 0:
        events_date = (events_date - datetime.timedelta(days=args.back_days))
    convert_aw_events_to_activities(events_date, args.ignore_hints, args.is_import_debug_buckets)


if __name__ == '__main__':
    LOG = setup_logging()
    main()
