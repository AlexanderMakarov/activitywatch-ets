#!/usr/bin/env python3
import datetime
import os
import argparse
import aw_client
from typing import List, Set, Tuple

from activity_merger.config.config import LOG, EVENTS_COMPARE_TOLERANCE_TIMEDELTA, MIN_DURATION_SEC, RULES,\
                                          DEBUG_BUCKETS_IMPORTER_NAME, BUCKET_DEBUG_RAW_RULE_RESULTS,\
                                          BUCKET_DEBUG_FINAL_RULE_RESULTS, BUCKET_DEBUG_ACTIVITES, \
                                          STRATEGIES
from activity_merger.domain.interval import Interval
from activity_merger.domain.metrics import Metrics
from activity_merger.domain.strategies import ActivitiesByStrategy
from activity_merger.helpers.helpers import setup_logging, valid_date, upload_events, delete_buckets
from activity_merger.domain.merger import report_from_buckets, analyze_buckets
from activity_merger.domain.analyzer import analyze_intervals, ProblemReporter, merge_activities
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
        delete_buckets([BUCKET_DEBUG_RAW_RULE_RESULTS, BUCKET_DEBUG_FINAL_RULE_RESULTS, BUCKET_DEBUG_ACTIVITES],
                       client)
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
        delete_buckets([BUCKET_DEBUG_RAW_RULE_RESULTS, BUCKET_DEBUG_FINAL_RULE_RESULTS, BUCKET_DEBUG_ACTIVITES],
                       client)
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
    delete_buckets([BUCKET_DEBUG_RAW_RULE_RESULTS, BUCKET_DEBUG_FINAL_RULE_RESULTS, BUCKET_DEBUG_ACTIVITES], client)
    if analyzer_result.raw_rule_result_debug_events:
        LOG.info(upload_events(analyzer_result.raw_rule_result_debug_events, DEBUG_BUCKETS_IMPORTER_NAME,
                               BUCKET_DEBUG_RAW_RULE_RESULTS, True, client=client))
    if analyzer_result.final_rule_result_debug_events:
        LOG.info(upload_events(analyzer_result.final_rule_result_debug_events, DEBUG_BUCKETS_IMPORTER_NAME,
                               BUCKET_DEBUG_FINAL_RULE_RESULTS, True, client=client))
    if analyzer_result.activity_debug_events:
        LOG.info(upload_events(analyzer_result.activity_debug_events, DEBUG_BUCKETS_IMPORTER_NAME,
                               BUCKET_DEBUG_ACTIVITES, True, client=client))


def convert_aw_events_to_activities(events_date: datetime.datetime, ignore_hints: Set[str],
                                    is_import_debug_buckets: bool) -> AnalyzerResult:
    """
    Gets all ActivityWatch events for the specified date, builds linked list of intervals from them,
    analyzes intervals, converts them into combined activities by specified (and fine-tuned per person) rules,
    prints them into output.
    :param events_date: Date to get events on.
    :param ignore_hints: Set of problems to disable in logs.
    :param is_import_debug_buckets: Flag to assemble and import into ActivityWatch debugging information as events for
    "debugging" buckets.
    :return: `AnalyzerResult` object or `None` if no intervals to analyze were found.
    """
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    # # Build time-ordered linked list of intervals by provided events.
    # interval = get_interval(events_date, client)
    # if interval is None:
    #     LOG.warning("Can't find events/intervals for %s. Doing nothing.", events_date.date())
    #     return None
    # # Convert (analyze) intervals list into activities.
    # analyzer_result: AnalyzerResult = analyze_intervals(interval, MIN_DURATION_SEC, RULES, is_import_debug_buckets,
    #                                                     ignore_hints)
    LOG.info("Starting to build activities per strategy...")
    activities_by_strategy, metrics = get_activities_by_strategy(events_date, client)
    LOG.info("Analyzed all buckets separately. Results:%s", metrics)
    LOG.info("Starting to assemble resulting activities from all strategies...")
    # TODO add ability to skip metrics starting with 'events with data '.
    LOG.info("Got following activities-per-strategy:\n>>> %s", "\n>>> ".join(str(x) for x in activities_by_strategy))

    analyzer_result = merge_activities(activities_by_strategy)
    LOG.info(analyzer_result.to_str())
    # LOG.info(analyzer_result.to_str(append_equal_intervals_longer_that=MIN_DURATION_SEC))
    # if is_import_debug_buckets:
    #     reload_debug_buckets(analyzer_result, client)
    # return analyzer_result


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
                             f"Supported values in importance order: {ProblemReporter.SUPPORTED_ITEMS.keys()}. "
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
    convert_aw_events_to_activities(events_date, set(args.ignore_hints), args.is_import_debug_buckets)


if __name__ == '__main__':
    LOG = setup_logging()
    main()
