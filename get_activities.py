#!/usr/bin/env python3
import argparse
import datetime
import os
from typing import Dict, List, Tuple

import aw_client

import activity_merger.config.config as config
from activity_merger.domain.analyzer import (
    AnalyzerStep,
    ChopActivitiesByResultTreeStep,
    DebugBucketsHandler,
    MakeCandidatesTreeStep,
    MakeResultTreeFromSelfSufficientActivitiesStep,
    MergeCandidatesTreeIntoResultTreeWithBIFinderStep,
    aggregate_strategies_results_to_activities,
)
from activity_merger.domain.basic_interval_finder import (
    FromCandidateActivitiesByScoreBIFinder,
    FromCandidatesByLogisticRegressionBIFinder,
    FromCandidatesByProximityAndDurationBIFinder,
    JiraIdBIFinder,
)
from activity_merger.domain.input_entities import Event
from activity_merger.domain.merger import apply_strategies_on_events
from activity_merger.domain.metrics import Metrics
from activity_merger.domain.output_entities import AnalyzerResult
from activity_merger.domain.strategies import StrategyApplyResult
from activity_merger.helpers.event_helpers import upload_events
from activity_merger.helpers.helpers import setup_logging, valid_date

LOG = setup_logging()


def delete_debug_buckets(client: aw_client.ActivityWatchClient) -> List[str]:
    """
    Deletes all debug buckets, returns names of them.
    """
    result = []
    try:
        buckets = client.get_buckets()
        for bucket_id in buckets:
            if bucket_id.startswith(config.DEBUG_BUCKET_PREFIX):
                client.delete_bucket(bucket_id, True)
                result.append(bucket_id)
    except Exception as ex:
        LOG.exception(
            "Can't connect to ActivityWatcher. Please check that it is enabled on localhost: %s", ex, exc_info=True
        )
        exit(1)
    return result


def clean_debug_buckets_and_apply_strategies_on_one_day_events(
    start_datetime: datetime.datetime, client: aw_client.ActivityWatchClient
) -> Tuple[List[StrategyApplyResult], Metrics]:
    """
    Connects to ActivityWatch, gets list of buckets and applies all strategies on all events in them.
    :param events_date: Date and time to get events for.
    :param client: ActivityWatch client to use.
    :return: Linked list of `Interval`-s.
    """
    try:
        # Remove debug buckets because they may become sources of events.
        LOG.info("Deleted [%s] debug buckets.", ", ".join(delete_debug_buckets(client)))
        # Get existing buckets.
        buckets = client.get_buckets()
    except Exception as ex:
        LOG.exception(
            "Can't connect to ActivityWatcher. Please check that it is enabled on localhost: %s", ex, exc_info=True
        )
        exit(1)
    LOG.info("Buckets to analyze: [%s]", ", ".join(buckets.keys()))
    return apply_strategies_on_events(
        client,
        start_datetime,
        start_datetime + datetime.timedelta(days=1),
        client.get_buckets(),
        config.STRATEGIES,
        config.EVENTS_COMPARE_TOLERANCE_TIMEDELTA,
    )


def load_debug_buckets(debug_dict: Dict[str, List[Event]], client: aw_client.ActivityWatchClient):
    """
    Uploads events representing analyzer debug information into ActivityWatch "debug" buckets.
    :param analyzer_result: Result to get data from.
    :param client: ActivityWatch client to use.
    """
    for bucket_id, events in debug_dict.items():
        LOG.info(upload_events(events, config.DEBUG_BUCKETS_IMPORTER_NAME, bucket_id, client=client))


def reload_debug_buckets(debug_dict: Dict[str, List[Event]], client: aw_client.ActivityWatchClient):
    """
    Uploads events representing analyzer debug information into ActivityWatch "debug" buckets. Removes these
    bucket preliminary.
    :param analyzer_result: Result to get data from.
    :param client: ActivityWatch client to use.
    """
    delete_debug_buckets(client)
    load_debug_buckets(debug_dict, client)


class UploadDebugBucketsStep(AnalyzerStep):
    """
    Step to upload debug buckets and clear them from `AnalyzerStep` context to don't re-upload later.
    """

    def __init__(self, client: aw_client.ActivityWatchClient, is_import_debug_buckets: bool):
        super(UploadDebugBucketsStep, self).__init__()
        self.client = client
        self.is_import_debug_buckets = is_import_debug_buckets

    def get_description(self) -> str:
        return "Uploading 'debug' buckets if need."

    def check_context(self, context: Dict[str, any]) -> None:
        if self.is_import_debug_buckets:
            assert "debug_buckets_handler" in context, "Need in 'debug_buckets_handler' property"

    def run(self, context: Dict[str, any], metrics: Metrics):
        if self.is_import_debug_buckets:
            debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
            load_debug_buckets(debug_buckets_handler.events, self.client)
            del context["debug_buckets_handler"]


BI_FINDERS = {
    "manual-score": FromCandidateActivitiesByScoreBIFinder(),
    "logistic-regression": FromCandidatesByLogisticRegressionBIFinder().with_coefs(
        config.BIFINDER_LOGISTIC_REGRESSION_COEF, config.BIFINDER_LOGISTIC_REGRESSION_INTERCEPT
    ),
    "proximity-duration": FromCandidatesByProximityAndDurationBIFinder(),
    "jira-id": JiraIdBIFinder(),
}


def convert_aw_events_to_activities(
    events_datetime: datetime.datetime,
    bi_finder_name: str,
    ignore_substrings: List[str],
    is_only_good_strategies_for_description: bool,
    is_import_debug_buckets: bool,
) -> AnalyzerResult:
    """
    Gets all ActivityWatch events for the specified date, builds linked list of intervals from them,
    analyzes intervals, converts them into combined activities by specified (and fine-tuned per person) rules,
    prints them into output.
    :param events_datetime: Date and time to get events on.
    :param ignore_substrings: List of substrings to ignore in logs (metrics substrings).
    :param is_use_all_strategies_for_description: Flag to use all strategies to build resulting activities description.
    :param is_import_debug_buckets: Flag to assemble and import into ActivityWatch debugging information as events for
    "debugging" buckets.
    :return: `AnalyzerResult` object or `None` if no intervals to analyze were found.
    """
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    bi_finder = BI_FINDERS[bi_finder_name]  # Fail on wrong bi_finder_name as soon as possible.
    # First load events for the day and print metrics.
    LOG.info("Starting to build activities per strategy for events starting on %s...", events_datetime)
    strategy_apply_result, metrics = clean_debug_buckets_and_apply_strategies_on_one_day_events(events_datetime, client)
    metrics_strings = list(metrics.to_strings(ignore_with_substrings=ignore_substrings))
    LOG.info("Analyzed all buckets separately, metrics:\n  %s", "\n  ".join(metrics_strings))
    LOG.info(
        "Got following activities-per-strategy:\n%s",
        "\n".join(x.to_string(ignore_metrics_by_substrings=ignore_substrings) for x in strategy_apply_result),
    )
    # Next analyze strategy_apply_result with multiple steps and print results as well.
    analyzer_result = aggregate_strategies_results_to_activities(
        strategy_apply_results=strategy_apply_result,
        steps=[
            MakeResultTreeFromSelfSufficientActivitiesStep(
                is_add_debug_buckets=True,
                is_only_good_strategies_for_description=is_only_good_strategies_for_description,
            ),
            ChopActivitiesByResultTreeStep(is_skip_afk=True, is_skip_self_sufficient_strategies=True),
            MakeCandidatesTreeStep(is_add_debug_buckets=is_import_debug_buckets),
            # Upload debug buckets before last most complext step - it may fail.
            UploadDebugBucketsStep(client=client, is_import_debug_buckets=is_import_debug_buckets),
            MergeCandidatesTreeIntoResultTreeWithBIFinderStep(
                bi_finder=bi_finder,
                is_add_debug_buckets=is_import_debug_buckets,
                is_only_good_strategies_for_description=is_only_good_strategies_for_description,
            ),
        ],
        ignore_substrings=ignore_substrings,
    )
    LOG.info(analyzer_result.to_str(ignore_metrics_by_substrings=ignore_substrings))
    # Import results back into ActivityWatch if need.
    if is_import_debug_buckets:
        load_debug_buckets(analyzer_result.debug_dict, client)
    return analyzer_result


def main():
    parser = argparse.ArgumentParser(
        description="Calls local ActivityWatch for all available events on specified date,"
        " analyzes all events to build list of activities per importer, merges them"
        " into list of actvities."
        "\nTo see debug logs need to set environment variable 'LOGLEVEL=debug',"
        " but they are mosly for ActivityWatch client communication debugging."
    )
    parser.add_argument(
        "date",
        nargs="?",
        type=valid_date,
        help="Date to analyze AcivityWatch events in format 'YYYY-mm-dd'. By-default is today."
        f" Note that day border is {config.DAY_BORDER}."
        " If don't set here then date is calculated as today-'back days'.",
    )
    parser.add_argument(
        "-b",
        "--back-days",
        type=int,
        help="How many days back search events on. I.e. '1' value means 'search for yesterday.",
    )
    parser.add_argument(
        "-f",
        "--bi-finder",
        dest="bi_finder_name",
        default=config.DEFAULT_BIFINDER,
        choices=BI_FINDERS.keys(),
        help="Basic interval finder to use."
        " Note that only 'simple' works with some predefined values, other need to tune.",
    )
    parser.add_argument(
        "-i",
        "--ignore-substrings",
        nargs="*",
        default=[],
        help="Substrings to ignore metrics or other logs with. Helps reduce log messages."
        " Just add substring from logs you don't want to see."
        " Supports multiple arguments therefore should be the last flag in the command line."
        ' Example of usage (are filtered logs containing "events with data" and "prefix b"):'
        ' `./get_activities.py -b 1 -i "events with data" "prefix b"`',
    )
    parser.add_argument(
        "-d",
        "--debug-buckets",
        dest="is_import_debug_buckets",
        action="store_true",
        help="Flag to import debugging buckets into ActivityWatch which allows to debug analyzing"
        f" behavior. All such bucket ID's starts with' {config.DEBUG_BUCKET_PREFIX}' string."
        " They are very handy on http://localhost:5600/#/timeline page "
        " (in ActivityWatch v0.12.1 need to refresh browser page to see them)."
        " Note that these debugging buckets are pre-removed (for the whole time)"
        " and aren't machine-specific. Also they may be quite heavy for UI to render.",
    )
    parser.add_argument(
        "-g",
        "--only-good-strategies-description",
        dest="is_only_good_strategies_for_description",
        action="store_true",
        help="Flag to build resulting activities description only from marked as"
        "'produces good activity name' strategies. Descriptions would be shorter but less informative.",
    )
    args = parser.parse_args()
    events_date = args.date if args.date else datetime.datetime.today().astimezone()
    if args.back_days and args.back_days > 0:
        events_date = events_date - datetime.timedelta(days=args.back_days)
    events_datetime = events_date.replace(hour=0, minute=0, second=0, microsecond=0).astimezone() + config.DAY_BORDER
    convert_aw_events_to_activities(
        events_datetime=events_datetime,
        bi_finder_name=args.bi_finder_name,
        ignore_substrings=list(args.ignore_substrings),
        is_only_good_strategies_for_description=args.is_only_good_strategies_for_description,
        is_import_debug_buckets=args.is_import_debug_buckets,
    )


if __name__ == "__main__":
    main()
