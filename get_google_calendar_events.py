#!/usr/bin/env python3
import argparse
import datetime
import logging
from typing import List, Set, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build

from activity_merger.config.config import (
    DAY_BORDER,
    GOOGLE_CALENDAR_BUCKET_ID,
    GOOGLE_CALENDAR_ID,
    GOOGLE_SERVICE_ACCOUNT_KEY_PATH,
    GOOGLE_CALENDAR_SCRAPER_NAME,
    LOG,
)
from activity_merger.domain.input_entities import Event
from activity_merger.helpers.event_helpers import event_to_str, upload_events
from activity_merger.helpers.helpers import datetime_to_time_str, ensure_datetime, setup_logging, valid_date


LOG = setup_logging()


def _get_gcalendar_events_one_day(gcalendar_service, calendar_id: str, search_datetime: datetime.datetime) -> List:
    assert calendar_id, "Calendar ID is not specified."
    assert search_datetime, "Date to search is not specified."

    LOG.info("Getting '%s' Google Calendar events after %s", calendar_id, search_datetime)
    events_result = (
        gcalendar_service.events()
        .list(
            calendarId=calendar_id,
            timeMin=search_datetime.isoformat(),
            timeMax=(search_datetime + datetime.timedelta(days=1)).isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return events_result.get("items", [])


def _parse_iso8601(datetime_str):
    return datetime.datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S%z')


def _convert_gcalendar_events_to_aw_events(gcalendar_events: List) -> List[Event]:
    events = []
    for event in gcalendar_events:
        start_time = _parse_iso8601(event['start']['dateTime'])
        end_time = _parse_iso8601(event['end']['dateTime'])
        events.append(
            Event(
                GOOGLE_CALENDAR_BUCKET_ID,
                start_time,
                (end_time - start_time),
                {
                    "summary": event['summary'],
                    "organizer": event['organizer']['email'],
                    "updated": str(event['updated']),
                    "eventType": event['eventType'],
                },
            )
        )
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Calls Google Calendar API to get events on the given date,"
        " parses all found events in it and loads them into ActivityWatch."
        " To see more logs use `export LOGLEVEL=DEBUG` (or `set ...` on Windows)."
    )
    parser.add_argument(
        "search_date",
        nargs="?",
        type=valid_date,
        default=datetime.datetime.now().astimezone(),
        help="Date to look for Google Calendar events in format 'YYYY-mm-dd'. By default is today."
        f" Note that day border is {DAY_BORDER}."
        " If don't set here then date is calculated as today-'back days'.",
    )
    parser.add_argument(
        "-b",
        "--back-days",
        type=int,
        help="Overwrites 'date' if specified. Sets how many days back search events on."
        " I.e. '1' value means 'search for yesterday'.",
    )
    parser.add_argument(
        "-c",
        "--calendar-name",
        type=str,
        default=GOOGLE_CALENDAR_ID,
        help="ID or name of Google Calendar to get events from.",
    )
    parser.add_argument(
        "--service-account-file",
        type=str,
        default=GOOGLE_SERVICE_ACCOUNT_KEY_PATH,
        help="Path to service account file for which is shared Google Calendar." " TODO",
    )
    parser.add_argument(
        "-r",
        "--replace",
        dest="is_replace_bucket",
        action="store_true",
        help=f"Flag to delete ActivityWatch '{GOOGLE_CALENDAR_BUCKET_ID}' bucket first."
        " Removes all previous events in it, for all time.",
    )
    parser.add_argument(
        "--dry-run",
        dest="is_dry_run",
        action="store_true",
        help="Flag to just log events but don't upload into ActivityWatch.",
    )
    args = parser.parse_args()
    search_date = args.search_date
    if args.back_days:
        assert args.back_days >= 0, f"'back_days' value ({args.back_days}) should be positive or 0."
        search_date = datetime.datetime.today().astimezone() - datetime.timedelta(days=args.back_days)
    search_datetime = ensure_datetime(search_date).replace(hour=0, minute=0, second=0, microsecond=0) + DAY_BORDER
    # Get Google Calendar events.
    credentials = service_account.Credentials.from_service_account_file(
        args.service_account_file, scopes=["https://www.googleapis.com/auth/calendar.readonly"]
    )
    gcalendar_service = build("calendar", "v3", credentials=credentials)
    gcalendar_events = _get_gcalendar_events_one_day(gcalendar_service, args.calendar_name, search_datetime)
    LOG.info("Received %d events from '%s' Google Calendar.", len(gcalendar_events), args.calendar_name)
    if not gcalendar_events:
        LOG.warning("Can't find '%s' Google Calendar events on %s.", args.calendar_name, search_datetime)
        return
    events = _convert_gcalendar_events_to_aw_events(gcalendar_events)
    LOG.info("Ready to upload %d events:\n  %s", len(events), "\n  ".join(event_to_str(x) for x in events))
    # Load events into ActivityWatcher
    if not args.is_dry_run:
        LOG.info(
            upload_events(
                events,
                GOOGLE_CALENDAR_SCRAPER_NAME,
                GOOGLE_CALENDAR_BUCKET_ID,
                args.is_replace_bucket,
                aw_client_name="gcalendar.events",
            )
        )


if __name__ == "__main__":
    LOG = setup_logging()
    main()
