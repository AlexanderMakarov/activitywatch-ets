#!/usr/bin/env python3
import datetime
from typing import List, Set, Tuple
import argparse
import difflib
import jira
import logging

from activity_merger.config.config import LOG, JIRA_SCRAPER_NAME, JIRA_BUCKET_ID, JIRA_URL, JIRA_LOGIN_EMAIL,\
    JIRA_LOGIN_API_TOKEN, JIRA_PROJECTS, JIRA_ISSUES_MAX, EVENTS_COMPARE_TOLERANCE_TIMEDELTA
from activity_merger.helpers.helpers import setup_logging, valid_date, ensure_datetime, upload_events, event_to_str,\
                                            CURRENT_TIMEZONE
from activity_merger.domain.input_entities import Event


JIRA_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
JIRA_TICKET_FIELDS_VARIABLE_UPDATE_COMPLEXITY = {'description', 'summary', 'labels', 'Component'}


def _get_jira_issues(server_url: str, email: str, api_token: str, projects: List[str], search_date: datetime)\
        -> List[jira.Issue]:
    assert server_url, "Jira server URL is not specified."
    assert email, "Jira login email is not specified."
    assert api_token, "Jira login API token is not specified."
    assert projects, "Jira projects to consider are not specified."
    assert search_date, "Date to search is not specified."
    connection = jira.JIRA(server=server_url, basic_auth=(email, api_token))
    # end_date = search_date.date() + datetime.timedelta(days=1)
    LOG.info(f"Searching {projects} issues updated during or after {search_date.date()} and touched by {email}.")
    jql = f"project IN ({','.join(projects)}) AND updated >= '{search_date.date()}' AND ("\
          "reporter was currentUser()"\
          " OR commentedBy = currentUser()"\
          " OR assignee was currentUser()"\
          " OR status changed BY currentUser())"
    return connection.search_issues(jql, expand="changelog", maxResults=JIRA_ISSUES_MAX)


def _jira_story_item_field_to_string(field) -> str:
    return str(field) if field else ""


def _calculate_diff(old_value: str, new_value: str) -> Tuple[str, int]:
    change_desc = []
    symbols_count = 0
    # Treat inputs as arrays of characters to get precise change.
    for opcode in difflib.SequenceMatcher(None, old_value, new_value).get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag in {'replace', 'delete'}:
            change_desc.append("-" + old_value[i1:i2])
            symbols_count += i2 - i1
        if tag in {'replace', 'insert'}:
            change_desc.append("+" + new_value[j1:j2])
            symbols_count += j2 - j1
    change_desc = '\n'.join(change_desc)
    return change_desc, symbols_count


def _parse_events_from_story_without_duration(story: jira.resources.PropertyHolder, jira_id: str,\
        summary: str, created: datetime.datetime) -> Tuple[List[Event], Set[str]]:
    unsupported_fields = set()  # For debugging custom Jira projects/servers.
    events = []
    for item in story.items:
        field = item.field
        symbols_count = 1  # 1 is default for unsupported fields.
        change_desc = item.toString
        if field in JIRA_TICKET_FIELDS_VARIABLE_UPDATE_COMPLEXITY:
            new_value = _jira_story_item_field_to_string(item.toString)
            old_value = _jira_story_item_field_to_string(item.fromString)
            # Calculate difference in text. Treat inputs as arrays of characters.
            change_desc, symbols_count = _calculate_diff(old_value, new_value)
        elif field in {'Link'}:
            new_value = _jira_story_item_field_to_string(item.to)
            old_value = _jira_story_item_field_to_string(item.__dict__['from'])  # 'from' is reserved keyword in Python
            # First calculate difference in symbols length.
            symbols_count = len(new_value) - len(old_value)
            # Note that in case of removing link toString/fromString may be None.
            # Also to/from doesn't bring information about type of link (relates to, blocks, caused by, etc.).
            change_desc = f"{field} changed from '{'' if item.fromString is None else item.fromString}' to "\
                          f"'{'' if item.toString is None else item.toString}'."
            # Field may be truncuted by simple "press cross icon" so treat deletion as a single keystroke.
            if symbols_count < 0:
                symbols_count = 1
        elif field in {'RemoteIssueLink'}:  # Hard to find out exact link type.
            symbols_count = 1  # Listed actions may require keystrokes but are fast actions anyway.
        elif field in {'assignee', 'status', 'duedate', 'priority', 'Fix Version', 'resolution'}:
            new_value = _jira_story_item_field_to_string(item.toString)
            old_value = _jira_story_item_field_to_string(item.fromString)
            change_desc = f"{field} changed from '{old_value}' to '{new_value}'."
            symbols_count = 1  # Listed actions may require keystrokes but are fast actions anyway.
        else:
            unsupported_fields.add(field)
        # Make separate event for each change in issue - flat events are easier to handle.
        events.append(Event(JIRA_BUCKET_ID, created, None, {
            'jira_id': jira_id,
            'title': summary,
            'field': field,
            'symbols_count': symbols_count,
            'change_desc': change_desc,
        }))
    return events, unsupported_fields


def _format_jira_event_for_log(event: Event) -> str:
    data = event.data
    field = data['field']
    change_desc = f"{data['symbols_count']} changes in '{field}'"
    if field not in JIRA_TICKET_FIELDS_VARIABLE_UPDATE_COMPLEXITY:
        change_desc += ": " + data['change_desc']
    return f"{{{event.timestamp.astimezone(CURRENT_TIMEZONE):%H:%M:%S} {data['jira_id']} {change_desc}}}"


def get_events_from_jira(issues: List[jira.Issue], author_email: str, change_date: datetime.datetime) -> List[Event]:
    """
    Filters all issues by specific date and specified author. Collects events from them.
    :param issues: Jira issues to inspect in chronological order.
    :param author_email: Account email to filter activities by.
    :return: List of events based on Jira issues activites performed by specific account.
    """
    # First generate as much events as possible and without right duration.
    events: List[Event] = []
    unsupported_fields = set()
    change_date = ensure_datetime(change_date).date()
    for issue in issues:
        jira_id = issue.key
        title = issue.fields.summary if hasattr(issue.fields, 'summary') else "<cannot parse>"
        # Don't use 'raw' value(s) because Jira may rename fields.
        # Note that Jira history is provided in reversed order. They will be ordered by date later on.
        for story in issue.changelog.histories:
            # Skip changes by other people or by not signed actors.
            if not hasattr(story.author, 'emailAddress') or story.author.emailAddress != author_email:
                continue
            # Skip changes made other date.
            created = datetime.datetime.strptime(story.created, JIRA_DATETIME_FORMAT)
            if created.date() != change_date:
                continue
            story_events, story_unsupported_fields = _parse_events_from_story_without_duration(story, jira_id,
                                                                                               title, created)
            events.extend(story_events)
            unsupported_fields.update(story_unsupported_fields)
    unsupported_desc = " All fields are supported." if not unsupported_fields else\
        " During parsing handled with 'default' behavior following unknown fields from Jira issues: "\
            + str(unsupported_fields)
    LOG.info("Parsed %d events from %d issues.%s", len(events), len(issues), unsupported_desc)
    if len(events) <= 0:
        return []
    # Here events created on "per issue" basis though doesn't have durations and may intersect. Also theirs 'timestamp'
    # field contains "end of event" datetime and duration may be shorter than tolerance though they would be skipped.
    # Need to sort, set 'duration' and maybe tune 'timestamp' to make one consequitive line of "not too short" events.
    events = sorted(events, key=lambda x: x.timestamp)
    if LOG.level <= logging.DEBUG:
        LOG.debug("Having following events without durations yet:\n  %s",
                  "\n  ".join(_format_jira_event_for_log(x) for x in events))
        LOG.debug("Calculating duration and adjusting events:")
    result_events = []
    # If try to adjust "previous" only event if new one too short then it is not enough for Jira.
    # Because one Jira action may trigger few changes by one human action. For example:
    # - Moving ticket to "Done" triggers Fix Version, resolution, etc. updates in the same ticket.
    # - Making "relates to" link in one ticket to another makes mirror changes in other ticket.
    # If make 'data' of one event contain changes for few tickets it would be hard to analyse and merge later.
    # Therefore buffer any number of events pointing to the same date and next "propagate" them back in time,
    # cutting duration from "prevous" "long" event.
    pending_events = []
    # Assumption: as an start for the first event use start of the day.
    event_start: datetime.datetime = ensure_datetime(events[0].timestamp.date())
    first_pending_start: datetime.datetime = event_start
    for event in events + [None]:  # Add extra iteration at the end to handle last pending event(s).
        duration = event.timestamp - event_start if event else datetime.timedelta(milliseconds=0)
        # Check we have pending events and current one doesn't need to be postponed. Or it is the last iteration.
        if (pending_events and duration > EVENTS_COMPARE_TOLERANCE_TIMEDELTA) or event is None:
            # Free buffer using first pending event duration to accomodate all next ones.
            reversed_buffer = []
            # Iterate pending events in reversed order to get first pending event as a donor.
            oldest_in_pending = pending_events[0]
            latest_saved_timestamp = None
            for pending in reversed(pending_events):
                end = pending.timestamp  # Or earlier.
                if pending == oldest_in_pending:
                    # Add all remained duration to the first/oldest event.
                    start = first_pending_start
                    if end - start < EVENTS_COMPARE_TOLERANCE_TIMEDELTA:
                        # Case when last event in a day is too short. Extend its end to be later.
                        end = start + EVENTS_COMPARE_TOLERANCE_TIMEDELTA
                else:
                    # All events except oldest one in "pending" are with short duration - extend them to minimum.
                    if latest_saved_timestamp:
                        end = latest_saved_timestamp
                    start = end - EVENTS_COMPARE_TOLERANCE_TIMEDELTA
                # Assure that there were no miscalculations above and build full ActivityWatch event.
                assert end - start >= EVENTS_COMPARE_TOLERANCE_TIMEDELTA,\
                       f"Can't distribute Jira events duration using as a donor {_format_jira_event_for_log(pending)}."
                reversed_buffer.append(
                    Event(JIRA_BUCKET_ID, start, end - start, pending.data)
                )
                latest_saved_timestamp = start  # Use start because we are going back in time.
            result_events.extend(reversed(reversed_buffer))  # Don't forget to un-reverse.
            first_pending_start = end
            event_start = end
            pending_events = [event]  # Current event is a donor for the next "pending" events chunk.
        else:
            # In both opposite cases need to postpone event creation - it may become a donor for following ones.
            pending_events.append(event)
            event_start = event.timestamp
    return result_events


def main():
    parser = argparse.ArgumentParser(
        description="Calls JIRA API to get issues updated by given account on given date,"
                    " parses all found events in it and loads them into ActivityWatch."
                    " To see more logs use `export LOGLEVEL=DEBUG` (or `set ...` on Windows)."
    )
    parser.add_argument('search_date', nargs='?', type=valid_date, default=datetime.datetime.now().astimezone(),
                        help="Date to look for Jira events in format 'YYYY-mm-dd'. By default is today.")
    parser.add_argument('-b', '--back-days', type=int,
                        help="Overwrites 'date' if specified. Sets how many days back search events on."
                             " I.e. '1' value means 'search for yesterday'.")
    parser.add_argument('-p', '--projects', type=str, default=JIRA_PROJECTS,
                        help="Comma-separated list of Jira project ID's to scrape events from."
                             " Note that Jira API allows to get some limited number of issues at once"
                             " and it is a limit for scraping.")
    parser.add_argument('-e', '--email', type=str, default=JIRA_LOGIN_EMAIL,
                        help="Email address to login into Jira.")
    parser.add_argument('-a', '--api-token', type=str, default=JIRA_LOGIN_API_TOKEN,
                        help="Jira API token to login. For details see https://support.atlassian.com/atlassian-account"
                             "/docs/manage-api-tokens-for-your-atlassian-account/.")
    parser.add_argument('-s', '--server', type=str, default=JIRA_URL,
                        help="URL to Web (MS Office Web Apps) Outlook. Page where email box opens."
                             " May look like 'https://company.jira.net'.")
    parser.add_argument('-r', '--replace', dest='is_replace_bucket', action='store_true',
                        help=f"Flag to delete ActivityWatch '{JIRA_BUCKET_ID}' bucket first."
                             " Removes all previous events in it, for all time.")
    parser.add_argument('--dry-run', dest='is_dry_run', action='store_true',
                        help="Flag to just log events but don't upload into ActivityWatch.")
    args = parser.parse_args()
    search_date = args.search_date
    if args.back_days:
        assert args.back_days >= 0,\
            f"'back_days' value ({args.back_days}) should be positive or 0."
        search_date = (datetime.datetime.today().astimezone() - datetime.timedelta(days=args.back_days))
    projects = [str(x).strip() for x in args.projects.split(',')]  # Clean up input from extra spaces.
    # Get "touched" Jira issues list.
    issues = _get_jira_issues(args.server, args.email, args.api_token, projects, search_date)
    LOG.info("Received %d issues from Jira [%s] projects.", len(issues), args.projects)
    events = get_events_from_jira(issues, args.email, search_date)
    LOG.info("Ready to upload %d events:\n  %s", len(events), "\n  ".join(event_to_str(x) for x in events))
    if not events:
        LOG.warning("Can't find Jira activity on %s for %s account in [%s] projects.",
                    args.search_date, args.email, args.projects)
    # Load events into ActivityWatcher
    if not args.is_dry_run:
        LOG.info(upload_events(events, JIRA_SCRAPER_NAME, JIRA_BUCKET_ID, args.is_replace_bucket,
                               aw_client_name="jira.issue.activity"))


if __name__ == '__main__':
    LOG = setup_logging()
    main()
