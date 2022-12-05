#!/usr/bin/env python3
import datetime
from typing import List, Set, Tuple
import argparse
import difflib
import jira

from activity_merger.config.config import LOG, JIRA_SCRAPER_NAME, JIRA_BUCKET_ID, JIRA_URL, JIRA_LOGIN_EMAIL,\
    JIRA_LOGIN_API_TOKEN, JIRA_PROJECTS, JIRA_ISSUES_MAX, EVENTS_COMPARE_TOLERANCE_TIMEDELTA
from activity_merger.helpers.helpers import setup_logging, valid_date, ensure_datetime, upload_events
from activity_merger.domain.input_entities import Event


JIRA_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


def _get_jira_issues(server_url: str, email: str, api_token: str, projects: List[str], search_date: datetime)\
        -> List[jira.Issue]:
    assert server_url, "Jira server URL is not specified."
    assert email, "Jira login email is not specified."
    assert api_token, "Jira login API token is not specified."
    assert projects, "Jira projects to consider are not specified."
    assert search_date, "Date to search is not specified."
    connection = jira.JIRA(server=server_url, basic_auth=(email, api_token))
    end_date = search_date.date() + datetime.timedelta(days=1)
    LOG.info(f"Searching issues updated from {search_date.date()} to {end_date} for projects: {projects}.")
    issues: List[jira.Issue] = []
    for project in projects:
        project_id = project.strip()
        project_issues = connection.search_issues(
            f"project = '{project_id}' and updated >= '{search_date.date()}' and updated < '{end_date}'"
             " order by updated",
            expand="changelog", maxResults=JIRA_ISSUES_MAX
        )
        LOG.info(f"Got {len(project_issues)} issues information for '{project_id}' project.")
        issues.extend(project_issues)
    return issues


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


def _parse_event_from_story_without_duration(story: jira.resources.PropertyHolder, jira_id: str,
        change_date: datetime.datetime) -> Tuple[List[Event], Set[str]]:
    created = datetime.datetime.strptime(story.created, JIRA_DATETIME_FORMAT)
    if ensure_datetime(change_date).date() != created.date():
        return [], {}
    unsupported_fields = set()  # For debugging custom Jira projects/servers.
    events = []
    for item in story.items:
        field = item.field
        symbols_count = 1  # 1 by default for unsupported fields.
        change_desc = item.toString
        if field in {'description', 'summary', 'labels', 'Component'}:
            new_value = _jira_story_item_field_to_string(item.toString)
            old_value = _jira_story_item_field_to_string(item.fromString)
            # Calculate difference in text. Treat inputs as arrays of characters.
            change_desc, symbols_count = _calculate_diff(old_value, new_value)
            # symbols_count = len(change_desc) - 1 * len(diff)  # 3 stands for +/-, \n, space.
        elif field in {'Link'}:
            new_value = _jira_story_item_field_to_string(item.to)
            old_value = _jira_story_item_field_to_string(item.__dict__['from'])  # 'from' is reserved keyword in Python
            # First calculate difference in symbols length.
            symbols_count = len(new_value) - len(old_value)
            # Field may be truncuted by simple "press cross icon" so treat deletion as a single keystroke.
            if symbols_count < 0:
                symbols_count = 1
        elif field in {'RemoteIssueLink'}:  # Hard to find out exact link type.
            symbols_count = 1  # Listed actions may require keystrokes but are fast actions anyway.
        elif field in {'assignee', 'status', 'duedate', 'priority'}:
            new_value = _jira_story_item_field_to_string(item.toString)
            old_value = _jira_story_item_field_to_string(item.fromString)
            change_desc = f"{field} changed from '{old_value}' to '{new_value}'."
            symbols_count = 1  # Listed actions may require keystrokes but are fast actions anyway.
        else:
            unsupported_fields.add(field)
        # Make separate event for each change in issue - flat events are easier to handle.
        events.append(Event(JIRA_BUCKET_ID, created, None, {
            'jira_id': jira_id,
            'field': field,
            'change_desc': change_desc,
            'symbols_count': symbols_count,  # Some gauge of "how many effort was added" TODO
        }))
    return events, unsupported_fields


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
    for issue in issues:
        jira_id = issue.key
        # Don't use 'raw' value because Jira may rename fields.
        for story in reversed(issue.changelog.histories):  # Jira history is provided in reverse order.
            if hasattr(story.author, 'emailAddress') and story.author.emailAddress == author_email:
                story_events, story_unsupported_fields = _parse_event_from_story_without_duration(story, jira_id,
                                                                                                  change_date)
                events.extend(story_events)
                unsupported_fields.update(story_unsupported_fields)
    unsupported_desc = " All fields are supported." if not unsupported_fields else\
        " During parsing handled with 'default' behavior following unknown fields from Jira issues: "\
            + str(unsupported_fields)
    LOG.info(f"Parsed {len(events)} events from {len(issues)} issues.{unsupported_desc}")
    # Here events created on "per issue" basis though have a lot of intersections. Also theirs
    # 'timestamp' field contains "end of event" datetime and duration may be happen shorter than tolerance.
    # Need to merge them and adjust theirs 'duration' to make one consequitive line of "not too short" events.
    # Note that do this for more than 2 events is hard and may make no sense. TODO consider remove it.
    events = sorted(events, key=lambda x: x.timestamp)
    # As an start for the first event use start of the day.
    event_start: datetime.datetime = ensure_datetime(events[0].timestamp.date())
    result_events = []
    pending_event = None
    for event in events:
        duration = event.timestamp - event_start
        # Check that we need extend current event and may adjust pending event.
        if pending_event and duration <= EVENTS_COMPARE_TOLERANCE_TIMEDELTA \
                and pending_event.data['jira_id'] == event.data['jira_id']\
                and pending_event.duration > 2 * EVENTS_COMPARE_TOLERANCE_TIMEDELTA:
            # If same Jira ticket event was performed before and has enough duration to fit one more then
            # cut some duraion from previous event...
            pending_duration = pending_event.duration - EVENTS_COMPARE_TOLERANCE_TIMEDELTA
            result_events.append(Event(JIRA_BUCKET_ID, event_start, pending_duration, pending_event.data))
            # ... and extend duration for the current one.
            event_start = pending_event.timestamp + pending_duration
            pending_event = Event(JIRA_BUCKET_ID, event_start, event.timestamp - event_start, event.data)
        else:
            if pending_event:
                result_events.append(pending_event)
            pending_event = Event(JIRA_BUCKET_ID, event_start, duration, event.data)
        # Calculate start of the next event.
        event_start = pending_event.timestamp + pending_event.duration
    if pending_event:
        result_events.append(pending_event)
    return result_events


def main():
    parser = argparse.ArgumentParser(
        description="Calls JIRA API to get issues updated by given account on given date,"
                    " parses all found events in it and loads them into ActivityWatch."
    )
    parser.add_argument('-d', '--search-date', dest='search_date', type=valid_date, default=datetime.datetime.now(),
                        help="Date to look for Jira events in format 'YYYY-mm-dd'. By default is today.")
    parser.add_argument('-p', '--projects', type=str, default=JIRA_PROJECTS,
                        help="Comma-separated list of Jira project ID's to scrape events from."
                             "Note that Jira API allows to get some limited number of issues at once"
                             " and it is a limit for scraping.")
    parser.add_argument('-e', '--email', type=str, default=JIRA_LOGIN_EMAIL,
                        help="Email address to login into Jira.")
    parser.add_argument(
        '-a', '--api-token',
        type=str, default=JIRA_LOGIN_API_TOKEN,
        help="Jira API token to login. For details see "
             "https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/."
    )
    parser.add_argument('-s', '--server', type=str, default=JIRA_URL,
                        help="URL to Web (MS Office Web Apps) Outlook. Page where email box opens."
                             "May look like 'https://company.jira.net'.")
    parser.add_argument('-r', '--replace', dest='is_replace_bucket', action='store_true',
                        help=f"Flag to replace all events in ActivityWatch {JIRA_BUCKET_ID} bucket.")
    parser.add_argument('--dry-run', dest='is_dry_run', action='store_true',
                        help=f"Flag to just log events but don't upload into ActivityWatch.")
    args = parser.parse_args()
    issues = _get_jira_issues(args.server, args.email, args.api_token, args.projects.split(','), args.search_date)
    LOG.info(f"Parsed {len(issues)} issues from Jira [{args.projects}] projects.")
    events = get_events_from_jira(issues, args.email, args.search_date)
    LOG.info(f"Ready to upload {len(events)} events:" + "\n  " + "\n  ".join(str(x) for x in events))
    if not events:
        LOG.warn(f"Can't find Jira activity on {args.search_date} for {args.email} account in [{args.projects}] projects.")
    # Load events into ActivityWatcher
    if not args.is_dry_run:
        upload_events(events, JIRA_SCRAPER_NAME, "jira.issue.activity", JIRA_BUCKET_ID, args.is_replace_bucket)
        LOG.info("Uploaded all events into ActivityWatch.")


if __name__ == '__main__':
    LOG = setup_logging()
    main()
