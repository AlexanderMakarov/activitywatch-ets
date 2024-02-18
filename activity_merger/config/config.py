import datetime
import logging
import socket
from functools import partial
from typing import Dict, List

from activity_merger.domain.input_entities import Strategy


# ---------- PERSONAL SETTINGS ---------- Certanly of very probable to change settings.
# How short may activity be in seconds. Strictness of this setting depends on the current logic.
MIN_ACTIVITY_DURATION_SEC = 15 * 60  # 0.25 hours
# How long may activity be in seconds. Strictness of this setting depends on the current logic.
MAX_ACTIVITY_DURATION_SEC = 2 * 60 * 60  # 2 hours
# Comma-separated list of folders to search git repos in.
GIT_FOLDERS_WITH_REPOS = "/home/user/code"
# Absolute path to Firefox profile folder to grab OWA events under.
# On Linux it looks like '/home/{username}/.mozilla/firefox/{some_id}.default-release/'.
# Open 'about:profiles' in Firefox - "Root Directory" value for section with "Default Profile = yes".
FIREFOX_PROFILE_PATH: str = "/home/user/snap/firefox/common/.mozilla/firefox/ooooooooooo.default-0000000000000"
# URL to home page of Web (MS Office Web Apps) Outlook. May look like 'https://mail.company.com/owa'.
OWA_URL: str = "https://mail.company.com/owa"
# URL to Jira main/host page. May look like 'https://company.jira.net'.
JIRA_URL: str = "https://company.atlassian.net"
# Email to log in into Jira.
JIRA_LOGIN_EMAIL: str = "firstname.lastname@company.com"
# API Token to login into Jira.
# See https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/
JIRA_LOGIN_API_TOKEN: str = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
# Comma-separated list of Jira project ID's to scrape events from. Like "OTD,EDIF".
JIRA_PROJECTS: str = "OTD,EDIF"
# Number of folders to scan for git repos starting from any in `GIT_FOLDERS_WITH_REPOS`.
# If GIT_FOLDERS_WITH_REPOS=code then value 2 here enables to check "code/repo/subrepo" but not "code/one/two/repo".
GIT_DEPTH_IN_FOLDER = 2
# Google Calendar ID to get events from. Often equal to email address or user name.
# Go to Calendar settings and scroll to "Integrate calendar" section, here would be "Calendar ID" field.
GOOGLE_CALENDAR_ID = 'email@gmail.com'
# Default BIFinder to use.
DEFAULT_BIFINDER = "proximity-duration"
# 4 values for "simple" basic interval finder. Sum of them should be equal to 1.
BIFINDER_SIMPLE_START_POINT_PROXIMITY = 0.5
BIFINDER_SIMPLE_DENSITY = 0.1
BIFINDER_SIMPLE_DURATION_ON_INTERSECTION_INTERVAL = 0.2
BIFINDER_SIMPLE_DURATION_BETWEEN_MIN_AND_MAX = 0.2
# 1 valut for "proximity-duration" basic interval finder.
BIFINDER_PROXIMITY_DURATION_MAX_REWARDED_START_POINT_PROXIMITY_SEC = 120
# 2 values for "LogisticRegression" basic interval finder. Better to setup with 'tune_rules.py'.
BIFINDER_LOGISTIC_REGRESSION_COEF = [
    5.89142423e-01,
    5.27761326e00,
    8.11248910e-01,
    -2.53023071e00,
    2.91461678e00,
    1.76025011e-01,
    4.93918463e-01,
    7.49051148e-01,
    -4.57952598e-04,
    0.00000000e00,
    7.44482549e-01,
    -7.44458994e-01,
    -7.44916947e-01,
]
BIFINDER_LOGISTIC_REGRESSION_INTERCEPT = -7.45467602


# ---------- COMMON SETTINGS ---------- Settings with values suitable for most people.
def __window_activity_name_sentence_builder(groups_data: List[Dict[str, str]]) -> str:
    """
    Activity name sentence builder for WindowsManager events.
    """
    # Build description for each group.
    app_groups = {}
    for data in groups_data:
        # Expecting either only 'app' key or 'app'+'title' keys.
        app = data.get("app")
        title = data.get("title", "").strip()
        if title:
            if len(title) > 30:
                title = title[:27] + "..."
            app_groups.setdefault(app, []).append(f"'{title}'")
        else:
            app_groups.setdefault(app, [])
    # Build description for each app group.
    descriptions = []
    for app, titles in app_groups.items():
        if titles:
            descriptions.append(f"in {app} on {', '.join(titles)}")
        else:
            descriptions.append(f"in {app}")
    # Combine all descriptions.
    return f"Worked {', '.join(descriptions)}."


def __jira_activity_name_sentence_builder(groups_data: List[Dict[str, str]]) -> str:
    """
    Activity name sentence builder for Jira events.
    """
    # Group entries by "jira_id"
    jira_activities = {}
    for data in groups_data:
        # Expecting either only 'jira_id' key or 'jira_id'+'field' keys.
        jira_id = data["jira_id"]
        field = data.get("field")
        jira_activities.setdefault(jira_id, set()).add(field)
    # Build descriptions for each Jira ID separately.
    descriptions = []
    for jira_id, fields in jira_activities.items():
        fields.discard(None)  # Remove None entries, if any
        if fields:
            fields_list = ", ".join(sorted(fields))
            field_word = "fields" if len(fields) > 1 else "field"
            descriptions.append(f"{jira_id}: updated {fields_list} {field_word}.")
        else:
            descriptions.append(f"{jira_id}: worked on.")
    # Join descripitons for all Jira ID-s.
    return " ".join(descriptions)


def __outlook_activity_name_sentence_builder(groups_data: List[Dict[str, str]]) -> str:
    """
    Activity name sentence builder for Outlook events.
    """
    # Only 1 event. Should has 'name', 'location', 'type', 'sender' keys with probably empty values.
    # Ignore "type" value because it is often wrong. "location" is not informative as well.
    data = groups_data[0]
    name = data.get("name")
    sender = data.get("sender")
    sentence: str = "Meeting "
    if name:
        sentence = f"Meeting '{name}'"
    if sender:
        return f"{sentence} organised by {sender}."
    return f"{sentence}."


def __web_browser_activity_name_sentence_builder(groups_data: List[Dict[str, str]]) -> str:
    """
    Activity name sentence builder for Web Browser events.
    """
    # Extract page name for each group.
    page_names = []
    for data in groups_data:
        # Expecting 'url', 'title', 'audible' and some other not informative keys.
        title = data.get("title")
        if title:
            page_names.append(f"'{title}'")
        else:
            page_names.append(data.get("url"))
    return f"Browsed {', '.join(page_names)} page(s)."


def __ide_activity_name_sentence_builder(ide_name: str, groups_data: List[Dict[str, str]]) -> str:
    """
    Activity name sentence builder for IDEA or VSCode/Codium events.
    """
    # Should has 'project', 'file' fields.
    # Group projects and files.
    project_descriptions = set()
    file_descriptions = set()
    for group in groups_data:
        project = group.get("project")
        file = group.get("file")
        if project:
            project_descriptions.add(project)
        if file:
            # Split by "/" and take the last part if longer than 30 characters
            if len(file) > 30:
                file = file.split("/")[-1]
            file_descriptions.add(file)
    # Building the project and file descriptions
    sentence = f"Worked in {ide_name} on "
    if project_descriptions:
        project_list = ", ".join(sorted(project_descriptions))
        sentence += f"{sentence}{project_list} {'project' if len(project_descriptions) == 1 else 'projects'}"
    if file_descriptions:
        file_list = "', '".join(sorted(file_descriptions))
        sentence = f"{sentence} with '{file_list}' {'file' if len(file_descriptions) == 1 else 'files'}"
    return sentence + "."

# Strategies to aggregate each watcher/exporter events into activities.
# Order of strategies means order of handling and fulfillment of inter-strategies dependencies.
# Therefore first should be AFK, next OS Windows Manager.
# For settings see docs for `Strategy` class. Also first AFK strategy below is provided with all settings.
STRATEGIES = [
    Strategy(
        name="AFK",
        bucket_prefix="aw-watcher-afk",
        in_each_event_is_activity=False,
        in_trustable_boundaries="strict",
        in_events_density_matters=False,
        in_activities_may_overlap=False,
        in_skip_key_regexp=None,
        in_only_key_regexp=None,
        in_may_be_offline=False,
        in_only_not_afk=False,
        in_only_if_window_app=None,
        in_only_if_window_app_soft=False,
        in_group_by_keys=None,
        out_self_sufficient=False,
        out_self_sufficient_interval_rank=0,
        out_produces_good_activity_name=False,
        out_activity_name_sentence_builder=lambda _: None,  # Don't contribute to activity name.
    ),
    Strategy(
        name="WindowsManager->Slack Huddle",
        bucket_prefix="aw-watcher-window",
        in_trustable_boundaries="strict",
        in_events_density_matters=True,
        in_only_key_regexp={"title": "^Slack - (.+?) - Huddle"},
        in_group_by_keys=[("title",)],
        out_self_sufficient=True,
        out_self_sufficient_interval_rank=20,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=lambda x: f"On Slack Huddle with {', '.join(d['title'] for d in x)}.",
    ),
    Strategy(
        name="WindowsManager->Zoom meeting",
        bucket_prefix="aw-watcher-window",
        in_trustable_boundaries="strict",
        in_events_density_matters=True,
        in_only_key_regexp={"app": "^zoom$", "title": "Zoom Meeting|Meeting Chat|zoom|Zoom"},
        in_group_by_keys=[
            (
                "app",
                "title",
            )
        ],
        out_self_sufficient=True,
        out_self_sufficient_interval_rank=30,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=lambda _: "On Zoom Meeting.",
    ),
    Strategy(
        name="WindowsManager->Other",
        bucket_prefix="aw-watcher-window",
        in_trustable_boundaries="strict",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        # "unknown" (linux) and "loginwindow" (MAC OS) events are useless.
        # "Slack - (.+?) - Huddle" is used by strategy above.
        # "Zoom Meeting|Meeting Chat|zoom|Zoom" is used by strategy above.
        in_skip_key_regexp={"app": "unknown|loginwindow", "title": "Slack - (.+?) - Huddle|Zoom Meeting|Meeting Chat|zoom|Zoom"},
        in_only_not_afk=True,
        # Title may change very often, but better to keep track of apps as well.
        in_group_by_keys=[
            ("app",),
            (
                "app",
                "title",
            ),
        ],
        out_activity_name_sentence_builder=__window_activity_name_sentence_builder,
    ),
    Strategy(
        name="Watchdog",
        bucket_prefix="aw-watcher-watchdog",
        in_each_event_is_activity=True,
        in_trustable_boundaries="strict",
        in_may_be_offline=True,
        out_self_sufficient=True,
        out_self_sufficient_interval_rank=50,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=lambda x: x[0]["data"],  # The only event and key.
    ),
    Strategy(
        name="Outlook",
        bucket_prefix="outlook_aw_events_scraper",
        in_each_event_is_activity=True,
        in_trustable_boundaries="dim",  # Sometimes it may start later or finish earlier. Plus preparation before.
        out_self_sufficient=True,
        out_self_sufficient_interval_rank=10,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=__outlook_activity_name_sentence_builder,
    ),
    Strategy(
        name="Jira",
        bucket_prefix="jira_aw_events_scraper",
        in_trustable_boundaries="end",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        # Don't make Jira activites span afk gaps. JiraIdBIFinder should search Jira ID-s in window title and so on.
        in_only_not_afk=True,
        # Jira activity is not limited by browser or other places where Jira ID exists in window title.
        in_only_if_window_app=None,
        in_group_by_keys=[
            (
                "jira_id",
                "field",
            ),
            ("jira_id",),
        ],
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=__jira_activity_name_sentence_builder,
    ),
    Strategy(
        name="WebBrowser->MayBeAFK",
        bucket_prefix="aw-watcher-web",
        in_trustable_boundaries="start",
        in_events_density_matters=True,
        in_only_key_regexp={"url": "^https://(meet\.google\.com|youtube|www\.udemy\.com).*"},
        in_only_if_window_app=["firefox", "safari", "chrome"],
        out_activity_name_sentence_builder=__web_browser_activity_name_sentence_builder,
    ),
    Strategy(
        name="WebBrowser->Other",
        bucket_prefix="aw-watcher-web",
        in_trustable_boundaries="dim",
        in_events_density_matters=True,
        in_skip_key_regexp={"url": "^https://(meet\.google\.com|youtube|www\.udemy\.com).*"},
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_only_if_window_app=["firefox", "safari", "chrome"],
        out_activity_name_sentence_builder=__web_browser_activity_name_sentence_builder,
    ),
    Strategy(
        name="IDEA",  # Linux.
        bucket_prefix="aw-watcher-idea",
        in_trustable_boundaries="start",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_only_if_window_app=["jetbrains-idea"],
        # IDEA watcher may produce events if app is in the background.
        in_only_if_window_app_soft=True,
        in_group_by_keys=[
            ("project",),
            ("file",),
        ],  # IDEA events may lack 'project' field. But 'file' contians full path.
        out_activity_name_sentence_builder=partial(__ide_activity_name_sentence_builder, "IDEA"),
    ),
    Strategy(
        name="IDEA",  # Mac OS X.
        bucket_prefix="aw-watcher-intellij-idea",
        in_trustable_boundaries="start",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_only_if_window_app=["IntelliJ IDEA"],
        # IDEA watcher may produce events if app is in the background.
        in_only_if_window_app_soft=True,
        in_group_by_keys=[
            ("project", "file"),
            ("file",),
        ],  # IDEA events may lack 'project' field. But 'file' contians full path.
        out_activity_name_sentence_builder=partial(__ide_activity_name_sentence_builder, "IDEA"),
    ),
    Strategy(
        name="VSCode",
        bucket_prefix="aw-watcher-vscode",
        in_trustable_boundaries="start",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_group_by_keys=[("project",), ("file",)],  # Events may lack 'project' field. But 'file' contians full path.
        in_only_if_window_app=["code", "vscodium"],
        # VSCode watcher may produce events if app in the background (rarely and only on Mac).
        in_only_if_window_app_soft=True,
        out_activity_name_sentence_builder=partial(__ide_activity_name_sentence_builder, "VSCode"),
    ),
    Strategy(
        name="Git",
        bucket_prefix="git_aw_events_scraper",
        in_trustable_boundaries="end",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_group_by_keys=[("repo",)],
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=lambda x: f"Committed into {', '.join(d['repo'] for d in x)} repository(ies).",
    ),
]

# ---------- FINE TUNING SETTINGS ---------- Something which may be changed only in rare cases.
# Tolerance to use when comparing events. Events shorter than this value are ignored.
# If duration between start and end of different events is equal or less then they are treated adjacent.
EVENTS_COMPARE_TOLERANCE_TIMEDELTA = datetime.timedelta(0, 1, 0)  # 1 sec
# Limit of resulting activities.
LIMIT_OF_RESULTING_ACTIVITIES = 100
# Don't convert into activity-by-strategies events spanning too small interval.
TOO_SMALL_INTERVAL_SEC = 30
# When few events are aggregated into one named interval then need to choose minimal value for it.
# For example 0.3 value means that interval should be covered by events at least on 30%.
MIN_DENSITY_FOR_SPARCE_INTERVALS = 0.3
# Name of ActivityWatch client created to import debugging buckets.
DEBUG_BUCKETS_IMPORTER_NAME = "activity_merger_debugger"
# Prefix for debug buckets. Starts with "z" to be at bottom in ActivityWatch UI.
DEBUG_BUCKET_PREFIX = "z"
# Timezone to show dates.
CURRENT_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo  # Use system timezone.
# Day time to use as "days" separator. Midnight may don't work well. Should be timedelta to add to midnight.
DAY_BORDER = datetime.timedelta(hours=5)
# Default logger. Used for cases when script is called as a library.
LOG: logging.Logger = logging.getLogger(__name__)
# ActivityWatch client name and screenshots prefix for OWA365/Web Outlook Calendar events.
OWA_SCRAPER_NAME = "outlook_aw_events_scraper"
# ActivityWatch bucket ID for OWA365/Web Outlook Calendar events.
OWA_BUCKET_ID = f"{OWA_SCRAPER_NAME}_{socket.gethostname()}"
# Max number of "scroll back" operations on OWA365/Web Outlook Calendar page.
OWA_MAX_SCROLL_BACK = 31
# ActivityWatch client name for Jira-based events.
JIRA_SCRAPER_NAME = "jira_aw_events_scraper"
# ActivityWatch bucket ID for Jira-based events.
JIRA_BUCKET_ID = f"{JIRA_SCRAPER_NAME}_{socket.gethostname()}"
# Number of issues to ask Jira API for. Note that if ask many days back then this value should be big.
JIRA_ISSUES_MAX = 100
# ActivityWatch client name for GIT-based events.
GIT_SCRAPER_NAME = "git_aw_events_scraper"
# ActivityWatch bucket ID for GIT-based events.
GIT_BUCKET_ID = f"{GIT_SCRAPER_NAME}_{socket.gethostname()}"
# Google Service Account keys file path.
GOOGLE_SERVICE_ACCOUNT_KEY_PATH = "google_api_service_account_key.json"
# ActivityWatch client name for Google Calendar events.
GOOGLE_CALENDAR_SCRAPER_NAME = "gcalendar_aw_scraper"
# ActivityWatch bucket ID for Google Calendar based events.
GOOGLE_CALENDAR_BUCKET_ID = f"{GOOGLE_CALENDAR_SCRAPER_NAME}_{socket.gethostname()}"
