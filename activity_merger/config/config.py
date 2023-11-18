import datetime
import logging
import socket
from typing import Dict, List, Tuple
from functools import partial

from activity_merger.domain.input_entities import Strategy
# from activity_merger.domain.strategies import ActivityByStrategy


# ---------- PERSONAL SETTINGS ---------- Certanly of very probable to change settings.
# How short may activity be in seconds. Strictness of this setting depends on the current logic.
MIN_ACTIVITY_DURATION_SEC = 15 * 60  # 0.25 hours
# How long may activity be in seconds. Strictness of this setting depends on the current logic.
MAX_ACTIVITY_DURATION_SEC = 2 * 60 * 60  # 2 hours
# Comma-separated list of folders to search git repos in.
GIT_FOLDERS_WITH_REPOS = "/home/user/code"
# Absolute path to Firefox profile folder to grab OWA events under.
# On Linux it looks like '/home/{username}/.mozilla/firefox/{some_id}.default-release/'.
# May be found by opening 'about:profiles' in Firefox - "Root Directory" value.
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
BAFinder_LogisticRegression_coef = [ 5.89142423e-01, 5.27761326e+00, 8.11248910e-01, -2.53023071e+00,
   2.91461678e+00,  1.76025011e-01,  4.93918463e-01, 7.49051148e-01,
  -4.57952598e-04,  0.00000000e+00,  7.44482549e-01, -7.44458994e-01,
  -7.44916947e-01]
BAFinder_LogisticRegression_intercept = -7.45467602


# ---------- COMMON SETTINGS ---------- Settings with values suitable for most people.
# def __window_activity_name_sentence_builder(activities: List[ActivityByStrategy]) -> str:
#     """
#     Activity name sentence builder for Windows Manager events.
#     """
#     # Group activities by 'app'.
#     grouped_activities = {}
#     for activity in activities:
#         app_data = activity.grouping_data.get_data()
#         app = app_data.get('app')
#         grouped_activities.setdefault(app, []).append(app_data)
    
#     # Sort groups by the sum of duration*density within each group.
#     sorted_groups = sorted(grouped_activities.items(), key=lambda x: sum(a.duration() * a.density for a in x[1]), reverse=True)

#     # Build description for each group.
#     descriptions = []
#     for app, activities in sorted_groups:
#         description = f"in '{app}'"
#         titles = [activity.grouping_data.get_data().get('title', '?') for activity in activities]
#         if any(title != '?' for title in titles):  # Include titles only if at least one is not '?'
#             description += " on '" + "', '".join(filter(lambda t: t != '?', titles)) + "'"
#         descriptions.append(description)

#     # Combine all descriptions
#     return f"Worked {', '.join(descriptions)}."


def __jira_activity_name_sentence_builder(data: Dict[str, str]) -> str:
    """
    Activity name sentence builder for Jira events.
    """
    # Expecting either only 'jira_id' key or 'jira_id'+'field' keys.
    if len(data) == 1:
        return f"{data[0][1]}:"  # Just set Jira ID(s) as prefix.
    else:
        as_dict = dict(data)  # Should have 'jira_id' and 'field' keys.
        return f"{as_dict['jira_id']}: updated {as_dict['field']} field(s)."


def __outlook_activity_name_sentence_builder(data: Dict[str, str]) -> str:
    """
    Activity name sentence builder for Outlook events.
    """
    # Should have 'name', 'location', 'type', 'sender' keys.
    # And it should have by 1 value for the each key, probably empty.
    # Ignore "type" value because it is often wrong. "location" is not informative as well.
    name = data.get("name")
    sender = data.get("sender")
    sentence: str = "Meeting "
    if name:
        sentence = f"Meeting '{name}'"
    if sender:
        return f"{sentence} organised by {sender}."
    return f"{sentence}."


def __web_browser_activity_name_sentence_builder(data: Dict[str, str]) -> str:
    """
    Activity name sentence builder for Web Browser events.
    """
    # Should have 'url', 'title', 'audible' and some other not informative keys.
    titles = data.get("title")
    sentence: str = "Browsed "
    if titles:
        return f"{sentence} '{titles}' page(s)."
    else:
        return f"{sentence} {data.get('url')} page(s)."


def __ide_activity_name_sentence_builder(ide_name, data: Dict[str, str]) -> str:
    """
    Activity name sentence builder for IDEA or VSCode/Codium events.
    """
    # Should have 'project', 'file' fields.
    projects = data.get("project")
    files = data.get("file")
    if files and len(files) > 30:  # If got too long path to file then use only file name, not full path.
        files = files.split("/")[-1]
    sentence: str = "Worked "
    if projects:
        if files:
            sentence = f"Worked on '{projects}' project(s) and {files} file(s)"
        else:
            sentence = f"Worked on '{projects}' project(s)"
    elif files:
        sentence = f"Worked on {files} file(s)"
    return f"{sentence} in {ide_name}."


# Strategies to aggregate each watcher/exporter events into activities.
# Order of strategies means order of handling and fulfillment of inter-strategies dependencies.
# Therefore first should be AFK, next OS Windows Manager.
# For settings see docs for `Strategy` class. Also first AFK strategy below is provided with all settings.
STRATEGIES = [
    Strategy(
        name="AFK",
        bucket_prefix="aw-watcher-afk",
        in_each_event_is_activity = False,
        in_trustable_boundaries="strict",
        in_events_density_matters = False,
        in_activities_may_overlap = False,
        in_skip_key_regexp = None,
        in_only_key_regexp = None,
        in_may_be_offline = False,
        in_only_not_afk = False,
        in_only_if_window_app = None,
        in_group_by_keys = None,
        out_self_sufficient = False,
        out_produces_good_activity_name = False,
        out_activity_name_sentence_builder=lambda _: None,  # Don't contribute to activity name.
    ),
    Strategy(
        name="WindowsManager->Slack Huddle",
        bucket_prefix="aw-watcher-window",
        in_trustable_boundaries="strict",
        in_events_density_matters=True,
        in_only_key_regexp = {"title": "Slack - (.+?) - Huddle"},
        in_group_by_keys=[("title",)],
        out_self_sufficient=False,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=lambda x: f"On Slack Huddle with {x['title']}.",
    ),
    Strategy(
        name="WindowsManager->Zoom meeting",
        bucket_prefix="aw-watcher-window",
        in_trustable_boundaries="strict",
        in_events_density_matters=True,
        in_only_key_regexp = {"app": "zoom", "title": "Zoom Meeting|Meeting Chat|zoom|Zoom"},
        in_group_by_keys=[("app", "title",)],
        out_self_sufficient=False,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=lambda _: "On Zoom Meeting.",
    ),
    Strategy(
        name="WindowsManager->Other",
        bucket_prefix="aw-watcher-window",
        in_trustable_boundaries="strict",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        # "unknown" events are useless.
        # "Slack - (.+?) - Huddle" is used by strategy above.
        # "Zoom Meeting|Meeting Chat|zoom|Zoom" is used by strategy above.
        in_skip_key_regexp={"app": "unknown", "title": "Slack - (.+?) - Huddle|Zoom Meeting|Meeting Chat|zoom|Zoom"},
        in_only_not_afk=True,
        # Title may change very often, but better to keep track of apps as well.
        in_group_by_keys=[("app",), ("app", "title",)],
        out_activity_name_sentence_builder=lambda x: f"Worked with {x['app']} application(s) on [{x['title']}].",  # TODO __window_activity_name_sentence_builder
    ),
    Strategy(
        name="Watchdog",
        bucket_prefix="aw-watcher-watchdog",
        in_each_event_is_activity=True,
        in_trustable_boundaries="strict",
        in_may_be_offline = True,
        out_self_sufficient=True,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=lambda x: x['data'],  # The only key is possible in 'data'.
    ),
    Strategy(
        name="Outlook",
        bucket_prefix="outlook_aw_events_scraper",
        in_each_event_is_activity=True,
        in_trustable_boundaries="dim",  # Sometimes it may start later or finish earlier. Plus preparation before.
        out_self_sufficient=True,
        out_produces_good_activity_name=True,
        out_activity_name_sentence_builder=__outlook_activity_name_sentence_builder,
    ),
    Strategy(
        name="Jira",
        bucket_prefix="jira_aw_events_scraper",
        in_trustable_boundaries="end",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_only_if_window_app=None,  # Jira activity may happen everywhere, not only in browser.
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
        name="WebBrowser",
        bucket_prefix="aw-watcher-web",
        in_trustable_boundaries="start",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_only_if_window_app=["firefox"],
        out_activity_name_sentence_builder=__web_browser_activity_name_sentence_builder,
    ),
    Strategy(
        name="IDEA",
        bucket_prefix="aw-watcher-idea",
        # Boundaries are "start" actually, but IDEA watcher may produce events if app in the background
        # so if keep "start" then "in_only_if_window_app" filter will loose interval completely.
        in_trustable_boundaries="dim",
        in_events_density_matters=True,
        in_activities_may_overlap=True,
        in_only_not_afk=True,
        in_only_if_window_app=["jetbrains-idea"],
        in_group_by_keys=[
            ("project",),
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
        out_activity_name_sentence_builder=lambda x: f"Committed into {x['repo']} repository(ies).",
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
