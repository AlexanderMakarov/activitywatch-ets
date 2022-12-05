import datetime
import logging
import socket
from ..domain.input_entities import EventKeyHandler, Rule


# ---------- PERSONAL SETTINGS ---------- Certanly of very probable to change settings.
# How short may be activity.
MIN_DURATION_SEC = 15 * 60  # 0.25 hours
# How long activity may be.
TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS = datetime.timedelta(2, 0, 0).seconds  # 2 hours
# List of rules describing "watcher" event activity and priority if different "watcher" events happened simultaneously.
# Keys matche bucket names start. If there will be few buckets with ID starting from key then all will be handled.
# If few keys match the same bucket then only first `EventKeyHandler` will be applied to the bucket events.
# Next see docstrings for `EventKeyHandler` and `Rule` classes.
RULES = {
    # Passive watcher, always provides value, even if user AFK. But "change value" event 100% shows activity.
    # data={app: str, title: str}.
    "aw-watcher-window": [
        EventKeyHandler("app", [
            Rule("zoom", 900, to_string=lambda _: None, subhandler=EventKeyHandler("title", [
                Rule("Zoom Meeting", 900, to_string=lambda _: "Zoom Meeting"),
                Rule(".*", 200, merge_next=True)
            ])),
            Rule("Slack", 890, to_string=lambda _: None, subhandler=EventKeyHandler("title", [
                Rule("Slack \| (.*?) \|.*", 889, to_string=lambda x: f"Slack {x.group(1)}"),
                Rule("(.*) screen share", 890, to_string=lambda x: f"Slack {x.group(1)}"),
            ])),
            # Skype doesn't provide info that it is a meeting.
            Rule("Skype", 880),
            Rule("unknown", 2, merge_next=True),  # Means that OS windows manager was unable to gather data.
            Rule("flameshot", 520),  # Screenshot tool.
            Rule("jetbrains-idea", 40),  # IDE. TODO ~2 seconds after "afk" watcher events -> are not counted!
            Rule("Double Commander", 35),  # File manager.
            Rule("smplayer", 36),  # Video player.
            Rule("FeatherPad", 37),  # Text editor.
            Rule("discord", 38),
        ]),
    ],
    # Passive watcher, always provides value, even if user AFK.
    # But "change value" event most probably shows activity (excluding web pages which change title periodically).
    # data={url: str, title: str, audible: bool, incognito: bool, tabCount: int}.
    "aw-watcher-web": [
        EventKeyHandler("url", [
            Rule("https://(vimbox|student)\.skyeng\.ru/.*", 501, skip=True), # English lesson, may look like AFK.
            Rule("https://gitlab\.akvelon\.net:9443/.*", 41, to_string=lambda _: "Akvelon GitLab"),
            Rule("https://akvelon\.atlassian\.net/wiki/.*", 42, to_string=lambda _: "Akvelon Wiki"),
            Rule("https://gitlab\.intapp\.com/.*", 43, to_string=lambda _: "Intapp GitLab"),
            Rule("https://wiki\.intapp\.com/wiki/.*", 44, to_string=lambda _: "Intapp Wiki"),
            Rule("https://mail\.akvelon\.com/.*", 45, to_string=lambda _: "Akvelon Mail"),
            Rule("https://intapp\.atlassian\.net/browse/.*", 46, to_string=lambda _: "Intapp Jira"),
            Rule("https://intapp\.zoom\.us/.*", 47, to_string=lambda _: "zoom"),
            Rule("https://www\.google\.com/.*", 48, to_string=lambda _: "www.google.com"),
            Rule("about:blank", 520, to_string=lambda _: None, subhandler=EventKeyHandler("title", [
                Rule("intapp\.atlassian\.net/browse/.*", 521, to_string=lambda _: "Intapp Jira"),
                Rule("zoom\.us/j/.*", 521, to_string=lambda _: "zoom"),
                Rule("logs\.(devops|us\.dev\.kube)\.intapp\.com.*", 401, to_string=lambda _: "Intapp Logs"),
                Rule("metrics\.(devops|us\.dev\.kube)\.intapp\.com.*", 400, to_string=lambda _: "Intapp Metrics"),
                Rule("gitlab\.akvelon\.net:9443/.*", 41, to_string=lambda _: "Akvelon GitLab"),
                Rule("New Tab", 42, merge_next=True),
                Rule("wiki\.intapp\.com/wiki/.*", 44, to_string=lambda _: "Intapp Wiki"),
                Rule("intapp\.zoom\.us/.*", 47, to_string=lambda _: "zoom"),
                # Last item as an "uncategorized site".
                Rule("(.+?)/.*", 3, to_string=lambda x: f"Firefox '{x.group(1)}'")
            ])),
            Rule("https://signin\.intapp\.com/", 49, to_string=lambda _: "Intapp SignIn"),
            Rule("https://intapp\.zendesk\.com/.*", 100, to_string=lambda _: "Intapp Zendesk"),
            Rule("https://docs\.google\.com/spreadsheets/.*", 101, to_string=lambda _: "Google Spreadsheets"),
            Rule("https://translate\.google.*", 102, to_string=lambda _: "Google Translate"),
            Rule("https://logs\.(devops|us\.dev\.kube)\.intapp\.com.*", 530, to_string=lambda _: "Intapp Logs"),
            Rule("https://metrics\.(devops|us\.dev\.kube)\.intapp\.com.*", 531, to_string=lambda _: "Intapp Metrics"),
            Rule("file:///.*", 532, to_string=lambda _: "Local file in browser"),
            # Last item as an "uncategorized site".
            Rule("https?://(.+?)/.*", 3, to_string=lambda x: f"Firefox '{x.group(1)}'")
        ]),
    ],
    # Passive watcher, always provides value, even if user AFK. But "change value" event shows activity/focus on.
    # data={file: str, projectPath: str, language: str, editor: const, editorVersion: const, eventType: const}
    # Need to handle only "switch to" intervals because watcher is strange.
    # Also keys are not stable in it, for example 'project' may be absent.
    "aw-watcher-idea": [
        EventKeyHandler("project", [
            Rule(".*", 100, to_string=lambda x: f"IDEA project '{x.group(0)}'")
        ]),
        EventKeyHandler("file", [
            Rule(".*", 100, to_string=lambda x: f"IDEA file '{x.group(0)}'")
        ])
    ],
}
# Absolute path to Firefox profile folder to grab OWA events under.
# On Linux it looks like '/home/{username}/.mozilla/firefox/{some_id}.default-release/'.
# May be found by opening 'about:profiles' in Firefox - "Root Directory" value.
FIREFOX_PROFILE_PATH: str = None
# URL to home page of Web (MS Office Web Apps) Outlook. May look like 'https://mail.company.com/owa'.
OWA_URL: str = None
# URL to Jira main/host page. May look like 'https://company.jira.net'.
JIRA_URL: str = None
# Email to log in into Jira.
JIRA_LOGIN_EMAIL: str = None
# API Token to login into Jira.
# See https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/
JIRA_LOGIN_API_TOKEN: str = None
# Comma-separated list of Jira project ID's to scrape events from. Like "OTD,EDIF".
JIRA_PROJECTS: str = None

# ---------- FINE TUNING SETTINGS ---------- Settings which a sutable for regular user.
# Tolerance to use when comparing events. Events shorter than this value are ignored.
# If duration between start and end of different events is equal or less then they are treated adjacent.
EVENTS_COMPARE_TOLERANCE_TIMEDELTA = datetime.timedelta(0, 1, 0)  # 1 sec
# Default priority of "afk" event. All events with equal or higher priority are treated as "independent"
# and may form separate activities.
AFK_RULE_PRIORITY = 500
# Default priority of "watchdog" watcher, aka maximum priority.
WATCHDOG_RULE_PRIORITY = 1000
# Timezone to show dates.
CURRENT_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo  # Use system timezone.
# Default logger. Used for cases when script is called as a library.
LOG: logging.Logger = logging.getLogger(__name__)
# ActivityWatch client name and screenshots prefix for OWA365/Web Outlook Calendar events.
OWA_SCRAPER_NAME = 'outlook_aw_events_scraper'
# ActivityWatch bucket ID for OWA365/Web Outlook Calendar events.
OWA_BUCKET_ID = f'{OWA_SCRAPER_NAME}_{socket.gethostname()}'
# Max number of "scroll back" operations on OWA365/Web Outlook Calendar page.
OWA_MAX_SCROLL_BACK = 31
# ActivityWatch client name for Jira-based events.
JIRA_SCRAPER_NAME = 'jira_aw_events_scraper'
# ActivityWatch bucket ID for Jira-based events.
JIRA_BUCKET_ID = f'{JIRA_SCRAPER_NAME}_{socket.gethostname()}'
# Number of issues to ask Jira API for.
JIRA_ISSUES_MAX = 100
