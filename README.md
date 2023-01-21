# ActivityWatch extentions to fill up ETS

## Idea

[ActivityWatch](https://activitywatch.net/) allows to measure a lot of different activities happened on a computer(s).
For "real time" measurement see [ActivityWatch Watchers list](https://docs.activitywatch.net/en/latest/watchers.html).
For "on demand" measurements see [ActivityWatch Importers list](https://docs.activitywatch.net/en/latest/importers.html).
But data from these sources is not enough to report your working hours to the employeer/clients or directly to yourself to adjust time management.

So the idea is to use out-of-the-box ActivityWatch data:
- AFK status from OS,
- active window from OS,
- browser events,
- IDEA (Jetbrains IDE) events
enhance it with data from working environment - Jira Cloud and MS Exchange Calendar (aka OWA, Office 365 Mail)
and somehow combine total ~2500 events per day into 10-20 activities by some predefined rules.
List of gneral rules for such adjustments:
- Activity can't be longer than 2 hours and duration is rounded to 0.25 hour.
- "Working" activities are separated from "not working" happening on the same computer.
- Activities shorter than 1 second are skipped.
- All timestamps are rounded to seconds.
Other details and more fine-grained rules may be checked in [config.py](/activity_merger/config/config.py) file.

## Setup

- Install Python 3.6+.
- Open root folder and run `pip3 install -r requirements.txt`.
- (only for Outlook Calendar events scraping) Install [geckodriver](https://github.com/mozilla/geckodriver) from GitHub repo [Releases](https://github.com/mozilla/geckodriver/releases) page. For Linux it is just extracting binary into some place in PATH.

## ActivityWatch extra data.

To add Jira and MS Exchange Calendar events was decided to use "on-demand importers" instead of "real time watchers" because:
- Jira Cloud addon may be added only by Jira account administrator. Therefore it should be quite hard for regular employee to add listener to Jira server. Jira provides authentication option via API token for which they have great respect.
- MS Exchange events in theory may be watched via local Outlook application. But in case of Linux operating system and configured MFA on the server WEB (browser based) UI is the only option. Because MS Exchange doesn't support IMAP/POP3 connection with MFA and even doesn't provide some API token like authentication option.

### Jira

To import Jira events into local (listening on localhost:5600) ActivityWatch server:
- (one time action) Put into [config.py](/activity_merger/config/config.py) at least following data:
  - `JIRA_URL`,
  - `JIRA_LOGIN_EMAIL`,
  - `JIRA_LOGIN_API_TOKEN`,
  - `JIRA_PROJECTS`;
- Run [get_jira_events.py](/get_jira_events.py) script with target date (run with `--help` for details),
After a few second script will grab all you actions in specified Jira projects Jira API for specific date and import into ActivityWatch.

### MS Exchange Calendar

To import MS Exchange Calendar events into local (listening on localhost:5600) ActivityWatch server:
- Open Firefox window under existing profile (i.e. not a private window) and authenticate into your Web Outlook,
- (one time action) Put into [config.py](/activity_merger/config/config.py) at least following data:
  - `FIREFOX_PROFILE_PATH`,
  - `OWA_URL`;
- Run [get_outlook_events.py](/get_outlook_events.py) script with target date (run with `--help` for details).
Script will open new Firefox window with the only tab of Web Outlook which will be authenticated automatically from cookies,
open Calendar with the specified date, grab from the screen bars of events, parse time and description from them
and send resulting data into ActivityWatch.

## Merging all ActivityWatch events into few activities

First of all need to adjust merger options for your needs.
I.e. configure `MIN_DURATION_SEC` and `TOO_LONG_ACTIVITY_ALERT_AFTER_SECONDS` in [config.py](/activity_merger/config/config.py) file.
Next please investigate example of `RULES` structure in the same file. Investigate classes which builds this structure.
Get concept of priority numbers in it - they are very important because will be considered on any event intersections/overlapping.

To setup `RULES` structure it is better to choose some date where expected activities are already found, Jira and MS Exchange Calendar data is imported
and build `RULES` iterating following actions:
- Run [activity_merger.py](/activity_merger.py) for needed date (run with `--help` for details).
- (on few first iterations) Investigate all logs from the start to understand which events were considered, which skipped at all or evicted by priority.
- Check resulting activities. On first iterations they most probably will be useless but after some corrections in `RULES` they will provide more sense.
- Adjust `RULES` with your findings. At start there will be "gotchas" about type of events in your environment and data inside them. Next your will find how to combine events as the same activity, how to separate similar by data events into absolutely different activities, behavior of your environment in general. The last thing would be fine tuning of priorities to decide how to treat simultaneous active window state change and IDEA state, why even during AFK state your need create activities and how to shape all low-level events into more high-level Jira and Calendar intervals.

Note that `RULES` working perfectly for one day may not work for the next day.
So be prepared for few days of initial setup and some corrections in case of some changes in your working environment.
For example after some local software updates "active window" events data may change, WEB browser data may change after updates on relevant sites, your may start to use new tool and so on.

Note that for `RULES` configuration it would be great to have:
- Not only text UI.
- Advanced UI with ability to inspect:
  - skipped events,
  - zoom into each resulting activity events,
  - change priority with simple dragging of events
- Some AI which would understand "expected" activities and automate `RULES` adjustements.
- Some AI which would understand "expected" activities and keep `RULES` inside (Deep Neural Network, Decision Tree, Random Forest types working on "normilized" `Interval`-s).

## Roadmap

- [x] Dry run for importers.
- [x] Why get_outlook_events.py default "today" don't implemented? It doesn't work with back_days=0.
- [x] Fix get_outlook_events.py: removes plugins.
- [x] Fix get_outlook_events.py: skips 'free' events.
- [ ] Debug: write a script to export all buckets data for a specific date.
- [ ] Complete get_activities.py
- [ ] Prepare script to run all event importers and get_activities.py for the specific date.
- [ ] Try it for myself. Adjust `RULES` and code if need.
- [ ] Use for ETS for a few days. Adjust `RULES`.
- [ ] Remove company-specific data from `RULES`.
- [ ] Prepare for distribution.
- [ ] Importers - support parsing few days at once.
- [ ] OWA importer - adopt Chrome as well.
