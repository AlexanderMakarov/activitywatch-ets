# ActivityWatch extentions to assemble time reports for employer

## Idea

Different organizations require to fill up daily time reports in a way like:
- 1 hour - updated X
- 0.5 hour - meeting with Y
- 2 hours - implemented Z

And usually most part of these activities happens on the computer. Especially for IT professions.

[ActivityWatch](https://activitywatch.net/) allows to measure a lot of different activities happened on a computer(s).
For "real time" measurement see [ActivityWatch Watchers list](https://docs.activitywatch.net/en/latest/watchers.html).
For "on demand" measurements see 
[ActivityWatch Importers list](https://docs.activitywatch.net/en/latest/importers.html).

ActivityWatch has [Categorization feature](https://docs.activitywatch.net/en/latest/features/categorization.html)
which allows to configure tree of "Categories" and filter events into these categories with (September 2023)
> The regular expression is matched on the ‘app’ and ‘title’ value of events (not yet URLs).

For time reports this feature is not flexible enough because:
1. For "time reports" we need name activity in a specific way. Not from predifined categories.
    I.e. mention that we worked exactly on "X" and meeting was especially with "Y".
2. Regexp on event's ‘data’ is not enough usually. Various event watchers and importers provide
    different events with different `data` fields.
3. ActivityWatch doesn't count events happened during "AFK" periods. But in some cases is it wrong behavior -
    imagine:
    - meetings where you just talking or listening without moving mouse,
    - time spent on watching education/promotion (for work) videos,
    - cases when your coworker asked some relevant to work question and you explained it also being AFK this time.
4. Some "unclear" intervals should be skipped as "not working" but some have to be added to next "working" interval.
    For example time to open browser page (via bookmarks or Google) depends on where you are ending up.
    Or browsing files on your computer may mean either searching photos from last corporative or searching
    working report asked by your coworker.
5. Parallel events should be distinguishable. For example if you've switched to web browser and started 
    to open various sites during writing code in IDE it is most probably because you need to search some information
    required to complete code, not to switch on private task or entertainmment (are you ;) ?). Or if you've
    started some Jira item, next spent some time on "scrum" meeting, next returned to IDE and next completed Jira
    task then meeting duration hardly should be reported under mentioned Jira item.
    One more "specific" example - while you are working on automating posting data from your applicaiton to
    Reddit/Facebook/etc. - most probably you are opening these sites to debug how your feature works,
    not to check new posts in your account.
6. Data from existing ActivityWatch watchers and importers usually is not enough. More data - more precise reports
    may be generated.

So the idea is to use out-of-the-box ActivityWatch data, enhance it with data from the popular 
time-aware tools like:
- Jira Cloud events ([get_jira_events.py](/get_jira_events.py)),
- MS Exchange Calendar ([get_outlook_events.py](/get_outlook_events.py)) - aka OWA, Office 365 Mail,
- Git commits ([get_git_events.py](/get_git_events.py))
and combine total ~2500 events per day
(depends on number of watchers/importers and style of work) into 5-12 activities by the set of predefined rules.
It is impossible to generate activities with human-friendly description on the bare heuristic without using
some kind of LLM-s (aka ChatGPT) trained on the personal data (which insecure for free if use public services).
So solution is not fully automated yet and relies on the user to assemble good activities and theirs names.
But it provides solid template to do it, basing all intervals not on a memory but on a data.
It allows to reduce effort spending on filling time reports, make them more precise.
Even without the necessity to fill time reports for the employer/client
this applicaiton's allows to handle personal time management in a measurable way and, therefore, effectively.

See [get_activities.py](/get_activities.py) script, run it with `--help` for the details.

Default (i.e. configurable if need) restrictions for "assembling activities" logic are:
- Activity can't be longer than 2 hours and duration is rounded to 0.25 hour.
- Events shorter than 1 second are skipped (it doesn't affect resulting activities intervals).
- All timestamps are rounded to seconds.

Everything is configurable in [config.py](/activity_merger/config/config.py) file.

In the result, application allows to mitigate of fix weaknesses of "native ActivityWatch" approach
mentioned above in a way:
1. Name of activity is assembled from events data, not from predefined names of categories.
2. Behavior of interpreting events from the each bucket is configured separately and not just
    "category A" or "category B" but as generic activity with generic name.
3. Each bucket is configured perconaly to be related to "AFK" or not.
4. "Uncategorisable" events may be skipped or be "outweighted" by events from the other bucket.
5. Activities are assembled basing on the all bucket's events.
    Moreover - one bucket may provide different activities suggestions for the same interval.
6. At least 3 sources of data/events (Jira, Outlook calendar, Git) are added.

## Setup

- Install Python 3.6+.
- Create venv with `python -m venv .venv` (be careful with version of Python).
    Activate it with something like `source venv/bin/activate`.
- Open root folder and run `pip3 install -r requirements.txt`.
- (only for Outlook Calendar events scraping) Install [geckodriver](https://github.com/mozilla/geckodriver)
    from GitHub repo [Releases](https://github.com/mozilla/geckodriver/releases) page.
    For Linux it is just extracting binary into some place in $PATH.

## ActivityWatch extra data.

To add Jira and MS Exchange Calendar events was decided to use "on-demand importers" instead of "real time watchers" because:
- Jira Cloud addon may be added only by Jira account administrator.
    Therefore it should be quite hard for regular employee to add listener to Jira server.
    Jira provides authentication option via API token which is used by [get_jira_events.py](/get_jira_events.py).
- MS Exchange events in theory may be watched via local Outlook application.
    But in case of Linux operating system and configured MFA on the server WEB (browser based) UI is the only option.
    Note that Office365 doesn't support IMAP/POP3 connection with enabled MFA and doesn't provide some API-token-like
    authentication options.
    As an option [aw-import-ical](https://github.com/ActivityWatch/aw-import-ical) may be used instead.

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

### Git

To build and import events from your Git daily activities:
- (one time action) Set in [config.py](/activity_merger/config/config.py) at least `GIT_FOLDERS_WITH_REPOS`.
  Probably it would be necessary to correct `GIT_DEPTH_IN_FOLDER` as well.
- Run [get_git_events.py](/get_git_events.py) script with target date (run with `--help` for details).
Script will check all repos in the specified folders for the specified day and with author equal to 
author recieved from `git config user.name` command (handles different in each repo).
From each commit would be created event with the same strategy as with Jira - i.e. first event starts from
the midnight and ends at the time of commit, second commit starts from the first and so on.
But in contrast to Jira (and relevant projects) it handles all repositories separately,
i.e. time of commit from A repo is not used for B repo events.

## Configure get_activities.py

Everything is configured in [config.py](/activity_merger/config/config.py).

## Roadmap

### Must:

- [x] Dry run for importers.
- [x] Why get_outlook_events.py default "today" don't implemented? It doesn't work with back_days=0.
- [x] Fix get_outlook_events.py: removes plugins.
- [x] Fix get_outlook_events.py: skips 'free' events.
- [x] Debug: write a script to export all buckets data for a specific date.
- [x] Add debugging buckets to produce insights about rules work into ActivityWatch UI "Timeline" page.
- [x] Write tests for merger with Stopwatch events involvement.
- [x] Prepare script to run all event importers and get_activities.py for the specific date.
- [x] For `Strategy.in_group_by_keys` need an ability specify "if key doesn't exist then use this", not both. IDEA case.
- [x] Add "git exporter".
- [x] Add to Strategy 'in_skip_events_with_key_value' or 'in_key_value_skip' property to skip "app=unknown" events.
- [x] Restore 'ignore_hints' in 'get_activities.py'.
- [x] Separate "strategy activity" and "result activity".
- [x] Add ability for activities within `out_activity_boundaries` not "whole" strategies keep track "max cut from left/right".
- [ ] Add ability to change strategies in `analyze_activities_per_strategy`.
- [ ] Add "out_only_if_window_app" to strategies (similar to out_only_not_afk).
- [ ] Keep exact events in result activities.
- [ ] Complete get_activities.py
- [ ] Try it for myself. Adjust `STRATEGIES` and code if need.
- [ ] Change licence.
- [ ] Use for ETS for a few days. Adjust `STRATEGIES`.
- [ ] Support case when Stopwatch events intersect with other ones like [0<-SW->2][1<-AFK->4][3<-SW->4]
- [ ] Prepare for distribution (decide how it would look like).
- [ ] Support intersecting AFK events in analyzer.analyze_activities_per_strategy
- [ ] Importers - support parsing few days at once.
- [ ] OWA importer - adopt Chrome as well.

### Questionable:

- [ ] Add support for TODO cases in test_merger.
- [ ] Think about way to make configuration file not to be "too deep in sources".
- [ ] Separate `aw_export_one_day.py` into use standalone. The same is useful with exporters as well.
  [ ] Improve "windows to activities" (`analyzer._window_to_activity`) with parallel sliding windows.
- [ ] Wrap into the web server - send JSON with day data - get activities.
- [ ] Interactive way to merge activities.
- [ ] Support remained "combinations of "in_" settings in `handle_events` (TODO-s in strategies.py)
