#!/usr/bin/env python3
import datetime
from typing import List, Set, Tuple, Union
import argparse
import os
import subprocess
import dataclasses
import logging

from activity_merger.config.config import LOG, GIT_FOLDERS_WITH_REPOS, GIT_SCRAPER_NAME, GIT_BUCKET_ID,\
                                          GIT_DEPTH_IN_FOLDER
from activity_merger.helpers.helpers import setup_logging, valid_date, ensure_datetime, upload_events, event_to_str,\
                                            CURRENT_TIMEZONE
from activity_merger.domain.input_entities import Event


@dataclasses.dataclass
class Commit:
    repo: str
    """Path to the git repository."""
    date: datetime
    """Time when commit was created or updated."""
    files: List[str]
    """List of updated files."""
    message: str
    """Commit message."""
    total_lines_changed: int
    """Total absolute number of changed lines."""

    def to_event(self) -> Event:
        return Event(GIT_BUCKET_ID, self.date, None, {
            'files': self.files,
            'message': self.message,
            'lines_changed': self.total_lines_changed,
        })


def _run_git_command(git_dir: str, command_suffix: str) -> Union[str, None]:
    try:
        return subprocess.check_output(
            f'git --git-dir={git_dir} {command_suffix}',
            shell=True,
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        LOG.info("Failed git command: %s\n%s", e, e.stdout)
        return None


def parse_commit_info(git_dir: str, commit_hash: str) -> Commit:
    commit_output = _run_git_command(git_dir, f'show --pretty="format:Date:   %ci%n%B" --stat {commit_hash}')
    lines = commit_output.split("\n")

    # Find empty line index to know where 'stat' part of message starts.
    empty_line_index = lines.index('') if '' in lines else len(lines)
    stat_start_index = empty_line_index + 1

    return Commit(
        repo=git_dir,
        date=datetime.datetime.strptime(lines[0].replace("Date:   ", "").strip(), "%Y-%m-%d %H:%M:%S %z"),
        files=[line.strip().split('|')[0].strip() for line in lines[stat_start_index:-1] if line.strip()],
        message="\n".join(lines[1:empty_line_index]),
        total_lines_changed=sum(int(word) for word in lines[-1].split() if word.isdigit()),
    )


def commits_for_day(day: datetime.datetime, folders: List[str], depth: int) -> List[Commit]:
    commit_list = []
    day_str = day.strftime("%Y-%m-%d")  # Prepare string with the requested day.

    for folder in folders:
        folder = os.path.abspath(folder)  # Convert to absolute path.
        base_level = len(folder.split(os.path.sep))  # Calculate place from which start to count depth.
        LOG.debug("Scanning '%s' folder for GIT repositories", folder)
        for dirpath, dirs, _ in os.walk(folder, topdown=True):
            if dirpath.count(os.path.sep) - base_level >= depth:
                dirs[:] = []  # Don't dig deeper.
                continue
            LOG.debug("Checking '%s' directory for GIT repositories", dirpath)
            if ".git" in dirs:
                LOG.info("Found GIT repository in '%s'. Checking for commits...", dirpath)
                git_dir = os.path.join(dirpath, '.git')
                # First find current user in repo.
                user_name = _run_git_command(git_dir, 'config user.name')
                if user_name is None:
                    continue
                # Next get commit hashes for current user and given day.
                commit_hashes = _run_git_command(
                    git_dir,
                    f'rev-list --all --author="{user_name}" --since="{day_str} 00:00" --until="{day_str} 23:59"'
                )
                if commit_hashes is None or len(commit_hashes) < 1:
                    LOG.debug("'%s' doesn't have commits for %s", git_dir, day_str)
                    continue
                else:
                    LOG.debug("Repo contains following commit hashes by '%s': %s", user_name, commit_hashes)
                # For each commit hash, get commit details.
                for commit_hash in commit_hashes.split("\n"):
                    if commit_hash:  # ignore empty lines
                        commit = parse_commit_info(git_dir, commit_hash)
                        LOG.info("%s: %s", user_name, commit)
                        commit_list.append(commit)
    return commit_list


def main():
    parser = argparse.ArgumentParser(
        description="Scans folder for GIT repositories, checks for commits on the given day,"
                    " converts into ActivityWatch events and loads them into ActivityWatch."
                    " Note that presence of GIT repository is happens by '.git' folder."
                    " To see more logs use `export LOGLEVEL=DEBUG` (or `set ...` on Windows)."
    )
    parser.add_argument('search_date', nargs='?', type=valid_date, default=datetime.datetime.now().astimezone(),
                        help="Date to look for Jira events in format 'YYYY-mm-dd'. By default is today.")
    parser.add_argument('-b', '--back-days', type=int,
                        help="Overwrites 'date' if specified. Sets how many days back search events on."
                             " I.e. '1' value means 'search for yesterday'.")
    parser.add_argument('-f', '--folders', type=str, default=GIT_FOLDERS_WITH_REPOS,
                        help="Comma-separated list of folders to search git repos in.")
    parser.add_argument('-d', '--depth', type=int, default=GIT_DEPTH_IN_FOLDER,
                        help="Number of folders to scan for git repos starting from any in `GIT_FOLDERS_WITH_REPOS`."
                             " E.g. ig folder is 'code' then value 2 here enables to check"
                             " 'code/repo/subrepo' but not 'code/jobs/a/repo'.")
    parser.add_argument('-r', '--replace', dest='is_replace_bucket', action='store_true',
                        help=f"Flag to delete ActivityWatch '{GIT_BUCKET_ID}' bucket first."
                             " Removes all previous events in it, for all time.")
    parser.add_argument('--dry-run', dest='is_dry_run', action='store_true',
                        help="Flag to just log events but don't upload into ActivityWatch.")
    args = parser.parse_args()
    search_date = args.search_date
    if args.back_days:
        assert args.back_days >= 0,\
            f"'back_days' value ({args.back_days}) should be positive or 0."
        search_date = (datetime.datetime.today().astimezone() - datetime.timedelta(days=args.back_days))
    assert args.depth > 0, f"'depth' value ({args.depth}) should be positive value"
    folders = [str(x).strip() for x in args.folders.split(',')]  # Clean up input from extra spaces.
    # Get "touched" Jira issues list.
    commits = commits_for_day(search_date, folders, args.depth)
    LOG.info("In %d folder found %d git repositories with %d commits for %s.", len(folders), 100500,
             len(commits), search_date)
    # events = get_events_from_jira(issues, args.email, search_date)
    # LOG.info("Ready to upload %d events:\n  %s", len(events), "\n  ".join(event_to_str(x) for x in events))
    # if not events:
    #     LOG.warning("Can't find Jira activity on %s for %s account in [%s] projects.",
    #                 args.search_date, args.email, args.projects)
    # # Load events into ActivityWatcher
    # if not args.is_dry_run:
    #     LOG.info(upload_events(events, JIRA_SCRAPER_NAME, JIRA_BUCKET_ID, args.is_replace_bucket,
    #                            aw_client_name="jira.issue.activity"))


if __name__ == '__main__':
    LOG = setup_logging()
    main()
