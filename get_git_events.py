#!/usr/bin/env python3
import argparse
import dataclasses
import datetime
import os
import subprocess
from typing import Dict, List, Optional, Union

from sympy import N

from activity_merger.config.config import (
    DAY_BORDER,
    GIT_BUCKET_ID,
    GIT_DEPTH_IN_FOLDER,
    GIT_FOLDERS_WITH_REPOS,
    GIT_SCRAPER_NAME,
    LOG,
)
from activity_merger.domain.input_entities import Event
from activity_merger.helpers.event_helpers import event_to_str, upload_events
from activity_merger.helpers.helpers import ensure_datetime, setup_logging, valid_date


@dataclasses.dataclass
class Commit:
    date: datetime
    """Time when commit was created or updated."""
    files: List[str]
    """List of updated files."""
    message: str
    """Commit message."""
    total_lines_changed: int
    """Total absolute number of changed lines."""
    refs: List[str]
    """List of branches and tags where commit was."""


def commits_to_events(commits: Dict[str, List[Commit]], search_datetime: datetime.datetime) -> List[Event]:
    """
    Converts a list of commits aggregated per repository into a list of events.
    Assumes that commits are sorted in reverse chronological order.
    :param commits: Dictionary of commits aggregated per GIT repository.
    :param search_datetime: Day and time for which commits are collected.
    :return: List of events from the commits.
    """
    events: List[Event] = []
    for repo, commits in commits.items():
        start_time = search_datetime
        for commit in reversed(commits):  # Note that commits are in reversed time order.
            # Order of fields is important - it is how they would be displayed in ActivityWatch UI.
            event = Event(
                GIT_BUCKET_ID,
                start_time,
                (commit.date - start_time),
                {
                    "message": commit.message,
                    "lines_changed": str(commit.total_lines_changed),
                    "refs": commit.refs,
                    "repo": repo,
                    "files": commit.files,
                },
            )
            events.append(event)
            start_time = commit.date
    return events


def _run_git_command(directory: str, command_suffix: str) -> Union[str, None]:
    try:
        return (
            subprocess.check_output(
                f"git --git-dir='{directory}/.git' {command_suffix}", shell=True, stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as err:
        LOG.error("Failed git command: %s\n%s", err, err.stdout)
        return None


def _parse_commit_info(directory: str, commit_hash: str) -> Optional[Commit]:
    git_show_output = _run_git_command(directory, f'show --pretty="format:Date:   %ci%n%B" --stat {commit_hash}')
    branch_contains = _run_git_command(directory, f'branch --contains {commit_hash}')
    if git_show_output and branch_contains:
        show_lines = git_show_output.split("\n")
        refs = [x.replace('*', '').strip() for x in branch_contains.split('\n') if x.strip()]
    else:
        LOG.error("Can't parse %s commit data from '%s' repository!", commit_hash, directory)
        return None

    # Find empty line index to know where 'stat' part of message starts.
    empty_line_index = show_lines.index("") if "" in show_lines else len(show_lines)
    stat_start_index = empty_line_index + 1

    return Commit(
        date=datetime.datetime.strptime(show_lines[0].replace("Date:   ", "").strip(), "%Y-%m-%d %H:%M:%S %z"),
        files=[line.strip().split("|")[0].strip() for line in show_lines[stat_start_index:-1] if line.strip()],
        message="\n".join(show_lines[1:empty_line_index]),
        total_lines_changed=sum(int(word) for word in show_lines[-1].split() if word.isdigit()),
        refs=refs,
    )


def commits_for_one_day(start_datetime: datetime.datetime, folders: List[str], depth: int) -> Dict[str, List[Commit]]:
    """
    Returns a list of commits from author=current.use, for the given day and aggregated by repository.
    :param start_datetime: Date and time to start search commits from.
    :param folders: List of folders to search for commits with the given depth.
    :param depth: Number of subfolders to search for commits in each given folder.
    :return: List of commits aggregated by repository. Commits are sorted in reverse time order.
    """
    # Prepare string with the requested interval.
    since_str = start_datetime.strftime("%Y-%m-%d %H:%M")
    until_str = (start_datetime + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    result: Dict[str, List[Commit]] = {}
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
                # First find current user in repo.
                user_name = _run_git_command(dirpath, "config user.name")
                if user_name is None:
                    continue
                # Next get commit hashes for current user and given day.
                commit_hashes = _run_git_command(
                    dirpath,
                    f'rev-list --all --author="{user_name}" --since="{since_str}" --until="{until_str}"',
                )
                if commit_hashes is None or len(commit_hashes) < 1:
                    LOG.debug("'%s' doesn't have %s's commits for %s...%s", dirpath, user_name, since_str, until_str)
                    continue
                else:
                    LOG.debug("Repo contains following commit hashes by '%s': %s", user_name, commit_hashes)
                # For each commit hash, get commit details.
                commit_list = []
                for commit_hash in commit_hashes.split("\n"):
                    if commit_hash:  # ignore empty lines
                        commit = _parse_commit_info(dirpath, commit_hash)
                        if commit:
                            LOG.info("%s: %s", user_name, commit)
                            commit_list.append(commit)
                result[dirpath] = commit_list
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Scans folder for GIT repositories, checks for commits on the given day,"
        " converts into ActivityWatch events and loads them into ActivityWatch."
        " Note that presence of GIT repository is happens by '.git' folder."
        " To see more logs use `export LOGLEVEL=DEBUG` (or `set ...` on Windows)."
    )
    parser.add_argument(
        "search_date",
        nargs="?",
        type=valid_date,
        default=datetime.datetime.now().astimezone(),
        help="Date to look for Jira events in format 'YYYY-mm-dd'. By default is today."
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
        "-f",
        "--folders",
        type=str,
        default=GIT_FOLDERS_WITH_REPOS,
        help="Comma-separated list of folders to search git repos in.",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=GIT_DEPTH_IN_FOLDER,
        help="Number of folders to scan for git repos starting from any in `GIT_FOLDERS_WITH_REPOS`."
        " E.g. ig folder is 'code' then value 2 here enables to check"
        " 'code/repo/subrepo' but not 'code/jobs/a/repo'.",
    )
    parser.add_argument(
        "-r",
        "--replace",
        dest="is_replace_bucket",
        action="store_true",
        help=f"Flag to delete ActivityWatch '{GIT_BUCKET_ID}' bucket first."
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
    assert args.depth > 0, f"'depth' value ({args.depth}) should be positive value"
    folders = [str(x).strip() for x in args.folders.split(",")]  # Clean up input from extra spaces.
    # Find all GIT repositories and parse `Commit` objects from them.
    commits = commits_for_one_day(search_datetime, folders, args.depth)
    commits_sum = sum(len(x) for x in commits.values())
    LOG.info(
        "In %d folder found %d git repositories with %d commits for %s.",
        len(folders),
        len(commits),
        commits_sum,
        search_datetime,
    )
    events = commits_to_events(commits, search_datetime)
    LOG.info("Ready to upload %d events:\n  %s", len(events), "\n  ".join(event_to_str(x) for x in events))
    if not events:
        LOG.warning("Can't find GIT activity on %s in [%s] folders.", search_datetime, folders)
    # Load events into ActivityWatcher
    elif not args.is_dry_run:
        LOG.info(
            upload_events(
                events, GIT_SCRAPER_NAME, GIT_BUCKET_ID, args.is_replace_bucket, aw_client_name="git.commit.activity"
            )
        )


if __name__ == "__main__":
    LOG = setup_logging()
    main()
