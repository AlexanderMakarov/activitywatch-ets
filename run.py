#!/usr/bin/env python3
import argparse
import multiprocessing
import subprocess
import sys
import threading

import activity_merger.config.config as config
import get_activities
from activity_merger.helpers.helpers import setup_logging, valid_date

LOG = setup_logging()


def _read_stream(stream, color):
    while True:
        line = stream.readline()
        if not line:
            break
        print(f"{color}{line.strip()}\033[0m")


def run_script(script_info: tuple[str, list[str], str]) -> int:
    """Run a script with arguments and return its exit status."""
    script_path, script_args, color = script_info
    try:
        # Use sys.executable to ensure the same Python interpreter is used.
        command = [sys.executable, script_path] + script_args
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
            # Create separate threads for reading stdout and stderr
            stdout_thread = threading.Thread(target=_read_stream, args=(proc.stdout, color))
            stderr_thread = threading.Thread(target=_read_stream, args=(proc.stderr, color))
            stdout_thread.start()
            stderr_thread.start()
            # Wait for threads to finish.
            stdout_thread.join()
            stderr_thread.join()
            # Wait for the subprocess to exit.
            proc.wait()
        return proc.returncode
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return -1  # Return a non-zero value to indicate an error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For the given date runs local importers of ActivityWatch events "
        "(which are enabled in 'config.py') and next builds activities."
        " Expected to be as simple as possible with all configuration placed in 'config.py'."
        "\nTo see debug logs need to set environment variable 'LOGLEVEL=debug',"
        " but they are mosly for ActivityWatch client communication debugging."
    )
    parser.add_argument(
        "date",
        nargs="?",
        type=valid_date,
        help="Date in format 'YYYY-mm-dd' to get data for. By-default is today."
        f" Note that day border is {config.DAY_BORDER}."
        " If don't set here then date is calculated as today-'back days'.",
    )
    parser.add_argument(
        "-b",
        "--back-days",
        type=int,
        help="How many days back run for. I.e. '1' value means 'work with yesterday's data.",
    )
    args = parser.parse_args()
    events_datetime = get_activities.calculate_events_datetime(args.date, args.back_days)

    # Prepare scripts to run in parallel (as processes - they are not async).
    tasks: list[tuple[str, list[str]], str] = []  # Script path, arguments, terminal color code.
    # As arguments provide just date, each script handles config.DAY_BORDER in its own way.
    arguments = [events_datetime.strftime('%Y-%m-%d')]
    if config.RUN_GIT_EVENTS_IMPORTER:  # Git = green.
        tasks.append(("get_git_events.py", arguments, "\033[32m"))
    if config.RUN_GOOGLE_CALENDAR_EVENTS_IMPORTER:  # Google = blue.
        tasks.append(("get_google_calendar_events.py", arguments, "\033[34m"))
    if config.RUN_JIRA_EVENTS_IMPORTER:  # Jira = red.
        tasks.append(("get_jira_events.py", arguments, "\033[31m"))
    if config.RUN_OUTLOOK_EVENTS_IMPORTER:  # Outlook = yellow.
        tasks.append(("get_outlook_events.py", arguments, "\033[33m"))
    # Run scripts in parallel processes.
    if tasks:
        with multiprocessing.Pool(len(tasks)) as pool:
            results = pool.map(run_script, tasks)
        # Check that all scripts not failed.
        is_all_passed = True
        for (script, _, _), return_code in zip(tasks, results):
            if return_code == 0:
                print(f"{script} completed successfully.")
            else:
                print(f"{script} failed with return code {return_code}.")
                is_all_passed = False
        assert is_all_passed, "At least one events importer failed. See outputs above."

    # Analyze events into activities.
    get_activities.convert_aw_events_to_activities(
        events_datetime=events_datetime,
        bi_finder_name=config.DEFAULT_BIFINDER,
        ignore_substrings=list(config.RUN_IGNORE_SUBSTRINGS),
        is_only_good_strategies_for_description=config.RUN_ONLY_GOOD_STRATEGIES_FOR_DESCRIPTIONS,
        is_import_debug_buckets=config.RUN_WITH_IMPORTING_DEBUG_BUCKETS,
    )
