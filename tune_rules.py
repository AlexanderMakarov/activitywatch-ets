#!/usr/bin/env python3
import argparse
import contextlib
import datetime
import os
import sys
from typing import Dict, List, Tuple

import aw_client
import dill  # For pickle-ing lambdas need to use 'dill' package.
import intervaltree

from activity_merger.domain.basic_activity_finder import BAFinder, IntervalFeatures
from activity_merger.domain.strategies import ActivityByStrategy

# Don't use convenient pyinput because https://pynput.readthedocs.io/en/latest/limitations.html#platform-limitations
# For Unix terminal:
try:
    import termios
except ImportError:
    pass
# For Windows terminal:
try:
    import msvcrt
except ImportError:
    pass
from pick import pick

from activity_merger.config.config import DEBUG_BUCKETS_IMPORTER_NAME, LOG
from activity_merger.domain.analyzer import (
    AnalyzerStep,
    ChopActivitiesByResultTreeStep,
    DebugBucketsHandler,
    MakeCandidatesTreeStep,
    MakeResultTreeFromSelfSufficientActivitiesStep,
    MergeCandidatesTreeIntoResultTreeWithDedicatedBAFinderStep,
    merge_activities,
)
from activity_merger.domain.metrics import Metrics
from activity_merger.domain.output_entities import AnalyzerResult
from activity_merger.helpers.helpers import datetime_to_time_str, setup_logging, upload_events, valid_date
from get_activities import get_activities_by_strategy, reload_debug_buckets


class TerminalLeader:
    """
    Abstract matching song leader which expects user answers in terminal. Wraps `Matcher`, manages "ask - answer -
    check" flow and prints statistic at the end. Delegates operating system specific actions to inheritors.
    """

    def __init__(self):
        pass

    def read_user_input(self) -> str:
        return input()

    def clean_lines(self, cnt: int):
        raise NotImplementedError()

    def ask_yes_no(self, question_without_yn: str) -> bool:
        """
        Prints question, appends ' [y/N]: ' legend to it and waits answer.
        :param question_without_yn: Question string/sentence.
        :return: `True` if user answered yes, `False` otherwise.
        """
        sys.stdout.write(question_without_yn + " [y/n]: ")
        sys.stdout.flush()
        result = self.read_user_input().lower() == "y"
        sys.stdout.write("\n")
        sys.stdout.flush()
        return result

    def ask_select_question(self, question: str, options: List[str]) -> Tuple[str, int]:
        """
        Draws multiple lines of text on the screen with options to choose one of them. Cleans console after itself.
        :param question: Question choose options for.
        :param options: List of options to choose.
        :return: Chosen option and it's index.
        """
        # 'pick' library cleans screen and draws menu with options. At end disappears and leaves old content.
        # TODO need abiity to:
        # - enter ID of activity.
        # - choose but "only until this point"
        return pick(options, question, multiselect=False, min_selection_count=1)

    def ask_multiselect_question(self, question: str, options: List[str]) -> List[Tuple[str, int]]:
        """
        Draws multiple lines of text on the screen with options to choose one or few. Cleans console after itself.
        :param question: Question choose options for.
        :param options: List of options to choose.
        :return: List of tuples with chosen option and index.
        """
        # 'pick' library cleans screen and draws menu with options. At end disappears and leaves old content.
        return pick(options, question, multiselect=True, default_index=2, min_selection_count=1)


class UnixTerminalLeader(TerminalLeader):
    """
    TerminalLeader for Unix machines. Relies on VT100 escape codes and TTY.
    """

    # FYI: https://wiki.bash-hackers.org/scripting/terminalcodes
    ANSI_CURSOR_UP = "\x1b[1A"  # Move cursor up (don't forget '\r' to put it on start of line).
    ANSI_CURSOR_SAVE_POSITION = "\033[s"  #'\x1b7'
    ANSI_CURSOR_TO_SAVED = "\033[u"  #'\x1b8'
    ANSI_CLEAR_CURRENT_LINE = "\x1b[2K"  # Clear the whole line where cursor is placed.
    ANSI_CLEAR_PREVIOUS_LINE = "\033[A"
    ANSI_CLEAR_TO_END_OF_SCREEN = "\033[J"  #'\x1b[J'
    ANSI_HIDE_CURSOR = "\033[?25l"
    ANSI_SHOW_CURSOR = "\033[?25h"
    ANSI_FONT_DECORATION_STOP = "\u001b[0m"
    ANSI_FONT_DECORATION_BOLD = "\u001b[1m"
    ANSI_FONT_DECORATION_UNDERLINE = "\u001b[4m"
    ANSI_SHIFT_CURSOR_LEFT_PREFIX = "\u001b["
    ANSI_SHIFT_CURSOR_LEFT_SUFFIX = "D"
    KEY_ESC = "\u00001b"  #  '\x1b'
    KEY_UP = "\u1b5b41"  # '\x1b[A'
    KEY_DOWN = "\u1b5b42"  #  '\x1b[B'
    KEY_LEFT = "\u1b5b44"  #  '\x1b[D'
    KEY_RIGHT = "\u1b5b43"  # '\x1b[C'

    def clean_lines(self, cnt: int):
        sys.stdout.write("".join([self.ANSI_CLEAR_PREVIOUS_LINE] * cnt))
        sys.stdout.flush()

    def ask_yes_no(self, question_without_yn: str) -> bool:
        # Also cleans question from the console.
        self._save_cursor_position()
        sys.stdout.write(question_without_yn + " [y/n]: ")
        sys.stdout.flush()
        result = input().lower() == "y"
        self._clear_up_to_saved_position()
        return result

    def _save_cursor_position(self):
        sys.stdout.write(self.ANSI_CURSOR_SAVE_POSITION)

    def _clear_up_to_saved_position(self):
        sys.stdout.write(self.ANSI_CURSOR_TO_SAVED + self.ANSI_CLEAR_TO_END_OF_SCREEN)
        sys.stdout.flush()

    def _save_cursor_position_and_print_multiselect_question(self, question: str, menu: List[List]):
        # item = [description, is_cursor, is_selected]
        self._save_cursor_position()
        sys.stdout.write("\n" + question)
        for item in menu:
            buffer = "\n "
            buffer += "-> (" if item[1] else "   ("
            buffer += "*) " if item[2] else " ) "
            buffer += item[0]
            sys.stdout.write(buffer)
        sys.stdout.flush()

    def _multiselect_question_move_cursor_and_reprint(self, question: str, menu: List[List], is_down: bool):
        pointed = [i for i, x in enumerate(menu) if x[1]]  # Get index or empty list.
        # Calculate desired position.
        desired_position: int
        if not pointed:
            desired_position = 0
        else:
            current_position = pointed[0]
            desired_position = current_position
            if is_down:
                if current_position < len(menu) - 1:
                    desired_position = current_position + 1
            else:
                if current_position > 0:
                    desired_position = current_position - 1
        # Change position.
        menu[current_position][1] = False
        menu[desired_position][1] = True
        # Note that `self.clean_lines(cnt)` works wrong after long lines being split due to lenght.
        self._clear_up_to_saved_position()
        self._save_cursor_position_and_print_multiselect_question(question, menu)

    def _multiselect_question_switch_option_and_reprint(self, question: str, menu: List[List]):
        for item in menu:
            if item[1]:
                item[2] = not item[2]
                self._multiselect_question_move_cursor_and_reprint(question, menu, True)
                return

    def ask_multiselect_question(self, question: str, options: List[str]) -> List[str]:
        if not options:  # Avoid errors on building menu (simplified code below)
            raise ValueError("ask_multiselect_question: Empty options are provided.")
        # Modify TTY. Based on https://stackoverflow.com/a/47955341 and https://stackoverflow.com/a/47197390/1535127
        # Save and clone TTY attributes.
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        # Correct TTY attributes to read single key strokes. See https://man7.org/linux/man-pages/man3/termios.3.html
        new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
        # Start block of changes which should be reverted afterwards.
        try:
            # Prepare menu. Set pointer on the first option.
            menu = [[x, False, False] for x in options]
            menu[0][1] = True
            # Hide cursor and print question with menu.
            # NOTE breaks cursor movements in xfce4-terminal sys.stdout.write(self.ANSI_HIDE_CURSOR)
            self._save_cursor_position_and_print_multiselect_question(question, menu)
            # Apply TTY modifications.
            termios.tcsetattr(fd, termios.TCSANOW, new)
            is_exit = False
            # Listen key strokes and modify menu.
            while True:
                event = os.read(fd, 3).decode()
                if len(event) == 3:
                    code = ord(event[2])  # All 3 numbers mean special key when code is a last number.
                    if code == 65:  # Up
                        self._multiselect_question_move_cursor_and_reprint(question, menu, False)
                    elif code == 66:  # Down
                        self._multiselect_question_move_cursor_and_reprint(question, menu, True)
                elif len(event) == 1:
                    if event[0] == " ":  # Space to set/unset menu item.
                        self._multiselect_question_switch_option_and_reprint(question, menu)
                    elif event[0] == "\n":  # Enter
                        is_exit = True
                    elif event[0] == "\x1b":  # Escape
                        for item in menu:  # Clean all selected options to simulate "I just want to exit!".
                            item[2] = False
                        is_exit = True
                if is_exit:
                    self._clear_up_to_saved_position()
                    break
        finally:
            # Revert TTY attributes.
            # TODO fix - doesn't revert in xfce4-terminal. STR: run once, stop, try again - menu doesn't clean.
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            # NOTE breaks cursor movements in xfce4-terminal sys.stdout.write(self.ANSI_SHOW_CURSOR)
        return [x[0] for x in menu if x[2]]


class Context:
    """
    Container for "tune rules" data. Allows save and restore data.
    """

    USED_RULES_METRIC_NAME = "used rules"
    FOUND_ACTIVITIES_METRIC_NAME = "found activities"
    SAVE_FILE_PATH = os.path.abspath("tune_rules-context.dill")

    def __init__(self, coefs: List = [], intercept: List = []) -> None:
        self.coefs: List = coefs
        self.intercept: List = intercept

    def save(self):
        """
        Saves itself into `SAVE_FILE_PATH` file.
        """
        with open(Context.SAVE_FILE_PATH, "wb") as f:
            dill.dump(self, f)
        LOG.info("Saved current context into %s file.", Context.SAVE_FILE_PATH)

    @staticmethod
    def read_from_file() -> "Context":
        """
        Reads context of tuning from the `SAVE_FILE_PATH` file.
        """
        with open(Context.SAVE_FILE_PATH, "rb") as f:
            result: Context = dill.load(f)
        LOG.info(
            "Restored context from '%s' file with %d coefs and %d intercept.",
            Context.SAVE_FILE_PATH,
            len(result.coefs),
            len(result.intercept),
        )
        return result

    def to_ba_finder(self) -> "SupervisedBAFinder":
        finder = SupervisedBAFinder(self)
        finder.set_coefs(self.coefs, self.intercept)
        return finder


class SupervisedBAFinder(BAFinder):
    """
    Wrapper under `BAFinder` which may train model inside by asking user for the right activity-by-strategies
    for "next" time slots.
    """

    def __init__(self, context: Context) -> None:
        super(SupervisedBAFinder, self).__init__()
        self.context = context
        self.training_data: List[Tuple[IntervalFeatures, int]] = []
        if sys.platform.startswith("win"):
            LOG.error("Windows terminal is not supported yet")
            exit(1)
            # leader = WindowsTerminalLeader() TODO
        self.leader: TerminalLeader = UnixTerminalLeader()

    def _add_answer(self, features: List[IntervalFeatures], index_of_chosen: int):
        for i, feature in enumerate(features):
            self.training_data.append((feature, 1 if i == index_of_chosen else 0))

    def find_top(
        self,
        candidates: List[intervaltree.Interval],
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
    ) -> Tuple[intervaltree.Interval, float, float]:
        # Calculate features for all intersecting intervals. Note that here may be few hundreds candidates.
        features = self.calculate_features(candidates, start_point, end_point, max_duration_seconds)
        # Prepare candidates to show: sort them and convert into strings.
        # For sorting use "overlap_ratio"
        sorted_indices = sorted(range(len(features)), key=lambda i: features[i].overlap_ratio, reverse=True)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_candidates = [candidates[i] for i in sorted_indices]
        options = []
        for candidate in sorted_candidates:
            activity: ActivityByStrategy = candidate.data
            options.append(str(activity))
        question = "Which activity-by-strategy from 'z###-*' buckets (see ActivityWatch 'Timeline') represents " +\
                   f"activty started from {datetime_to_time_str(start_point)} to {datetime_to_time_str(end_point)}?"
        answer: Tuple[str, int] = self.leader.ask_select_question(question, options)
        try:
            index_chosen_in_sorted_candidates = answer[1]
            result = sorted_candidates[index_chosen_in_sorted_candidates]
            # If there were no exceptions above then add to training data.
            self._add_answer(sorted_features, index_chosen_in_sorted_candidates)
            return result, 1, 0
        except (ValueError, IndexError) as e:
            raise ValueError(f"Wrong answer/choice '{answer}': {e}") from e

    def dump_results(self, is_ask_user: bool = False):
        self.context.save()
        if not is_ask_user or self.leader.ask_yes_no("Train by answers"):
            self.train(self.training_data)
            LOG.info("Training results: coef_=%s, intercept_=%s", self.model.coef_, self.model.intercept_)


class UploadDebugBucketsAndResetStep(AnalyzerStep):
    def __init__(self, client: aw_client.ActivityWatchClient):
        super(UploadDebugBucketsAndResetStep, self).__init__()
        self.client = client

    def get_description(self) -> str:
        return "Uploading 'debug' buckets."

    def check_context(self, context: Dict[str, any]) -> None:
        assert "debug_buckets_handler" in context, "Need in 'debug_buckets_handler' property"

    def run(self, context: Dict[str, any], metrics: Metrics) -> bool:
        debug_buckets_handler: DebugBucketsHandler = context.get("debug_buckets_handler")
        reload_debug_buckets(debug_buckets_handler.events, self.client)
        # Remove existing DebugBucketsHandler from context to don't load duplicates.
        # TODO don't remove "self_sufficient" results!
        del context["debug_buckets_handler"]
        return True


def tune_rules(events_date: datetime.datetime, is_use_saved_context: bool):
    """
    Gets all ActivityWatch events for the specified date, builds linked list of intervals from them,
    analyzes intervals, converts them into combined activities by specified (and fine-tuned per person) rules,
    prints them into output.
    :param events_date: Date to work with events on.
    :param is_use_saved_context: Flag to read data saved from previous run.
    :return: 'AnalyzerResult' object or 'None' if no intervals to analyze were found.
    """
    client = aw_client.ActivityWatchClient(os.path.basename(__file__))
    if is_use_saved_context:
        context = Context.read_from_file()
    else:
        context = Context()
    # Build ActivitiesByStrategy list by provided events date.
    activities_by_strategy, metrics = get_activities_by_strategy(events_date, client)
    # TODO: don't see options for manual judgements:
    # - with proper splitting by AFK
    if not activities_by_strategy:
        LOG.warning("Can't find events/intervals for %s. Doing nothing.", events_date.date())
        return None

    # Start to decide activities with last step involving user interactions.
    ba_finder = context.to_ba_finder()
    analyzer_result: AnalyzerResult = merge_activities(
        activities_by_strategy=activities_by_strategy,
        steps=[
            MakeResultTreeFromSelfSufficientActivitiesStep(True),
            ChopActivitiesByResultTreeStep(True, True),
            MakeCandidatesTreeStep(True),
            UploadDebugBucketsAndResetStep(client),
            MergeCandidatesTreeIntoResultTreeWithDedicatedBAFinderStep(ba_finder=ba_finder, is_add_debug_buckets=True),
        ],
    )
    if analyzer_result:
        LOG.info(analyzer_result.to_str())
        for bucket_id, events in analyzer_result.debug_dict.items():
            LOG.info(upload_events(events, DEBUG_BUCKETS_IMPORTER_NAME, bucket_id, client=client))
        ba_finder.dump_results(is_ask_user=True)
    else:
        LOG.error("Haven't received analyzer results!")
    return analyzer_result


def main():
    parser = argparse.ArgumentParser(
        description="Makes the same as 'get_activities' but together with providing result asks user about "
        "what they expect in order to tune inner 'find basic activity' model for user data.\n"
        "Always populates 'debug buckets' to allow user to choose from.\n"
        "In details script work looks like:\n"
        f"1. If configured it reads previous session from '{Context.SAVE_FILE_PATH}' file if it exists.\n"
        "2. Reads events for specified day, makes all required steps to build activities.\n"
        "3. When starts to find basic activities it asks user for each place - what to add. "
        "Options are sorted basing on existing coefficients.\n"
        "4. Basing on user answers it changes model coefficients to use them on next days.\n"
        "5. If need to correct results scrip may be executed on the few days.\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "date",
        nargs="?",
        type=valid_date,
        help="Date to analyze AcivityWatch events in format 'YYYY-mm-dd'. By-default is today. "
        "If omit here but set 'back days' argument then date is calculated as today - back_days.",
    )
    parser.add_argument(
        "-b",
        "--back-days",
        type=int,
        help="How many days back search events on. I.e. '1' value means 'search for yesterday.",
    )
    parser.add_argument(
        "-l",
        "--load-context",
        dest="is_use_saved",
        action="store_true",
        help="Flag to load saved context from previous execution.",
    )
    args = parser.parse_args()
    events_date = args.date if args.date else datetime.datetime.today().astimezone()
    if args.back_days and args.back_days > 0:
        events_date = events_date - datetime.timedelta(days=args.back_days)
    tune_rules(events_date, args.is_use_saved)


@contextlib.contextmanager  # TODO remove after fixing all terminal issues
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


def main2():  # TODO remove after fixing all terminal issues
    print("exit with ^C or ^D")
    with raw_mode(sys.stdin):
        try:
            while True:
                ch = sys.stdin.read(1)
                if not ch or ch == chr(4):
                    break
                print("%02x" % ord(ch))
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    LOG = setup_logging()
    main()
