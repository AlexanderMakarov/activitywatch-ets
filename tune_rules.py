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
import curses

from activity_merger.config.config import DEBUG_BUCKETS_IMPORTER_NAME, LOG
from activity_merger.domain.analyzer import (
    RA_DEBUG_BUCKET_NAME,
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
from get_activities import clean_debug_buckets_and_apply_strategies_on_one_day_events, reload_debug_buckets


class PickTerminalUI:
    """
    Terminal UI based on 'pick' library.
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


class CursesTerminalUI:
    """
    Terminal UI based on 'curses' library.
    """

    def ask_yes_no(self, prefix_with_yn: List[str]) -> bool:
        """
        Prints the given prefix and waits 'y' or 'n' keypress.
        :param prefix_with_yn: List of lines to print as a question. It is good to finish it with '[y/n]' prompt.
        :returns: True if was chosen 'y', False if was chosen 'n'.
        """

        def curses_runner(stdscr, prefix_with_yn):
            curses.curs_set(0)
            stdscr.keypad(1)
            return CursesTerminalUI._ask_yes_no(stdscr, prefix_with_yn)

        return curses.wrapper(curses_runner, prefix_with_yn)

    @staticmethod
    def _ask_yes_no(stdscr, prefix_with_yn: List[str]) -> bool:
        while True:
            for i, line in enumerate(prefix_with_yn):
                stdscr.addstr(i, 0, line)
            stdscr.refresh()
            char = stdscr.getch()
            if char in [ord("y"), ord("Y")]:
                return True
            elif char in [ord("n"), ord("N")]:
                return False

    @staticmethod
    def _reprint_input_with_menu(stdscr, prefix: List[str], input_str: str, cursor_pos: int, menu: List[str]):
        stdscr.clear()
        rows, columns = stdscr.getmaxyx()
        # To avoid "addwstr() returned ERR" errors from `stdscr.addstr` need to print only in 1 screen, not more.
        # Note that last line will be wrapped so may span multiple lines.
        max_option_len = columns - 2  # Keep 2 characters for the pointer.
        line_index = 0
        for i, line in enumerate(prefix):
            stdscr.addstr(line_index, 0, line)
            line_index += 1
        stdscr.addstr(line_index, 0, "filter by: " + input_str)
        line_index += 1
        # Calculate the number of rows left.
        visible_options = rows - line_index - 1  # Show 1 line less.
        # Shift slice of menu to show to ensure the selected option is always visible.
        offset = max(0, cursor_pos + 1 - visible_options)
        for i in range(visible_options):
            menu_index = offset + i
            if menu_index >= len(menu):
                break
            option = menu[menu_index][:max_option_len]
            if menu_index == cursor_pos:
                stdscr.addstr(line_index, 0, f"* {option}", curses.A_REVERSE)  # Reverse colors for selection.
            else:
                stdscr.addstr(line_index, 0, f"  {option}")
            line_index += 1
        stdscr.refresh()

    def ask_select_question_with_type_filter(
        self, prefix: List[str], options: List[Tuple[str, str]]
    ) -> Tuple[str, int]:
        """
        Asks user for "select" question with ability to input string to filter options list by "contains".
        :param prefix: List of lines to prepend menu. Usually contains question.
        :param options: List of tuples where first element is what to print to the user
        and second element is string to filter by.
        :return: A tuple with chosen option and index of it in the options. If [None, -1] then user decided to break.
        """
        if not options:  # Avoid errors on building menu (simplifies code below)
            raise ValueError("ask_select_question_with_type_filter: Empty options are provided.")
        # Prepare full menu.
        menu = [x[0] for x in options]

        def curses_runner(stdscr, prefix, menu):
            curses.curs_set(0)
            stdscr.keypad(1)
            input_str = ""
            cursor_pos = 0
            chosen_item = None
            chosen_index = -1
            while True:
                self._reprint_input_with_menu(stdscr, prefix, input_str, cursor_pos, menu)
                key = stdscr.getch()
                if key == curses.KEY_UP and cursor_pos > 0:
                    cursor_pos -= 1
                elif key == curses.KEY_DOWN and cursor_pos < len(menu) - 1:
                    cursor_pos += 1
                elif key == curses.KEY_ENTER or key in [10, 13]:
                    # Restore chosen place in options from potentially shrinked menu.
                    chosen_item = menu[cursor_pos]
                    chosen_index = next(i for i, x in enumerate(options) if x[0] == chosen_item)
                    break
                elif 32 <= key <= 126:  # printable characters
                    input_str += chr(key)
                    menu = [x[0] for x in options if len(input_str) == 0 or input_str in x[1]]
                    cursor_pos = 0
                elif key == 263 and len(input_str) > 0:  # BACKSPACE
                    input_str = input_str[:-1]
                    menu = [x[0] for x in options if len(input_str) == 0 or input_str in x[1]]
                    cursor_pos = 0
                elif key == 27:  # ESCAPE
                    if CursesTerminalUI._ask_yes_no(stdscr, prefix + ["Do you want to break? [y/n]"]):
                        chosen_item = None
                        chosen_index = -1
                        break
            stdscr.clear()
            stdscr.addstr(0, 0, "You chose: " + menu[cursor_pos])
            return [chosen_item, chosen_index]
            # stdscr.refresh()
            # stdscr.getch()

        return curses.wrapper(curses_runner, prefix, menu)


class Context:
    """
    Container for "tune rules" data. Allows save and restore data.
    """

    USED_RULES_METRIC_NAME = "used rules"
    FOUND_ACTIVITIES_METRIC_NAME = "found activities"
    SAVE_FILE_PATH = os.path.abspath("tune_rules-context.dill")

    def __init__(self, coefs: List, intercept: List) -> None:
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
        self.leader: CursesTerminalUI = CursesTerminalUI()

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
            option_str = str(activity)
            options.append((option_str, option_str))  # Make the whole acitivity text as "searchable".
        question = (
            f"{datetime_to_time_str(start_point)} to {datetime_to_time_str(end_point)} - choose activity-by-strategy "
            "from 'z###-*' buckets on ActivityWatch 'Timeline' page for start of this interval."
        )
        # TODO (impr) need interactions to:
        # - revert previous decision
        # - show previous choice
        # - improve "decided to exit" handling
        answer: Tuple[str, int] = self.leader.ask_select_question_with_type_filter([question], options)
        try:
            index_chosen_in_sorted_candidates = answer[1]
            if index_chosen_in_sorted_candidates < 0:
                raise ValueError("Used decided to exit")
            result = sorted_candidates[index_chosen_in_sorted_candidates]
            # If there were no exceptions above then add to training data.
            self._add_answer(sorted_features, index_chosen_in_sorted_candidates)
            return result, 1, 0
        except (ValueError, IndexError) as e:
            raise ValueError(f"Wrong answer/choice '{answer}': {e}") from e

    def dump_results(self, is_ask_user: bool = False):
        self.context.save()
        if not is_ask_user or self.leader.ask_yes_no(["Train by answers? [y/n]"]):
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
        context = Context([], [])
    # Build ActivitiesByStrategy list by provided events date.
    strategy_apply_result, metrics = clean_debug_buckets_and_apply_strategies_on_one_day_events(events_date, client)
    metrics_strings = list(metrics.to_strings())
    # Don't print resulting activity-by-strategies - better to see them in ActivityWatch UI.
    LOG.info("Analyzed all buckets separately, common metrics:\n%s", "\n".join(metrics_strings))
    LOG.info(
        "\n".join(x.strategy.name + " metrics:\n" + "\n".join(x.metrics.to_strings()) for x in strategy_apply_result)
    )
    if not strategy_apply_result:
        LOG.warning("Can't build activity-by-strategies by events/intervals for %s. Doing nothing.", events_date.date())
        return None

    # Start to decide activities with last step involving user interactions.
    ba_finder = context.to_ba_finder()
    analyzer_result: AnalyzerResult = merge_activities(
        strategy_apply_result=strategy_apply_result,
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
        # Reload only "resulting activity" debug bucket events! See UploadDebugBucketsAndResetStep.
        LOG.info(
            upload_events(
                events=analyzer_result.debug_dict[RA_DEBUG_BUCKET_NAME],
                event_type=DEBUG_BUCKETS_IMPORTER_NAME,
                bucket_id=RA_DEBUG_BUCKET_NAME,
                is_replace=True,
                client=client,
            )
        )
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
