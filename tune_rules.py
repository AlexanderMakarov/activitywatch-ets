#!/usr/bin/env python3
import datetime
import os
import sys
import argparse
import copy
# import pickle
import dill as pickle  # For pickle-ing lambdas need to use 'dill' package.
from typing import Any, List, Dict, Tuple
import contextlib
import aw_client
# Don't use convenient pyinput because https://pynput.readthedocs.io/en/latest/limitations.html#platform-limitations
# For Unix terminal:
try:
    import termios
except ImportError:
    pass
# For Windows terminal:
# try:
#     import msvcrt
# except ImportError:
#     pass
from pick import pick

from activity_merger.config.config import LOG, MIN_DURATION_SEC, RULES
from activity_merger.domain.input_entities import EventKeyHandler, Event
from activity_merger.domain.interval import Interval
from activity_merger.helpers.helpers import event_data_to_str, setup_logging, valid_date
from activity_merger.domain.analyzer import analyze_intervals, ProblemReporter, ANALYZE_MODE_TUNER
from activity_merger.domain.output_entities import AnalyzerResult
from get_activities import get_interval, upload_debug_buckets, print_analyzer_result


class Decision:
    """
    Shallow wrapper around 'Interval' without recursive links and with extra information
    like user decision for this interval and problems with them.
    """
    SKIP = 1
    MERGE_NEXT = 2

    def __init__(self, interval: Interval) -> None:
        # Copy interval without 'prev' and 'next' attributes to don't get errors caused by big recursion.
        self.interval = Interval(interval.start_time, interval.end_time)
        self.interval.events = interval.events
        self.decision = None
        self.problem = None


class Context:
    FILE_PATH = os.path.abspath("tune_rules-context.pickle")

    def __init__(self, decisions: List[Decision], first_interval: Interval, rules: Dict[str, List[EventKeyHandler]])\
            -> None:
        self.decisions = decisions
        self.first_interval = first_interval
        self.rules = rules

    def save(self):
        tmp = self.first_interval
        try:
            self.first_interval = None  # Can't save 'Interval' due to deep recursion.
            with open(Context.FILE_PATH, "wb") as f:
                pickle.dump(self, f)
            LOG.info("Saved current context into %s file.", Context.FILE_PATH)
        finally:
            self.first_interval = tmp

    @staticmethod
    def build_intervals_from_decisions(decisions: List[Decision]) -> Interval:
        interval = None
        for decision in decisions:
            d_interval = decision.interval
            # 'interval if interval else None' below is needed because Pyton passes value by link always.
            interval = Interval(d_interval.start_time, d_interval.end_time, interval if interval else None)
            interval.events = d_interval.events
        return interval

    @staticmethod
    def read_from_file() -> 'Decision':
        with open(Context.FILE_PATH, "rb") as f:
            result: Context = pickle.load(f)
            result.first_interval = Context.build_intervals_from_decisions(result.decisions)
            result.first_interval = result.first_interval.iterate_prev()
        LOG.info("Restored context with %d intervals from %s file.", len(result.decisions), Context.FILE_PATH)
        return result

    def get_undecided_intervals(self) -> List[Decision]:
        return [x for x in self.decisions if not x.decision]


class TerminalLeader:
    """
    Abstract matching song leader which expects user answers in terminal. Wraps `Matcher`, manages "ask - answer -
    check" flow and prints statistic at the end. Delegates operating system specific actions to inheritors.
    """

    def __init__(self):
        pass

    def read_user_input(self) -> str:
        raise NotImplementedError()

    def clean_lines(self, cnt: int):
        raise NotImplementedError()

    def ask_yes_no(self, question_without_yn: str) -> bool:
        sys.stdout.write(question_without_yn + " [y/n]: ")
        sys.stdout.flush()
        result = self.read_user_input().lower() == "y"
        sys.stdout.write("\n")
        sys.stdout.flush()
        return result

    def _ask_multiselect_question(self, question: str, options: List[str]) -> List[str]:
        """
        Draws multiple lines of text on the screen with options to choose one or few. Cleans console after itself.
        :param question: Question choose options for.
        :param options: List of options to choose.
        :return: List of chosen options.
        """
        # 'pick' library cleans screen and draws menu with options. At end disappears and leaves old content.
        result = pick(options, question, multiselect=True, default_index=2, min_selection_count=1)
        return result

    def _ask_decision(self, interval: Interval) -> List[Any]:
        question = interval.to_str(only_time=True)
        SKIP_TEXT = 'skip from activities'
        MERGE_TEXT = 'merge with next interval' + "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        options = {  # Use keys as 'value to present to user' and values to return result.
            SKIP_TEXT: Decision.SKIP,
            MERGE_TEXT: Decision.MERGE_NEXT,
        }
        for event in interval.events:
            options[event_data_to_str(event)] = event
        # Start loop of asking. Because some answers may contradict each other.
        decision_items: List[str]
        while True:
            selected: List[str] = self._ask_multiselect_question(question, list(options.keys()))
            decision_items = []
            if SKIP_TEXT in selected:
                decision_items.append('skip')
            if MERGE_TEXT in selected:
                decision_items.append('merge')
            if len(decision_items) > 1:
                pass
            break
        LOG.info("%s %s", question, ', '.join(decision_items))
        return [options[x] for x in selected]

    def ask_decisions(self, undecided_list: List[Decision]) -> bool:
        """
        Interacts with user asking decisions for given list of Interval-s. At start displays legend and asks if need
        proceed.
        :param undecided_list: List of undecided 'Decision'-s to decide on.
        :return: `True` if need to stop tuning and just print result, `False` to proceed with one more iteration. 
        """
        if not self.ask_yes_no("Proceed with 'decide for interval' session?"):
            return True
        LOG.info("Next will be presented %d intervals with options to choose. "
                 "Point (with \u2191 and \u2193) and press 'space' on one or few options you think should represent "
                 "each interval. Press 'Enter' to apply and proceed. "
                 "Press 'Escape' or choose nothing to stop deciding.", len(undecided_list))
        cnt_decided = 0
        for decision in undecided_list:
            selected = self._ask_decision(decision.interval)
            if selected:
                decision.decision = selected
                cnt_decided += 1
            elif self.ask_yes_no(f"Decided only {cnt_decided} from {len(undecided_list)} intervals. "
                                 "Are you sure you want to stop earlier?"):
                break
        return False


class UnixTerminalLeader(TerminalLeader):
    # FYI: https://wiki.bash-hackers.org/scripting/terminalcodes
    ANSI_CURSOR_UP = '\x1b[1A'  # Move cursor up (don't forget '\r' to put it on start of line).
    ANSI_CURSOR_SAVE_POSITION = '\x1b7'
    ANSI_CURSOR_TO_SAVED = '\x1b8'
    ANSI_CLEAR_CURRENT_LINE = '\x1b[2K'  # Clear the whole line where cursor is placed.
    ANSI_CLEAR_PREVIOUS_LINE = '\033[A'
    ANSI_CLEAR_TO_END_OF_SCREN = '\x1b[J'
    ANSI_HIDE_CURSOR = '\033[?25l'
    ANSI_SHOW_CURSOR = '\033[?25h'
    ANSI_FONT_DECORATION_STOP = '\u001b[0m'
    ANSI_FONT_DECORATION_BOLD = '\u001b[1m'
    ANSI_FONT_DECORATION_UNDERLINE = '\u001b[4m'
    ANSI_SHIFT_CURSOR_LEFT_PREFIX = '\u001b['
    ANSI_SHIFT_CURSOR_LEFT_SUFFIX = 'D'
    KEY_ESC = '\u00001b' #  '\x1b'
    KEY_UP = '\u1b5b41' # '\x1b[A'
    KEY_DOWN = '\u1b5b42' #  '\x1b[B'
    KEY_LEFT = '\u1b5b44' #  '\x1b[D'
    KEY_RIGHT = '\u1b5b43' # '\x1b[C'
    KEY_MAPPING = {
        127: 'backspace',
        10: 'return',
        32: 'space',
        9: 'tab',
        27: 'esc',
        65: 'up',
        66: 'down',
        67: 'right',
        68: 'left'
    }

    def read_user_input(self) -> str:  # Based on https://stackoverflow.com/a/47955341
        # Get TTY attributes.
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        # Clone existing TTY attributes and correct them to read by one char.
        new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        # Read by char.
        user_input = ''
        try:
            termios.tcsetattr(fd, termios.TCSANOW, new)
            while True:
                character = os.read(fd, 1)
                # If user typed terminating key then stop listen input.
                if character == b'\n':
                    break
                elif character == b'\x7f':  # Backspace/delete is hit.
                    sys.stdout.write("\b%s %s\b" % (
                        self.ANSI_FONT_DECORATION_UNDERLINE, self.ANSI_FONT_DECORATION_STOP))
                    user_input = user_input[:-1]
                else:
                    try:
                        character_str = character.decode("utf-8")
                    except UnicodeDecodeError as e:
                        print("Unsupported character: %s" % e)
                        return user_input
                    sys.stdout.write(character_str)
                    user_input += character_str
                sys.stdout.flush()
        finally:
            # Revert TTY attributes.
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        return user_input

    def clean_lines(self, cnt: int):
        # sys.stdout.write("".join(["\r%s\%s" % (self.ANSI_ERASE_LINE, self.ANSI_CURSOR_UP)] * cnt))
        sys.stdout.write("".join([self.ANSI_CLEAR_PREVIOUS_LINE] * cnt))
        sys.stdout.flush()

    def ask_yes_no(self, question_without_legend: str) -> bool:
        sys.stdout.write(question_without_legend + " [y/n]: ")
        sys.stdout.flush()
        result = self.read_user_input().lower() == "y"
        sys.stdout.write(self.ANSI_CLEAR_PREVIOUS_LINE)
        sys.stdout.flush()
        return result

    def _save_cursor_position(self):
        sys.stdout.write(self.ANSI_CURSOR_SAVE_POSITION)

    def _clear_up_to_saved_position(self):
        sys.stdout.write(self.ANSI_CURSOR_TO_SAVED + self.ANSI_CLEAR_TO_END_OF_SCREN)
        sys.stdout.flush()

    def _save_cursor_position_and_print_multiselect_question(self, question: str, menu: List[List]):
        # item = [description, is_cursor, is_selected]
        self._save_cursor_position()
        sys.stdout.write('\n' + question)
        for item in menu:
            buffer = '\n '
            buffer += '-> (' if item[1] else '   ('
            buffer += '*) ' if item[2] else ' ) '
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

    def _ask_multiselect_question(self, question: str, options: List[str]) -> List[str]:
        if not options:  # Avoid errors on building menu (smelling code below)
            raise ValueError("Empty options are provided.")
        # Modify TTY. Based on https://stackoverflow.com/a/47955341 and https://stackoverflow.com/a/47197390/1535127
        # Save and clone TTY attributes.
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        # Correct TTY attributes to read single key strokes.
        new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        # Start block of changes which should be reverted afterwards.
        try:
            # Prepare menu. Set pointer on the first option.
            menu = [[x, False, False] for x in options]
            menu[0][1] = True
            # Hide cursor and print question with menu.
            sys.stdout.write(self.ANSI_HIDE_CURSOR)
            self._save_cursor_position_and_print_multiselect_question(question, menu)
            # Apply TTY modifications.
            termios.tcsetattr(fd, termios.TCSANOW, new)
            key = []
            is_exit = False
            # Listen key strokes and modify menu.
            while True:
                event = os.read(fd, 3).decode()
                if len(event) == 3:
                    k = ord(event[2])  # All 3 numbers mean special key when code is a last number.
                    key = UnixTerminalLeader.KEY_MAPPING.get(k, None)
                    if key == 'up':
                        self._multiselect_question_move_cursor_and_reprint(question, menu, False)
                    elif key == 'down':
                        self._multiselect_question_move_cursor_and_reprint(question, menu, True)
                elif len(event) == 1:
                    if event[0] == ' ':  # Space to choose menu item.
                        self._multiselect_question_switch_option_and_reprint(question, menu)
                    elif event[0] == '\n':  # Enter was hit.
                        is_exit = True
                    elif event[0] == '\x1b':  # Escape was hit.
                        for item in menu:  # Clean all selected options to simulate "I just want to exit!".
                            item[2] = False
                        is_exit = True
                if is_exit:
                    self._clear_up_to_saved_position()
                    break
        finally:
            # Revert TTY attributes.
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
            sys.stdout.write(self.ANSI_SHOW_CURSOR)  # Show cursor back.
        return [x[0] for x in menu if x[2]]

    def build_letters_number_hint_word(self, hint: Event) -> str:
        return '%s%s%s%s%s' % (hint.prefix, self.ANSI_FONT_DECORATION_UNDERLINE, " " * hint.word_len,
                               self.ANSI_FONT_DECORATION_STOP, hint.suffix)

    def ask_user_input_this_line(self, matched_line_text: str, prev_match_desc: str):
        line = "\r%s%s%s %s%s" % (
            self.ANSI_FONT_DECORATION_BOLD, "", self.ANSI_FONT_DECORATION_STOP, matched_line_text, "")
        sys.stdout.write(self.ANSI_CLEAR_CURRENT_LINE + line)
        sys.stdout.flush()

    def complete_line(self, matched_text: str, statistic: List[Event]):
        sys.stdout.write('\r%s%s' % (self.ANSI_CLEAR_CURRENT_LINE, ""))
        sys.stdout.flush()


def ask_decision_and_correct_rules(context: Context) -> bool:
    """
    Interacts with user asking about required intervals decisions, next makes suggestion for "analyze" rules.
    :param context: Context with "what to ask" and current progress.
    :return: Flag that user chose to stop tuning.
    """
    if sys.platform.startswith("win"):
        LOG.error("Windows terminal is not supported yet")
        # leader = WindowsTerminalLeader(Matcher(song, args))
    else:
        leader = UnixTerminalLeader()
    # Calculate which decisions need to make.
    undecided_intervals = context.get_undecided_intervals()
    # Interact with user asking for "proceed?" and decisions.
    is_exit = leader.ask_decisions(undecided_intervals)
    decided_intervals = [x for x in undecided_intervals if x.decision]
    LOG.info("Got decisions for %d from %d asked intervals.", len(decided_intervals), len(undecided_intervals))
    # Calculate new rules and show problems.
    context.recalculate_rules()
    return is_exit


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
        # Build time-ordered linked list of intervals by provided events.
        interval = get_interval(events_date, client)
        if interval is None:
            LOG.warning("Can't find events/intervals for %s. Doing nothing.", events_date.date())
            return None
        # Scroll to the first/oldest interval.
        interval = interval.iterate_prev()
        # Prepare context: make Session from 'interval' and make deep clone of rules.
        decisions = []
        interval.iterate_next(lambda x: decisions.append(Decision(x)))
        context = Context(decisions, interval, copy.deepcopy(RULES))
        # Save context right away to skip steps above next time.
        context.save()
    analyzer_result: AnalyzerResult
    LOG.info("---- Tuning started.")
    while True:
        # Analyze interval with hiding all problems logs. Upload debug buckets.
        analyzer_result = analyze_intervals(context.first_interval, MIN_DURATION_SEC, context.rules,
                                            ProblemReporter.SUPPORTED_PROBLEMS, ANALYZE_MODE_TUNER)
        upload_debug_buckets(analyzer_result, client)
        # Interact with user.
        is_exit = ask_decision_and_correct_rules(context)
        context.save()
        if is_exit:
            break
    LOG.info("---- Tuning completed.")
    print_analyzer_result(analyzer_result)
    return analyzer_result


def main():
    parser = argparse.ArgumentParser(
        description="Makes the same as 'get_activities' but together with providing result asks user about "
                    "what they expect and suggests corrections to current rules.\n"
                    "Can't make new rules and somehow correct configuration file. All these actions are manual!\n"
                    "In details it:\n"
                    "1. calls AcivityWatch for all available events on specified date,\n"
                    "2. merges all events by initial or built in previous run rules into linked list of 'intervals',\n"
                    "3. analyzes 'intervals' and makes 'actvities' from them,\n"
                    "4. imports analyzing intermediate results into AcivityWatch as events in 'debug' buckets,\n"
                    "5. asks user what is expected to be dominant event for all or 'arguable' intervals,\n"
                    "6. basing on answers corrects initial rules in-memory and saves context in file,\n"
                    "7. analyzes 'intervals' again (steps 3-4) and asks if need to improve existing rules further,\n"
                    "8. if tuned rules are OK or needs to manual correction then prints them and exits.\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('date', nargs='?', type=valid_date,
                        help="Date to analyze AcivityWatch events in format 'YYYY-mm-dd'. By-default is today. "
                             "If omit here but set 'back days' argument then date is calculated as today - back_days.")
    parser.add_argument('-b', '--back-days', type=int,
                        help="How many days back search events on. I.e. '1' value means 'search for yesterday.")
    parser.add_argument('-l', '--load-context', dest='is_use_saved', action='store_true',
                        help="Flag to load saved context from previous execution.")
    args = parser.parse_args()
    events_date = args.date if args.date else datetime.datetime.today().astimezone()
    if args.back_days and args.back_days > 0:
        events_date = (events_date - datetime.timedelta(days=args.back_days))
    tune_rules(events_date, args.is_use_saved)

@contextlib.contextmanager  # TODO remove
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


def main2():
    print('exit with ^C or ^D')
    with raw_mode(sys.stdin):
        try:
            while True:
                ch = sys.stdin.read(1)
                if not ch or ch == chr(4):
                    break
                print('%02x' % ord(ch))
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == '__main__':
    LOG = setup_logging()
    main()
