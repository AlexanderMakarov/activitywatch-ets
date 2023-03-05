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
        return self.read_user_input().lower() == "y"

    def _ask_multiselect(self, options: List[str], question: str) -> List[str]:
        """
        Draws multiple lines of text on the screen with options to choose one or few. Cleans console after itself.
        :param options: List of options to choose.
        :param question: Question choose options for.
        :return: List of chosen options.
        """
        # 'pick' library cleans screen and draws menu with options. At end disappears and leaves old content.
        result = pick(options, question, multiselect=True, default_index=2, min_selection_count=1)
        return result

    def _ask_decision(self, interval: Interval) -> List[Any]:
        title = interval.to_str(only_time=True) + ":"
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
            selected: List[str] = self._ask_multiselect(list(options.keys()), title)
            decision_items = []
            if SKIP_TEXT in selected:
                decision_items.append('skip')
            if MERGE_TEXT in selected:
                decision_items.append('merge')
            if len(decision_items) > 1:
                pass
            break
        LOG.info("%s %s", title, ', '.join(decision_items))
        return [options[x] for x in selected]

    def ask_decisions(self, undecided_list: List[Decision]) -> bool:
        self.ask_yes_no("Next will be presented intervals with options to choose. "
                        "Point (with \u2191 and \u2193) and press 'space' on one or few options "
                        "you expect to represent this interval. Press 'Enter' to apply and proceed. "
                        "Choose nothing to stop deciding. Ready?")
        cnt_decided = 0
        for decision in undecided_list:
            selected = self._ask_decision(decision.interval)
            if selected:
                decision.decision = selected
                cnt_decided += 1
            elif self.ask_yes_no(f"Decided {cnt_decided}/{len(undecided_list)}. Are you sure want to stop?"):
                return True
        return False


class UnixTerminalLeader(TerminalLeader):
    ANSI_ERASE_LINE = '\u001b[2K'  # Clear the whole line where cursor is placed.
    ANSI_CURSOR_UP = '\u001b[1A'  # Move cursor up (don't forget '\r' to put it on start of line).
    ANSI_CLEAR_PREVIOUS_LINE = '\033[A\033[A'
    ANSI_HIDE_CURSOR = '\033[?25l'
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

    def _move_menu_cursor_and_reprint(self, menu: List[List], is_down: bool):
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
        self.clean_lines(len(menu))
        self._print_menu(menu)

    def _select_menu_item_and_reprint(self, menu: List[List]):
        for item in menu:
            if item[1]:
                item[2] = True
                self._move_menu_cursor_and_reprint(menu, True)
                return

    def _print_menu(self, menu: List[List]):
        # item = [description, is_cursor, is_selected]
        for item in menu:
            buffer = '\n '
            buffer += '-> (' if item[1] else '   ('
            buffer += '*) ' if item[2] else ' ) '
            buffer += item[0]
            sys.stdout.write(buffer)
        sys.stdout.flush()

    def _ask_multiselect(self, options: List[str], question: str) -> List[str]:
        menu = [[x, False, False] for x in options]
        menu[0][1] = True  # Put cursor on the first option.
        sys.stdout.write("\n" + question + self.ANSI_HIDE_CURSOR)
        self._print_menu(menu)
        # Based on https://stackoverflow.com/a/47955341 and https://stackoverflow.com/a/47197390/1535127
        # Save and clone TTY attributes.
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        # Correct TTY attributes to read single key strokes.
        new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        # Listen key strokes.
        try:
            termios.tcsetattr(fd, termios.TCSANOW, new)
            key = []
            is_exit = False
            while True:
                event = os.read(fd, 3).decode()
                if len(event) == 3:
                    k = ord(event[2])  # All 3 numbers mean special key when code is a last number.
                    key = UnixTerminalLeader.KEY_MAPPING.get(k, None)
                    if key == 'up':
                        self._move_menu_cursor_and_reprint(menu, False)
                    elif key == 'down':
                        self._move_menu_cursor_and_reprint(menu, True)
                elif len(event) == 1:
                    if event[0] == ' ':  # Space to choose menu item.
                        self._select_menu_item_and_reprint(menu)
                    elif event[0] == '\n':  # Enter was hit.
                        is_exit = True
                    elif event[0] == '\x1b':  # Escape was hit.
                        for item in menu:  # Clean all selected options to simulate "I just want to exit!".
                            item[2] = False
                        is_exit = True
                if is_exit:
                    self.clean_lines(len(options) + 1)  # +1 is to remove question as well.
                    break
        finally:
            # Revert TTY attributes.
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        return [options[0] for x in menu if x[2]]

    def build_letters_number_hint_word(self, hint: Event) -> str:
        return '%s%s%s%s%s' % (hint.prefix, self.ANSI_FONT_DECORATION_UNDERLINE, " " * hint.word_len,
                               self.ANSI_FONT_DECORATION_STOP, hint.suffix)

    def ask_user_input_this_line(self, matched_line_text: str, prev_match_desc: str):
        line = "\r%s%s%s %s%s" % (
            self.ANSI_FONT_DECORATION_BOLD, "", self.ANSI_FONT_DECORATION_STOP, matched_line_text, "")
        sys.stdout.write(self.ANSI_ERASE_LINE + line)
        sys.stdout.flush()

    def complete_line(self, matched_text: str, statistic: List[Event]):
        sys.stdout.write('\r%s%s' % (self.ANSI_ERASE_LINE, ""))
        sys.stdout.flush()


def _ask_decision_and_correct_rules(context: Context) -> bool:
    if sys.platform.startswith("win"):
        LOG.error("Windows terminal is not supported yet")
        # leader = WindowsTerminalLeader(Matcher(song, args))
    else:
        leader = UnixTerminalLeader()
    # 1) check if we need one more iteration. TODO maybe ask it on each interval instead?
    intervals = context.get_undecided_intervals()
    # if leader.ask_yes_no(f"Remained {len(intervals)} undecided intervals. Proceed with tuning?"):
    #     return True
    # 2) ask decision for all undecided intervals.
    is_exit = leader.ask_decisions(intervals)
    decided_intervals = [x for x in intervals if x.decision]
    LOG.info("Got decisions for %d from %d asked on this iteration intervals.", len(decided_intervals), len(intervals))
    # 3) calculate new rules.
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
    while True:
        # Analyze interval with hiding all problems logs. Upload debug buckets.
        analyzer_result = analyze_intervals(context.first_interval, MIN_DURATION_SEC, context.rules,
                                            ProblemReporter.SUPPORTED_PROBLEMS, ANALYZE_MODE_TUNER)
        upload_debug_buckets(analyzer_result, client)
        LOG.info("---- Start tuning.")
        # Ask user for each interval (TODO only with problems) and adjust rules basing on it.
        is_one_more_iteration = _ask_decision_and_correct_rules(context)
        context.save()
        if not is_one_more_iteration:
            break
    LOG.info("---- Tuning completed. Final analyzing results:")
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
