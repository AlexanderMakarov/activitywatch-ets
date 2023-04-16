#!/usr/bin/env python3
import datetime
import os
import sys
import argparse
import copy
import dataclasses
import dill as pickle  # For pickle-ing lambdas need to use 'dill' package.
from typing import Any, List, Dict, Tuple
import contextlib
import aw_client
from activity_merger.domain.tuner import adjust_rules
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

from activity_merger.config.config import LOG, MIN_DURATION_SEC, RULES, BUCKET_DEBUG_RAW_RULE_RESULTS
from activity_merger.domain.input_entities import EventKeyHandler, Event, Rule
from activity_merger.domain.interval import Interval
from activity_merger.helpers.helpers import event_data_to_str, setup_logging, valid_date
from activity_merger.domain.analyzer import get_eventkeyhandlers_per_bucket_prefix, find_handler_for_event,\
                                            analyze_intervals, ProblemReporter, ANALYZE_MODE_TUNER
from activity_merger.domain.output_entities import AnalyzerResult
from activity_merger.domain.metrics import Metrics
from activity_merger.domain.tuner import Decision, adjust_rules
from get_activities import get_interval, reload_debug_buckets, print_analyzer_result


@dataclasses.dataclass(order=True)
class RuleNode:
    rule: Rule
    parent: 'RuleNode'
    children: List['RuleNode']
    # proposed_weight: int


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

    def _find_rules_per_decision(self, decision: Decision, eventkeyhandlers_per_bucket_prefix) -> Tuple[Rule]:
        rules = []
        for event in decision.events:
            handler = find_handler_for_event(event, eventkeyhandlers_per_bucket_prefix)
            if not handler:
                continue
            rule, _ = handler.get_rule(event)
            if rule:
                rules.append(rule)
        return tuple(sorted(rules))

    def _incosistent_decision_log(self, decision_object, group_decision_obj):
        a = (Decision.decision_item_to_str(x) for x in group_decision_obj.decision)
        b = (Decision.decision_item_to_str(x) for x in decision_object.decision)
        LOG.info("Inconsistency in decisions:\n"
                 "  for %s decision is\n    %s\n"
                 "  while for %s decision is\n    %s",
                 decision_object.interval.to_str(), "\n    ".join(b),
                 group_decision_obj, "\n    ".join(a))

    def recalculate_rules(self):
        """
        Modifies rules itself basing on decisions. If something contradicting or not not enough in decisions then
        explains it in logs.
        TODO saves decisions from this iterations and clears problems ones to ask user for decision one more time.
        """
        # 1. Iterate all decisions to:
        #   - checks which are decided
        #   - find contradictions like:
        #     * for similar set of events decisions are different in intervals
        #   - print contradictions
        #   - gather statistic (first part)
        # 2. correct rules basing on right decisions
        # 3. try to apply rules to all remained intervals and calculate activities
        # 4. print statistic
        # -----------
        # 1a - group decisons (intervals) by rules matching events inside.
        eventkeyhandlers_per_bucket_prefix = get_eventkeyhandlers_per_bucket_prefix(self.rules)
        combination_rules: Dict[Tuple, List[Decision]] = {}
        metrics = Metrics({
            'inconsistent_decision', self._incosistent_decision_log,
        }, None)
        for decision in self.decisions:
            if decision.decision:
                rules = self._find_rules_per_decision(decision, eventkeyhandlers_per_bucket_prefix)
                combination_rules.get(rules, set()).append(decision)
        # 1b + 2 - analyze resulting groups and either print contradictions or make correction to rules.
        rules_tree: RuleNode = None
        for rules, decision_objects in combination_rules.items():
            group_decision_obj = decision_objects[0]
            # Check for "no contradictions". TODO distinguish which rule is not specific enough.
            is_valid = True
            for decision_object in decision_objects:
                if decision_object.decision != group_decision_obj.decision:
                    metrics.report('inconsistent_decision', decision_object, decision_object=decision_object,
                                   group_decision_obj=group_decision_obj)
                    is_valid = False
            # If valid then build rules linked list with weights.
            if is_valid:
                # Need to build structure like: rA>rB, rC>rD, rD>rB, etc.
                # And sort them into a tree like: rB<rD<rC
                #                                   <rA
                # If it is impossible to build a tree/DAG (there are cycles) then report about all cases and fail.
                # Next scatter weights above this tree trying to keep at distance from each other and don't change.
                if rules_tree:
                    pass
        # Generate code in python to build a tree from dictionary of multiple "Rule" objects per multiple "Decision" objects.
        # If some decisions makes loops in this tree then show warnings.

        raise NotImplementedError("")
        return metrics


class TerminalLeader:
    """
    Abstract matching song leader which expects user answers in terminal. Wraps `Matcher`, manages "ask - answer -
    check" flow and prints statistic at the end. Delegates operating system specific actions to inheritors.
    """
    SKIP_TEXT = 'skip from activities'
    MERGE_TEXT = 'merge with next interval'

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
        interval_desc = interval.to_str(only_time=True)
        options = {  # Use keys as 'value to present to user' and values to return result.
            self.SKIP_TEXT: Decision.SKIP,
            self.MERGE_TEXT: Decision.MERGE_NEXT,
        }
        for event in interval.events:
            options[f"{event.bucket_id}: {event_data_to_str(event)}"] = event
        # Start loop of asking. Because some answers may contradict each other.
        result_desc_items: List[str]
        result = []
        while True:
            selected: List[str] = self._ask_multiselect_question(interval_desc, list(options.keys()))
            # Check for validity.
            if self.SKIP_TEXT in selected and self.MERGE_TEXT in selected:
                LOG.warning("It is impossible to both %s and %s. Please reconsider.", self.SKIP_TEXT, self.MERGE_TEXT)
                continue
            # Build text explanation of choice.
            result_desc_items = []  # Clear from previous attempt values.
            result = []
            for key in selected:
                value = options.get(key, None)
                result.append(value)
                if isinstance(value, Event):
                    result_desc_items.append(f"event from {value.bucket_id}")
                elif value == Decision.SKIP:
                    result_desc_items.append('skip')
                elif value == Decision.MERGE_NEXT:
                    result_desc_items.append('merge with next')
                else:
                    raise ValueError(f"Got wrong selected option '{key}.")
            break
        LOG.info("%s %s", interval_desc, ', '.join(result_desc_items))
        return result

    def ask_decisions(self, undecided_list: List[Decision]) -> bool:
        """
        Interacts with user asking decisions for given list of Interval-s. At start displays legend and asks if need
        proceed.
        :param undecided_list: List of undecided 'Decision'-s to decide on.
        :return: `True` if need to stop tuning and just print result, `False` to proceed with one more iteration. 
        """
        sys.stdout.write(
            "Next will be presented %d intervals with options to choose. "
            "Please open them in http://localhost:5600/#/timeline '%s' bucket as well. "
            "Note that this page with debug information may be quite heavy and slow due to number of elements. "
            "Point (with \u2191 and \u2193) and press 'Space' on one or few options you think should represent "
            "each interval. Press 'Enter' to apply and proceed. Press 'Escape' or choose nothing to stop deciding."
            "\nNote that for special behavior usually need to choose recorded event as well because:\n"
            "- %s - need to point event causing skipping,\n"
            "- %s - need to point event which makes this interval borrow meaning of the next interval. "
            "Except case when you decided to merge with next because of absence of event in the some bucket.\n" % \
            (len(undecided_list), BUCKET_DEBUG_RAW_RULE_RESULTS, self.SKIP_TEXT, self.MERGE_TEXT)
        )
        sys.stdout.flush()
        if not self.ask_yes_no("Proceed with 'decide for interval' session?"):
            return True
        cnt_decided = 0
        for decision in undecided_list:
            selected = self._ask_decision(decision.interval)
            if selected:
                decision.set_user_decision(selected)
                cnt_decided += 1
            elif self.ask_yes_no(f"Decided only {cnt_decided} from {len(undecided_list)} intervals. "
                                 "Are you sure you want to stop earlier?"):
                break
        return False


class UnixTerminalLeader(TerminalLeader):
    """
    TerminalLeader for Unix machines. Relies on VT100 escape codes and TTY.
    """

    # FYI: https://wiki.bash-hackers.org/scripting/terminalcodes
    ANSI_CURSOR_UP = '\x1b[1A'  # Move cursor up (don't forget '\r' to put it on start of line).
    ANSI_CURSOR_SAVE_POSITION = '\033[s'#'\x1b7'
    ANSI_CURSOR_TO_SAVED = '\033[u'#'\x1b8'
    ANSI_CLEAR_CURRENT_LINE = '\x1b[2K'  # Clear the whole line where cursor is placed.
    ANSI_CLEAR_PREVIOUS_LINE = '\033[A'
    ANSI_CLEAR_TO_END_OF_SCREEN = '\033[J'#'\x1b[J'
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
        if not options:  # Avoid errors on building menu (simplified code below)
            raise ValueError("_ask_multiselect_question: Empty options are provided.")
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
                    if event[0] == ' ':  # Space to set/unset menu item.
                        self._multiselect_question_switch_option_and_reprint(question, menu)
                    elif event[0] == '\n':  # Enter
                        is_exit = True
                    elif event[0] == '\x1b':  # Escape
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


def ask_decision_and_correct_rules(context: Context) -> bool:
    """
    Interacts with user asking about required intervals decisions, next makes suggestion for "analyze" rules.
    :param context: Context with "what to ask" and current progress.
    :return: Flag that user chose to stop tuning.
    """
    if sys.platform.startswith("win"):
        LOG.error("Windows terminal is not supported yet")
        # leader = WindowsTerminalLeader()
    else:
        leader = UnixTerminalLeader()
    # Calculate which decisions need to make.
    undecided_intervals: List[Decision] = context.get_undecided_intervals()
    # Interact with user asking for decisions like "this event describes this interval".
    is_exit = leader.ask_decisions(undecided_intervals)
    # Find out which intervals were decided on this round.
    decided_intervals_this_round = [x for x in undecided_intervals if x.decision]
    decided_intervals = [x for x in context.decisions if x.decision]
    LOG.info("Got decisions for %d from %d asked intervals. Total %d decided intervals from %d.",
             len(decided_intervals_this_round), len(undecided_intervals), len(decided_intervals),
             len(context.decisions))
    # Check if something was changed and adjust rules if yes.
    if decided_intervals:
        # Find out rules for each event in decided intervals (not this round, but all) to adjust them by decisions.
        eventkeyhandlers_per_bucket_prefix = get_eventkeyhandlers_per_bucket_prefix(context.rules)
        for decision in decided_intervals:
            # TODO print which rules were chosen per event.
            decision.set_rules_per_event(eventkeyhandlers_per_bucket_prefix)
        # Investigated decisions and adjust priorities for rules.
        rules, metrics = adjust_rules(decided_intervals, context.rules)
        LOG.info(metrics.to_str())
        assert False, "Need to complete code"
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
        reload_debug_buckets(analyzer_result, client)
        # Interact with user.
        is_exit = ask_decision_and_correct_rules(context)
        # TODO provide ability to reset some decisions (is_exit -> EXIT/APPLY_AND_SAVE/ONLY_APPLY)
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
