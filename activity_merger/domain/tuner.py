from typing import List, Dict, Tuple, Set, Union
import copy
from activity_merger.domain.input_entities import Event, EventKeyHandler, Rule
from activity_merger.domain.interval import Interval
from activity_merger.domain.analyzer import find_handler_for_event
from activity_merger.domain.metrics import Metrics
from ..config.config import LOG

SKIP_TEXT = 'skip from activities'
MERGE_TEXT = 'merge with next interval'
TUNER_ABILITIES_TEXT = "adjust Rule.priority values"

class IntervalWithDecision(Interval):
    """
    Wrapper around 'Interval' with extra information like user decision for this interval and problems with them.
    """
    SKIP = 1
    MERGE_NEXT = 2

    def __init__(self, interval: Interval) -> None:
        super().__init__(interval.start_time, interval.end_time, interval.prev, interval.next)
        self.events = interval.events
        # Cases:
        # - user decided (any, True)
        # - interval looks the same as decided by user so apply the same decision TODO do we need?
        self.decision: Set = None  # What user chose: SKIP, MERGE_NEXT, any number of Event-s.
        self.rules_per_event: Dict[Event, Rule] = dict()  # Rules which on this stage describes Event-s.
        self.is_user_decision = False  # Flag that decision was made by user TODO is it right?
        self.problem = None

    def get_serializable_copy(self) -> 'IntervalWithDecision':
        tmp = self.__class__.__new__(self.__class__)
        tmp.__dict__.update(self.__dict__)
        # Put True if value was specified, False otherwise.
        tmp.prev = bool(tmp.prev)
        tmp.next = bool(tmp.next)
        return tmp

    @staticmethod
    def from_serializable_copy(self, prev: 'IntervalWithDecision', nxt: 'IntervalWithDecision')\
            -> 'IntervalWithDecision':
        tmp = self.__class__.__new__(self.__class__)
        tmp.__dict__.update(self.__dict__)
        tmp.prev = prev if tmp.prev else None
        tmp.next = nxt if tmp.prev else None
        return tmp
        

    def set_user_decision(self, decision: Set[Union[int, Rule]]):
        self.decision = decision
        self.is_user_decision = True

    def set_rules_per_event(self, eventkeyhandlers_per_bucket_prefix: Dict[str, List[EventKeyHandler]]):
        for event in self.events:
            handler = find_handler_for_event(event, eventkeyhandlers_per_bucket_prefix)
            rule = None
            if not handler:
                self.rules_per_event[event] = rule
                continue
            rule, _ = handler.get_rule(event)
            self.rules_per_event[event] = rule

    @staticmethod
    def decision_item_to_str(item) -> str:
        if item == IntervalWithDecision.SKIP:
            return "skip from activities"
        elif item == IntervalWithDecision.MERGE_NEXT:
            return "merge with next interval"
        return str(item)


def _incosistent_decision_log(interval_with_decision, group_decision_obj):
    a = (IntervalWithDecision.decision_item_to_str(x) for x in group_decision_obj.decision)
    b = (IntervalWithDecision.decision_item_to_str(x) for x in interval_with_decision.decision)
    LOG.info("Inconsistency in decisions:\n"
                "  for %s decision is\n    %s\n"
                "  while for %s decision is\n    %s",
                interval_with_decision.to_str(), "\n    ".join(b),
                group_decision_obj, "\n    ".join(a))


def adjust_priorities(decisions: List[IntervalWithDecision]) -> Metrics:
    """
    Analyzes list of `IntervalWithDecision`-s and adjust inner `Rule`-s priorities in a way to satisfy decisions
    as mush as possible. Does it in-place. Reports all issues as metrics.
    :param decisions: List of `IntervalWithDecision` to correct attached rules in them. Rules are expected to be
    links on each other, not duplicates.
    :return: Metrics from corrective actions.
    """
    metrics = Metrics(
        {
            'inconsistent_decision': _incosistent_decision_log,
        },
        None
    )
    # Find contradictions and report them.
    input_rules_to_decision: Set[Tuple[Rule], IntervalWithDecision] = {}  # This map is not used further.
    for decision in decisions:
        input_rules = set(decision.rules_per_event.values())
        input_rules_tuple = tuple(sorted(input_rules, key=lambda r: r.key_pattern))
        if input_rules_tuple in input_rules_to_decision:
            prev_decision = input_rules_to_decision[input_rules_tuple].decision
            if prev_decision != decision.decision:
                _incosistent_decision_log(decision, input_rules_to_decision[input_rules_tuple])
                metrics.increment('inconsistent_decision', decision)
                # TODO metrics
                # LOG.warning(f"Contradicting decisions for input rules {input_rules_tuple}: {prev_decision} and {decision.decision}")
        else:
            input_rules_to_decision[input_rules_tuple] = decision
            metrics.increment('total_unique_intervals', decision)
    # Adjust priorities.
    rules = set()  # For metrics.
    unique_updated_rules = set()  # For metrics.
    for decision in decisions:
        input_rules = set(decision.rules_per_event.values())
        rules = rules.union(input_rules)
        non_selected_rules = input_rules.difference(decision.decision)

        # Handle SKIP and MERGE_NEXT. SKIP overwrites MERGE_NEXT.
        if IntervalWithDecision.SKIP in decision.decision:
            metrics.increment('total_decisions_to_skip', decision)
            for item in decision.decision:
                if isinstance(item, Rule):
                    item.skip = True
                    metrics.increment('total_rules_to_skip')
        elif IntervalWithDecision.MERGE_NEXT in decision.decision:
            metrics.increment('total_decisions_to_merge_with_next', decision)
            for item in decision.decision:
                if isinstance(item, Rule):
                    item.merge_next = True
                    metrics.increment('total_rules_to_merge_with_next')

        # Correct priority only if there are rules to compare with selected.
        if non_selected_rules:
            max_non_selected_priority = max(rule.priority for rule in non_selected_rules)
            for item in decision.decision:
                if isinstance(item, Rule):
                    rule = item
                    if rule.priority <= max_non_selected_priority:
                        # Update rule priority.
                        rule.priority = max_non_selected_priority + 1
                        # Update metrics with "rule priority was changed".
                        if rule not in unique_updated_rules:
                            metrics.increment('total_unique_updated_rules')
                            unique_updated_rules.add(rule)
                        metrics.increment('total_updated_rule_priority', decision)
    metrics.override('total_rules', len(rules), 0)
    return metrics
