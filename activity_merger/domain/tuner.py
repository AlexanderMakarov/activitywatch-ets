from typing import List, Dict, Tuple, Set, Union
import copy
from activity_merger.domain.input_entities import Event, EventKeyHandler, Rule
from activity_merger.domain.interval import Interval
from activity_merger.domain.analyzer import get_eventkeyhandlers_per_bucket_prefix, find_handler_for_event
from activity_merger.domain.metrics import Metrics
from ..config.config import LOG


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
        self.interval.events = interval.events  # Copy events as well.
        # Cases:
        # - user decided (any, True)
        # - interval looks the same as decided by user so apply the same decision TODO do we need?
        self.decision: Set = None  # What user chose: SKIP, MERGE_NEXT, any number of Event-s.
        self.rules_per_event: Dict[Event, Rule]  # Rules which on this stage describes Event-s.
        self.is_user_decision = False  # Flag that decision was made by user TODO is it right?
        self.problem = None

    def set_user_decision(self, decision: Set[Union[int, Rule]]):
        self.decision = decision
        self.is_user_decision = True

    def set_rules_per_event(self, eventkeyhandlers_per_bucket_prefix: Dict[str, List[EventKeyHandler]]):
        for event in self.interval.events:
            handler = find_handler_for_event(event, eventkeyhandlers_per_bucket_prefix)
            rule = None
            if not handler:
                self.rules_per_event[event] = rule
                continue
            rule, _ = handler.get_rule(event)
            self.rules_per_event[event] = rule

    @staticmethod
    def decision_item_to_str(item) -> str:
        if item == Decision.SKIP:
            return "skip from activities"
        elif item == Decision.MERGE_NEXT:
            return "merge with next interval"
        return str(item)


def find_rules_per_decision(decision: Decision, eventkeyhandlers_per_bucket_prefix: Dict[str, List[EventKeyHandler]])\
        -> Set[Rule]:
    rules = []
    for event in decision.interval.events:
        handler = find_handler_for_event(event, eventkeyhandlers_per_bucket_prefix)
        if not handler:
            continue
        rule, _ = handler.get_rule(event)
        if rule:
            rules.append(rule)
    return set(sorted(rules))

def _incosistent_decision_log(decision_object, group_decision_obj):
    a = (Decision.decision_item_to_str(x) for x in group_decision_obj.decision)
    b = (Decision.decision_item_to_str(x) for x in decision_object.decision)
    LOG.info("Inconsistency in decisions:\n"
                "  for %s decision is\n    %s\n"
                "  while for %s decision is\n    %s",
                decision_object.interval.to_str(), "\n    ".join(b),
                group_decision_obj, "\n    ".join(a))


def adjust_rules(decisions: List[Decision], rules: Dict[str, List[EventKeyHandler]])\
        -> Tuple[Dict[str, List[EventKeyHandler]], Metrics]:
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
    eventkeyhandlers_per_bucket_prefix = get_eventkeyhandlers_per_bucket_prefix(rules)
    combination_rules: Dict[Tuple, List[Decision]] = {}
    metrics = Metrics({
        'inconsistent_decision', _incosistent_decision_log,
    }, None)
    for decision in decisions:
        if decision.decision:
            rules = find_rules_per_decision(decision, eventkeyhandlers_per_bucket_prefix)
            combination_rules.get(rules, []).append(decision)
    # 1b + 2 - analyze resulting groups and either print contradictions or make correction to rules.
    rules_tree = None
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


def adjust_priorities(decisions: List[Decision]):
    # Find contradictions and report them.
    input_rules_to_decision = {}
    for decision in decisions:
        input_rules = set(decision.rules_per_event.values())
        input_rules_tuple = tuple(sorted(input_rules, key=lambda r: r.key_pattern))
        if input_rules_tuple in input_rules_to_decision:
            prev_decision = input_rules_to_decision[input_rules_tuple].decision
            if prev_decision != decision.decision:
                _incosistent_decision_log(decision, input_rules_to_decision[input_rules_tuple])
                # TODO metrics
                # LOG.warning(f"Contradicting decisions for input rules {input_rules_tuple}: {prev_decision} and {decision.decision}")
        else:
            input_rules_to_decision[input_rules_tuple] = decision
    # Adjust priorities.
    for decision in decisions:
        input_rules = set(decision.rules_per_event.values())
        non_selected_rules = input_rules.difference(decision.decision)

        if Decision.SKIP in decision.decision:
            for rule in decision.decision:
                if isinstance(rule, Rule):
                    rule.skip = True

        elif Decision.MERGE_NEXT in decision.decision:
            for rule in decision.decision:
                if isinstance(rule, Rule):
                    rule.merge_next = True

        # Correct priority only if there are rules to compare with selected.
        if non_selected_rules:
            max_non_selected_priority = max(rule.priority for rule in non_selected_rules)
            for rule in decision.decision:
                if isinstance(rule, Rule):
                    rule.priority = max(rule.priority, max_non_selected_priority + 1)
    # TODO metrics
