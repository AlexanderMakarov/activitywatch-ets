import unittest
import itertools
import copy
from typing import Dict, List, Set, Any
from parameterized import parameterized

from ..domain.interval import Interval
from ..domain.tuner import Decision, adjust_rules, adjust_priorities
from ..domain.input_entities import EventKeyHandler, Rule
from . import build_datetime


INTERVAL = Interval(build_datetime(1), build_datetime(2))
rule1p1 = Rule("1", 1)
rule1p4 = Rule("1", 4)
rule1p4m = Rule("1", 4, merge_next=True)
rule1p4s = Rule("1", 4, skip=True)
rule2p2 = Rule("2", 2)
rule3p3 = Rule("3", 3)


def _build_decision(decisions: List[Any], input_rules: List[Rule]) -> Decision:
    decision = Decision(INTERVAL)
    rules = copy.deepcopy(input_rules)  # Block tested method to change input rules.
    decision.decision = [next(x for x in rules if d == x) if isinstance(d, Rule) else d for d in decisions]
    decision.rules_per_event = dict((i, x) for i, x in enumerate(rules))
    return decision


def _get_rules_from_decisions(decisions: List[Decision]) -> List[Rule]:
    return sorted(itertools.chain(*(x.rules_per_event.values() for x in decisions)))


def _assertion_env_to_str(actual: List[Rule], expected: List[Rule], index):
    a = (repr(r) for r in actual)
    e = (repr(r) for r in expected)
    al = [(" >" if i == index else "  ") + x for i, x in enumerate(a)]
    el = [(" >" if i == index else "  ") + x for i, x in enumerate(e)]
    return "\nActuals rules:\n" + "\n".join(al) + "\nExpected rules:\n" + "\n".join(el)


class TestTuner(unittest.TestCase):

    @parameterized.expand([
        (
            "1 decision, matches rules",
            [
                _build_decision([rule2p2], [rule1p1, rule2p2])
            ],
            [rule1p1, rule2p2],  # No corrections.
        ),
        (
            "1 decision, mismatch rules",
            [
                _build_decision([rule1p1], [rule1p1, rule2p2, rule3p3])
            ],
            [rule1p4, rule2p2, rule3p3],  # Rule1 priority raised.
        ),
        (
            "1 decision, mismatch rules, merge_next",
            [
                _build_decision([Decision.MERGE_NEXT, rule1p1], [rule1p1, rule2p2, rule3p3])
            ],
            [rule1p4m, rule2p2, rule3p3],  # Rule1 priority raised and marked to merge next.
        ),
        (
            "1 decision, mismatch rules, skip",
            [
                _build_decision([Decision.SKIP, rule1p1], [rule1p1, rule2p2, rule3p3])
            ],
            [rule1p4s, rule2p2, rule3p3],  # Rule1 priority raised and marked to skip.
        ),
    ])
    def test_adjust_priorities(self, _, decisions: List[Decision], expected: List[Rule]):
        # Act
        adjust_priorities(decisions)
        # Assert
        result = _get_rules_from_decisions(decisions)
        self.assertEqual(len(result), len(expected))
        for i, (r, e) in enumerate(zip(result, expected)):
            try:
                self.assertEqual(r.key_pattern, e.key_pattern, "Wrong key pattern")
                self.assertEqual(r.priority, e.priority, "Wrong priority")
                self.assertEqual(r.skip, e.skip, "Wrong skip")
                self.assertEqual(r.merge_next, e.merge_next, "Wrong merge next")
            except AssertionError as err:
                err.args = (err.args[0] + _assertion_env_to_str(result, expected, i), *err.args[1:])
                raise
