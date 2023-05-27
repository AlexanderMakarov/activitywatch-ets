import unittest
import itertools
import copy
from typing import List, Tuple, Any
from parameterized import parameterized


from ..domain.interval import Interval
from ..domain.tuner import IntervalWithDecision, adjust_priorities
from ..domain.input_entities import Rule
from ..domain.metrics import Metric, Metrics
from . import build_datetime


INTERVAL = Interval(build_datetime(1), build_datetime(2))
METRIC1_0 = Metric(1, 0)
METRIC1_1 = Metric(1, INTERVAL.get_duration())
METRIC2_0 = Metric(2, 0)
METRIC2_2 = Metric(2, INTERVAL.get_duration() * 2)
METRIC3_0 = Metric(3, 0)
METRIC4_0 = Metric(4, 0)
rule1p1 = Rule("1", 1)
rule1p3 = Rule("1", 3)
rule1p4 = Rule("1", 4)
rule1p5 = Rule("1", 5)
rule1p4m = Rule("1", 4, merge_next=True)
rule1p4s = Rule("1", 4, skip=True)
rule2p2 = Rule("2", 2)
rule2p3 = Rule("2", 3)
rule2p4 = Rule("2", 4)
rule2p4s = Rule("2", 4, skip=True)
rule3p2 = Rule("3", 2)
rule3p3 = Rule("3", 3)


def _build_decisions(decisions_and_input_rules: List[Tuple[List[Any], List[Rule]]])\
        -> List[IntervalWithDecision]:
    # First get all rules and make copy of them to don't modify input rules.
    rules = set(itertools.chain(*(x[1] for x in decisions_and_input_rules)))
    rules = copy.deepcopy(dict((x, x) for x in rules))
    result = []
    for entry in decisions_and_input_rules:
        decision = IntervalWithDecision(INTERVAL)
        decision.decision = [rules[d] if isinstance(d, Rule) else d
                             for d in entry[0]]
        decision.rules_per_event = dict((i, x) for i, x in enumerate(rules[r] for r in entry[1]))
        result.append(decision)
    return result


def _get_rules_from_intervals(intervals: List[IntervalWithDecision]) -> List[Rule]:
    # Get all rules from all decisions, remove duplicates (full, not by key pattern only) and sort.
    return sorted(set(itertools.chain(*(x.rules_per_event.values() for x in intervals))))


def _assertion_env_to_str(actual: List[Rule], expected: List[Rule], index):
    a = (repr(r) for r in actual)
    e = (repr(r) for r in expected)
    al = [(" >" if i == index else "  ") + x for i, x in enumerate(a)]
    el = [(" >" if i == index else "  ") + x for i, x in enumerate(e)]
    return "\nActuals rules:\n" + "\n".join(al) + "\nExpected rules:\n" + "\n".join(el)


class TestTuner(unittest.TestCase):

    @parameterized.expand([
        (
            "1 decision, points highest",
            _build_decisions([([rule2p2], [rule1p1, rule2p2])]),
            [rule1p1, rule2p2],  # No corrections.
            {'total_rules': METRIC2_0, 'total_unique_intervals': METRIC1_1},
        ),
        (
            "1 decision, points lowest",
            _build_decisions([([rule1p1], [rule1p1, rule2p2, rule3p3])]),
            [rule1p4, rule2p2, rule3p3],  # Rule1 priority ^.
            {'total_rules': METRIC3_0, 'total_unique_intervals': METRIC1_1, 'total_unique_updated_rules': METRIC1_0,
             'total_updated_rule_priority': METRIC1_1},
        ),
        (
            "1 decision, points in middle",
            _build_decisions([([rule2p2], [rule1p1, rule2p2, rule3p3])]),
            [rule1p1, rule2p4, rule3p3],  # Rule2 priority ^.
            {'total_rules': METRIC3_0, 'total_unique_intervals': METRIC1_1, 'total_unique_updated_rules': METRIC1_0,
             'total_updated_rule_priority': METRIC1_1},
        ),
        (
            "1 decision, points lowest, merge_next",
            _build_decisions([([IntervalWithDecision.MERGE_NEXT, rule1p1], [rule1p1, rule2p2, rule3p3])]),
            [rule1p4m, rule2p2, rule3p3],  # Rule1 priority ^ and marked to merge next.
            {'total_decisions_to_merge_with_next': METRIC1_1, 'total_rules': METRIC3_0,
             'total_rules_to_merge_with_next': METRIC1_0, 'total_unique_intervals': METRIC1_1,
             'total_unique_updated_rules': METRIC1_0, 'total_updated_rule_priority': METRIC1_1},
        ),
        (
            "1 decision, points lowest, skip",
            _build_decisions([([IntervalWithDecision.SKIP, rule1p1], [rule1p1, rule2p2, rule3p3])]),
            [rule1p4s, rule2p2, rule3p3],  # Rule1 ^ raised and marked to skip.
            {'total_decisions_to_skip': METRIC1_1, 'total_rules': METRIC3_0, 'total_rules_to_skip': METRIC1_0,
             'total_unique_intervals': METRIC1_1, 'total_unique_updated_rules': METRIC1_0,
             'total_updated_rule_priority': METRIC1_1},
        ),
        (
            "1 decision, points lowest, skip and merge",
            _build_decisions([(
                [IntervalWithDecision.SKIP, IntervalWithDecision.MERGE_NEXT, rule1p1],
                [rule1p1, rule2p2, rule3p3])]
            ),
            # Rule1 priority ^ and marked to skip - merge is ignored.
            [rule1p4s, rule2p2, rule3p3],
            {'total_decisions_to_skip': METRIC1_1, 'total_rules': METRIC3_0, 'total_rules_to_skip': METRIC1_0,
             'total_unique_intervals': METRIC1_1, 'total_unique_updated_rules': METRIC1_0,
             'total_updated_rule_priority': METRIC1_1},
        ),
        (
            "2 decisions, both point highest",
            _build_decisions([
                ([rule3p3], [rule2p2, rule3p3]),
                ([rule2p2], [rule1p1, rule2p2])
            ]),
            [rule1p1, rule2p2, rule3p3],  # Same rules.
            {'total_rules': METRIC3_0, 'total_unique_intervals': METRIC2_2},
        ),
        (
            "2 decisions, one points lowest, other points highest",
            _build_decisions([
                ([rule2p2], [rule2p2, rule3p3]),
                ([rule2p2], [rule1p1, rule2p2])
            ]),
            [rule1p1, rule2p4, rule3p3],  # Rule2 priority ^.
            {'total_rules': METRIC3_0, 'total_unique_intervals': METRIC2_2, 'total_unique_updated_rules': METRIC1_0,
             'total_updated_rule_priority': METRIC1_1},
        ),
        (
            "2 decisions, both point lowest",
            _build_decisions([
                ([rule2p2], [rule2p2, rule3p3]),
                ([rule1p1], [rule1p1, rule2p2])
            ]),
            [rule1p5, rule2p4, rule3p3],  # Rule2 and Rule1 priority ^.
            {'total_rules': METRIC3_0, 'total_unique_intervals': METRIC2_2, 'total_unique_updated_rules': METRIC2_0,
             'total_updated_rule_priority': METRIC2_2},
        ),
        (
            "2 decisions, same events/rules, contradicts",
            _build_decisions([
                ([rule2p2], [rule1p1, rule2p2]),
                ([rule1p1], [rule1p1, rule2p2])
            ]),
            [rule1p3, rule2p2],  # Rule1 priority ^, Rule2 is ignored because conflict.
            {'inconsistent_decision': METRIC1_1, 'total_rules': METRIC2_0, 'total_unique_intervals': METRIC1_1,
             'total_unique_updated_rules': METRIC1_0, 'total_updated_rule_priority': METRIC1_1},
        ),
        (
            "2 decisions, different events/rules, contradicts",
            _build_decisions([
                ([rule2p2], [rule1p1, rule2p2, rule3p3]),
                ([rule1p1], [rule1p1, rule2p2])
            ]),
            [rule1p5, rule2p4, rule3p3],  # Rule1 priority ^, Rule2 priority ^.
            {'total_rules': METRIC3_0, 'total_unique_intervals': METRIC2_2, 'total_unique_updated_rules': METRIC2_0,
             'total_updated_rule_priority': METRIC2_2},
        ),
        (
            "2 decisions, one skip, different events/rules, contradicts",
            _build_decisions([
                ([IntervalWithDecision.SKIP, rule2p2], [rule1p1, rule2p2, rule3p3]),
                ([rule1p1], [rule1p1, rule2p2])
            ]),
            [rule1p5, rule2p4s, rule3p3],  # Rule1 priority ^ and skip, Rule2 priority ^.
            {'total_decisions_to_skip': METRIC1_1, 'total_rules': METRIC4_0, 'total_rules_to_skip': METRIC1_0,
             'total_unique_intervals': METRIC2_2, 'total_unique_updated_rules': METRIC2_0,
             'total_updated_rule_priority': METRIC2_2},
        ),
        (
            "2 decisions, one skip, one merge, different events/rules",
            _build_decisions([
                ([IntervalWithDecision.MERGE_NEXT, rule1p1], [rule1p1, rule2p2, rule3p3]),
                ([IntervalWithDecision.SKIP, rule2p2], [rule2p2, rule3p3])
            ]),
            [rule1p4m, rule2p4s, rule3p3],  # Rule1 priority ^ and skip, Rule2 priority ^.
            {'total_decisions_to_merge_with_next': METRIC1_1, 'total_decisions_to_skip': METRIC1_1,
             'total_rules': METRIC3_0, 'total_rules_to_merge_with_next': METRIC1_0, 'total_rules_to_skip': METRIC1_0,
             'total_unique_intervals': METRIC2_2, 'total_unique_updated_rules': METRIC2_0,
             'total_updated_rule_priority': METRIC2_2},
        ),
    ])
    def test_adjust_priorities(self, _, intervals: List[IntervalWithDecision], expected_rules: List[Rule],
                               expected_metrics: Metrics):
        # Act
        result: Metrics = adjust_priorities(intervals)
        # Assert
        actual_rules = _get_rules_from_intervals(intervals)
        self.assertEqual(len(actual_rules), len(expected_rules), "Wrong number of rules - check test corecctness")
        for i, (r, e) in enumerate(zip(actual_rules, expected_rules)):
            try:
                self.assertEqual(r.key_pattern, e.key_pattern, "Wrong key pattern")
                self.assertEqual(r.priority, e.priority, "Wrong priority")
                self.assertEqual(r.skip, e.skip, "Wrong skip")
                self.assertEqual(r.merge_next, e.merge_next, "Wrong merge next")
            except AssertionError as err:
                err.args = (err.args[0] + _assertion_env_to_str(actual_rules, expected_rules, i), *err.args[1:])
                raise
        self.assertDictEqual(
            {k: v for (k, v) in result.metrics.items() if v.cnt > 0},
            {k: v for (k, v) in expected_metrics.items() if v.cnt > 0},
            "Wrong metrics"
        )