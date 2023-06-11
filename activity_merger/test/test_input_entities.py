import unittest
from parameterized import parameterized

from . import build_datetime, build_timedelta
from ..domain.input_entities import Rule2, Event


EVENT = Event("b1", build_datetime(1), build_timedelta(1), {"foo": 1, "bar": "a", "name": "alex"})
RULE_A = Rule2("a", 3)
RULE_B = Rule2("b", 3)
RULE_1 = Rule2("a", 3)
RULE_NAME_ENDS_EX = Rule2(r"(\w+)ex", 3)

# Top level rules, per test.
MATCH_TOP_RULE = Rule2("b1", 1).with_subrules("bar", [RULE_B])

class TestRule(unittest.TestCase):

    @parameterized.expand([
        (
            "not_match_top",
            EVENT,
            Rule2("some", 1).with_subrules("bar", [RULE_A]),
            None, None
        ),
        (
            "match_top",
            EVENT,
            MATCH_TOP_RULE,
            MATCH_TOP_RULE, "b1"
        ),
        (
            "match_bottom",
            EVENT,
            Rule2("b1", 1).with_subrules("bar", [RULE_A]),
            RULE_A, "b1, a"
        ),
        (
            "description_from_regex_group",
            EVENT,
            Rule2("b1", 1).with_subrules("name", [RULE_B, RULE_NAME_ENDS_EX]),
            RULE_NAME_ENDS_EX, "b1, al"
        ),
    ])
    def test_find_rule_for_event(self, test_name: str, event: Event, rule: Rule2,
                                 expected_rule: Rule2, expected_desc: str):
        # Act
        result, desc = rule.find_rule_for_event(event)
        # Assert
        err_msg = f"'{test_name}' case wrong "
        self.assertEqual(result, expected_rule, err_msg + "rule")
        self.assertEqual(desc, expected_desc, err_msg + "description")
