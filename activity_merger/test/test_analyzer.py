import unittest
from typing import List, Dict, Tuple
from parameterized import parameterized

from . import build_datetime, build_timedelta, build_intervals_linked_list
from ..domain.metrics import Metric, Metrics
from ..domain.analyzer import analyze_intervals
from ..domain.input_entities import Rule2, Event
from ..domain.interval import Interval
from ..domain.output_entities import Activity, AnalyzerResult, RuleResult

BUCKET1 = 'buck1'
BUCKET2 = 'buck2'
DAY1 = build_datetime(1)
DAY2 = build_datetime(2)
DAY3 = build_datetime(3)
DAY4 = build_datetime(4)
DELTA1 = build_timedelta(1)
DELTA2 = build_timedelta(2)
DELTA1S = DELTA1.total_seconds()
DELTA2S = DELTA2.total_seconds()
event1A = Event(BUCKET1, DAY1, DELTA1, {"name": "a"})
event1B = Event(BUCKET1, DAY2, DELTA1, {"name": "b"})
event1C = Event(BUCKET1, DAY3, DELTA1, {"name": "c"})
event1D = Event(BUCKET1, DAY4, DELTA1, {"name": "d"})
event2A = Event(BUCKET2, DAY1, DELTA1, {"name": "a"})
event2B = Event(BUCKET2, DAY2, DELTA1, {"name": "b"})
event2C = Event(BUCKET2, DAY3, DELTA1, {"name": "c"})

rule1A = Rule2("a", 611).skip()
rule1B = Rule2("b", 612)
rule1C = Rule2("c", 613).placeholder()
rule1END = Rule2(".*", 601)
rule2A = Rule2("a", 711)
rule2B = Rule2("b", 712).merge_next()
rule2END = Rule2(".*", 701)
RULES_SET1 = [
    Rule2(BUCKET1, 600).with_subrules("name", [rule1A, rule1B, rule1C, rule1END]),
    Rule2(BUCKET2, 700).with_subrules("name", [rule2A, rule2B, rule2END]),
]


class TestAnalyzer(unittest.TestCase):

    @parameterized.expand([
        (
            "highest_priority_rule",
            build_intervals_linked_list([
                (1, True, 1, [event1A, event1B, event1C]),
                (1, False, 1, [event2A, event2B, event2C]),
            ]),
            RULES_SET1,
            AnalyzerResult(
                [
                    Activity(DAY1, DAY2, [
                        RuleResult(rule1C, event1C, "buck2 bucket, c", None)
                    ], "ac1", DELTA1S),
                    Activity(DAY2, DAY3, [
                        RuleResult(rule2B, event2B, "buck1 bucket, b", None)
                    ], "ac2", DELTA1S),
                ],
                None,
                Metrics.from_dict({
                    "intervals to build activities from": Metric(2, DELTA2S),
                    str(rule1C): Metric(1, DELTA1S),
                    str(rule2B): Metric(1, DELTA1S),
                    "intervals merged to next rule": Metric(1, DELTA1S),
                }),
                None, None, None
            ),
        ),
    ])
    def test_analyze_intervals(self, test_name: str, interval: Interval, rules: List[Rule2],
                               expected_result: AnalyzerResult):
        self.maxDiff = None
        # Act
        analyzer_result: AnalyzerResult = analyze_intervals(interval, 0.25, rules)
        # Assert
        err_msg = "\n'%s' case failed with result:\n%s" % (test_name, analyzer_result)
        # TODO compare metrics
        self.assertEqual(str(analyzer_result), str(expected_result), err_msg)
