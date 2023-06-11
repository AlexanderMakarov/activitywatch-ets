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

rule1A = Rule2(r"a", 611).skip()
rule1B = Rule2(r"b", 612)
rule1C = Rule2(r"c", 613).placeholder()
rule1END = Rule2(r".*", 601)
rule2A = Rule2(r"a", 711)
rule2B = Rule2(r"b", 712).merge_next()
rule2END = Rule2(r".*", 701)
RULES_SET1 = [
    Rule2(BUCKET1, 600).with_subrules("name", [rule1A, rule1B, rule1C, rule1END]),
    Rule2(BUCKET2, 700).with_subrules("name", [rule2A, rule2B, rule2END]),
]
METRIC1_1 = Metric(1, DELTA1S)
METRIC2_2 = Metric(2, DELTA2S)


class TestAnalyzer(unittest.TestCase):

    @parameterized.expand([
        (
            "highest_priority_rule",
            build_intervals_linked_list([
                (1, True, 1, [event1A, event1B, event1C, event1D]),
                (1, False, 1, [event2A, event2B, event2C]),
            ]),
            RULES_SET1,
            {
                "intervals to build activities from": METRIC2_2,
                str(rule1C): METRIC1_1,
                str(rule2B): METRIC1_1,
                "intervals merged to next rule": METRIC1_1,
                "intervals need to reveal rule for": METRIC1_1,
            },
            [
                Activity(DAY1, DAY2, [
                    RuleResult(rule1C, None, None, None)
                ], "buck2 bucket, c", DELTA1S),
                Activity(DAY2, DAY3, [
                    RuleResult(rule2B, None, None, None)
                ], "buck1 bucket, b", DELTA1S),
            ],
        ),
    ])
    def test_analyze_intervals(self, test_name: str, interval: Interval, rules: List[Rule2],
                               expected_metrics: Metrics, expected_activities: List[Activity]):
        self.maxDiff = None
        # Act
        analyzer_result: AnalyzerResult = analyze_intervals(interval, 0.25, rules)
        # Assert
        err_msg = f"'{test_name}' case wrong "
        self.assertDictEqual(
            {k: v for (k, v) in analyzer_result.metrics.metrics.items() if v.cnt > 0},
            {k: v for (k, v) in expected_metrics.items() if v.cnt > 0},
            err_msg + "metrics"
        )
        self.assertListEqual(analyzer_result.activities, expected_activities, err_msg + "activities")
