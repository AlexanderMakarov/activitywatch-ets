import unittest
import datetime
from typing import List, Dict, Tuple
from parameterized import parameterized
import intervaltree

from . import build_datetime, build_timedelta, build_intervals_linked_list
from ..domain.metrics import Metric, Metrics
from ..domain.analyzer import analyze_intervals, _exclude_tree_intervals, analyze_activities_per_strategy
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
event1B = Event(BUCKET1, DAY1, DELTA1, {"name": "b"})
event1C = Event(BUCKET1, DAY1, DELTA1, {"name": "c"})
event1D = Event(BUCKET1, DAY1, DELTA1, {"name": "d"})
event2A = Event(BUCKET2, DAY2, DELTA1, {"name": "a"})
event2B = Event(BUCKET2, DAY2, DELTA1, {"name": "b"})
event2C = Event(BUCKET2, DAY2, DELTA1, {"name": "c"})

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

HOUR7_00 = datetime.datetime(2023, 1, 1, 7)
HOUR7_30 = datetime.datetime(2023, 1, 1, 7, 30)
HOUR8_00 = datetime.datetime(2023, 1, 1, 8)
HOUR9_00 = datetime.datetime(2023, 1, 1, 9)
HOUR10_00 = datetime.datetime(2023, 1, 1, 10)
HOUR10_30 = datetime.datetime(2023, 1, 1, 10, 30)
HOUR11_00 = datetime.datetime(2023, 1, 1, 11)
HOUR11_30 = datetime.datetime(2023, 1, 1, 11, 30)
HOUR12_00 = datetime.datetime(2023, 1, 1, 12)
HOUR13_00 = datetime.datetime(2023, 1, 1, 13)
HOUR14_00 = datetime.datetime(2023, 1, 1, 14)
HOUR15_00 = datetime.datetime(2023, 1, 1, 15)
HOUR16_00 = datetime.datetime(2023, 1, 1, 16)
HOUR17_00 = datetime.datetime(2023, 1, 1, 17)
HOUR18_00 = datetime.datetime(2023, 1, 1, 18)
HOUR19_00 = datetime.datetime(2023, 1, 1, 19)
HOUR20_00 = datetime.datetime(2023, 1, 1, 20)
HOUR21_00 = datetime.datetime(2023, 1, 1, 21)
EVENT700_730 = Event("b1", HOUR7_00, datetime.timedelta(minutes=30), "7:00..7:30")
EVENT730_800 = Event("b1", HOUR7_30, datetime.timedelta(minutes=30), "7:30..8:00")
EVENT830_900 = Event("b1", HOUR8_00, datetime.timedelta(hours=1), "8:00..9:00")
EVENT900_1000 = Event("b1", HOUR9_00, datetime.timedelta(hours=1), "9:00..10:00")
EVENT1000_1030 = Event("b1", HOUR10_00, datetime.timedelta(minutes=30), "10:00..10:30")
EVENT1030_1130 = Event("b1", HOUR10_30, datetime.timedelta(hours=1), "10:30..11:30")
EVENT1130_1300 = Event("b1", HOUR11_30, datetime.timedelta(hours=1, minutes=30), "11:30..13:00")
EVENT1300_1400 = Event("b1", HOUR13_00, datetime.timedelta(hours=1), "13:00..14:00")
EVENT1400_1600 = Event("b1", HOUR14_00, datetime.timedelta(hours=2), "14:00..16:00")
EVENT1600_1700 = Event("b1", HOUR16_00, datetime.timedelta(hours=1), "16:00..17:00")
EVENT1700_1800 = Event("b1", HOUR17_00, datetime.timedelta(hours=1), "17:00..18:00")
EVENT1900_2000 = Event("b1", HOUR19_00, datetime.timedelta(hours=1), "19:00..20:00")
# activities: 7..9  10.......13  14...16 17..18
# intervals :   8...10 11.12 13....15
ACTIVITIES = [
    Activity(
        start_time=HOUR7_00,
        end_time=HOUR9_00,
        events=[EVENT700_730, EVENT730_800, EVENT830_900],
        description='a7-9',
        duration=7200,
        strategy=None,
    ),
    Activity(
        start_time=HOUR10_00,
        end_time=HOUR13_00,
        events=[EVENT1000_1030, EVENT1030_1130, EVENT1130_1300],
        description='a10-13',
        duration=10800,
        strategy=None,
    ),
    Activity(
        start_time=HOUR14_00,
        end_time=HOUR16_00,
        events=[EVENT1400_1600],
        description='a14-16',
        duration=7200,
        strategy=None,
    ),
    Activity(
        start_time=HOUR17_00,
        end_time=HOUR18_00,
        events=[EVENT1700_1800],
        description='a17-18',
        duration=3600,
        strategy=None,
    ),
    Activity(
        start_time=HOUR19_00,
        end_time=HOUR20_00,
        events=[EVENT1900_2000],
        description='a19-20',
        duration=3600,
        strategy=None,
    ),
]
TREE = intervaltree.IntervalTree([
    intervaltree.Interval(HOUR8_00, HOUR10_00),
    intervaltree.Interval(HOUR11_00, HOUR12_00),
    intervaltree.Interval(HOUR13_00, HOUR15_00),
    intervaltree.Interval(HOUR19_00, HOUR21_00),
])


class ActivityMatcher:
    expected: Activity

    def __init__(self, expected):
        self.expected = expected

    @staticmethod
    def __rule_results_to_str(activity: Activity) -> str:
        return "\n  ".join(str(x) for x in activity.events)

    def __repr__(self):
        return repr(self.expected) + "\n" + ActivityMatcher.__rule_results_to_str(self.expected)

    def __eq__(self, other):
        return self.expected.start_time == other.expected.start_time and \
               self.expected.end_time == other.expected.end_time and \
               self.expected.duration == other.expected.duration and \
               self.expected.description == other.expected.description and \
               ActivityMatcher.__rule_results_to_str(self.expected) == \
                   ActivityMatcher.__rule_results_to_str(other.expected)


class TestAnalyzer(unittest.TestCase):

    @parameterized.expand([
        (
            "highest_priority_rule_one_event",
            build_intervals_linked_list([
                (1, True, 1, [event1B]),
            ]),
            RULES_SET1,
            {
                "intervals to build activities from": METRIC1_1,
                str(rule1B): METRIC1_1,
            },
            [
                Activity(DAY1, DAY2, [
                    RuleResult(rule1B, None, "buck1, b", None),
                ], "buck1, b", DELTA1S, None),
            ],
        ),
        (
            "highest_priority_rule_merge_next_at_end",
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
                    RuleResult(rule1C, None, "buck1, c", None),
                    RuleResult(rule2B, None, "buck2, b", None),
                ], "buck1, c, buck2, b", DELTA2S, None),
            ],
        ),
    ])
    def test_analyze_intervals(self, test_name: str, interval: Interval, rules: List[Rule2],
                               expected_metrics: Dict[str, Metric], expected_activities: List[Activity]):
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
        self.assertListEqual(
            [ActivityMatcher(x) for x in analyzer_result.activities],
            [ActivityMatcher(x) for x in expected_activities],
            err_msg + "activities"
        )

    @parameterized.expand([
        # activities: 7..9  10.......13  14...16 17..18
        # intervals :   8...10 11.12 13....15
        # expected-------------------------------------
        #   on start: 7.8   10.11                17..18
        #     on end:             12.13    15.16 17..18
        #   on whole: 7.8   10.11 12.13    15.16 17..18
        # Note that activities duration is measured by events, not as "start - end".
        (
            'start_boundaries',
            ACTIVITIES,
            'start',
            TREE,
            [
                Activity(HOUR7_00, HOUR8_00, [EVENT700_730, EVENT730_800], 'a7-9', 3600, None),
                Activity(HOUR10_00, HOUR11_00, [EVENT1000_1030, EVENT1030_1130], 'a10-13', 5400, None),
                Activity(HOUR17_00, HOUR18_00, [EVENT1700_1800], 'a17-18', 3600, None),  # 'a14-16'.
            ],
            {
                'activities cut on end by tt': Metric(2, float(3600 + 5400)),  # half of 'a7-9' + part of 'a10-13'.
                'activities removed because impossible to cut on start by tt': Metric(1, 7200.0),
                'activities removed by tt': Metric(1, 3600.0),  # 'a19-20'.
            },
        ),
        (
            'end_boundaries',
            ACTIVITIES,
            'end',
            TREE,
            [
                Activity(HOUR12_00, HOUR13_00, [EVENT1130_1300], 'a10-13', 5400, None),
                Activity(HOUR15_00, HOUR16_00, [EVENT1400_1600], 'a14-16', 7200, None),  # Duration from the only event.
                Activity(HOUR17_00, HOUR18_00, [EVENT1700_1800], 'a17-18', 3600, None),
            ],
            {
                'activities cut on start by tt': Metric(2, float(5400 + 0)),  # Part of 'a10-13' and 'a14-16'.
                'activities removed because impossible to cut on end by tt': Metric(1, 7200.0),  # 'a7-9'.
                'activities removed by tt': Metric(1, 3600.0),  # 'a19-20'.
            },
        ),
        (
            'whole_boundaries',
            ACTIVITIES,
            'whole',
            TREE,
            [
                Activity(HOUR7_00, HOUR8_00, [EVENT700_730, EVENT730_800], 'a7-9', 3600, None),
                Activity(HOUR10_00, HOUR11_00, [EVENT1000_1030, EVENT1030_1130], 'a10-13', 5400, None),
                Activity(HOUR12_00, HOUR13_00, [EVENT1130_1300], 'a10-13', 5400, None),
                Activity(HOUR15_00, HOUR16_00, [EVENT1400_1600], 'a14-16', 7200, None),
                Activity(HOUR17_00, HOUR18_00, [EVENT1700_1800], 'a17-18', 3600, None),
            ],
            {
                'activities cut on start by tt': Metric(1, 0.0),  # 'a14-16' and one event remained duration.
                'activities cut on end by tt': Metric(1, 3600.0),  # 'a7-9'.
                'activities with cut out middle by tt': Metric(1, 7200.0),  # 'a10-13' remained part.
                'activities removed by tt': Metric(1, 3600.0),  # 'a19-20'.
            },
        ),
        (
            'whole_boundaries_big_activity',
            [
                Activity(
                    start_time=HOUR7_00,
                    end_time=HOUR18_00,
                    events=[EVENT700_730, EVENT730_800, EVENT830_900, EVENT900_1000, EVENT1000_1030, EVENT1030_1130,
                            EVENT1130_1300, EVENT1300_1400, EVENT1400_1600, EVENT1600_1700, EVENT1700_1800],
                    description='a7-18',
                    duration=39600,
                    strategy=None,
                ),
            ],
            'whole',
            TREE,
            [
                Activity(HOUR7_00, HOUR8_00, [EVENT700_730, EVENT730_800], 'a7-18', 3600, None),
                Activity(HOUR10_00, HOUR11_00, [EVENT1000_1030, EVENT1030_1130], 'a7-18', 5400, None),
                Activity(HOUR12_00, HOUR13_00, [EVENT1130_1300], 'a7-18', 5400, None),
                Activity(HOUR15_00, HOUR18_00, [EVENT1400_1600, EVENT1600_1700, EVENT1700_1800], 'a7-18', 14400, None),
            ],
            {
                'activities with cut out middle by tt': Metric(3, 73800.0),  # We only cut our parts from one activity.
            },
        ),
    ])
    def test_exclude_tree_intervals(self, test_name: str, activities: List[Activity], boundaries: str,
                                    tree: intervaltree.IntervalTree, 
                                    expected_activities: List[Activity], expected_metrics: Dict[str, Metric]):
        self.maxDiff = None
        metrics = Metrics({})
        # Act
        actual: List[Activity] = _exclude_tree_intervals(activities, boundaries, tree, metrics, "tt")
        # Assert
        err_msg = f"'{test_name}' case wrong "
        self.assertListEqual(actual, expected_activities, err_msg + "activities")
        self.assertDictEqual(
            {k: v for (k, v) in metrics.metrics.items() if v.cnt > 0},
            {k: v for (k, v) in expected_metrics.items()},
            err_msg + "metrics"
        )
