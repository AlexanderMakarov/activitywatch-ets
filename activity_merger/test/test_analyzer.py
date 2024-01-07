import dataclasses
import datetime
import unittest
from typing import Dict, List, Optional

import intervaltree
from parameterized import parameterized

from activity_merger.domain.analyzer import (
    _exclude_tree_intervals, _include_tree_intervals,
    find_next_uncovered_intervals, fit_interval_to_tree_to_dont_overlap)

from ..domain.input_entities import Event, IntervalBoundaries, Strategy
from ..domain.metrics import Metric, Metrics
from ..domain.output_entities import Activity
from ..domain.strategies import ActivityByStrategy
from . import build_datetime, build_timedelta

BUCKET1 = "buck1"
BUCKET2 = "buck2"
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

METRIC1_1 = Metric(1, DELTA1S)
METRIC2_2 = Metric(2, DELTA2S)

HOUR4_00 = datetime.datetime(2023, 1, 1, 4)
HOUR5_00 = datetime.datetime(2023, 1, 1, 5)
HOUR6_00 = datetime.datetime(2023, 1, 1, 6)
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
EVENT500_600 = Event("b1", HOUR5_00, datetime.timedelta(hours=1), "5:00..6:00")
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

STRATEGY = Strategy(name="ts", bucket_prefix="ts")
ACTIVITY = ActivityByStrategy(
    id=0,
    suggested_start_time=HOUR4_00,
    suggested_end_time=HOUR4_00,
    max_start_time=HOUR4_00,
    min_end_time=HOUR4_00,
    events=[],
    density=1,
    grouping_data="gd",
    strategy=STRATEGY,
)
STRATEGY_STRICT = Strategy(name="ts", bucket_prefix="ts", in_trustable_boundaries=IntervalBoundaries.STRICT)
ACTIVITY_STRICT = dataclasses.replace(ACTIVITY, strategy=STRATEGY_STRICT)
STRATEGY_START = Strategy(name="ts", bucket_prefix="ts", in_trustable_boundaries=IntervalBoundaries.START)
ACTIVITY_START = dataclasses.replace(ACTIVITY, strategy=STRATEGY_START)
STRATEGY_END = Strategy(name="ts", bucket_prefix="ts", in_trustable_boundaries=IntervalBoundaries.END)
ACTIVITY_END = dataclasses.replace(ACTIVITY, strategy=STRATEGY_END)
STRATEGY_DIM = Strategy(name="ts", bucket_prefix="ts", in_trustable_boundaries=IntervalBoundaries.DIM)
ACTIVITY_DIM = dataclasses.replace(ACTIVITY, strategy=STRATEGY_DIM)
ACTIVITIES = [
    ActivityByStrategy(
        id=1,
        suggested_start_time=HOUR5_00,
        suggested_end_time=HOUR6_00,
        max_start_time=HOUR5_00,
        min_end_time=HOUR6_00,
        events=[EVENT500_600],
        density=0,
        grouping_data="a5-6",
        strategy=STRATEGY,
    ),
    ActivityByStrategy(
        id=2,
        suggested_start_time=HOUR7_00,
        suggested_end_time=HOUR9_00,
        max_start_time=HOUR7_00,
        min_end_time=HOUR9_00,
        events=[EVENT700_730, EVENT730_800, EVENT830_900],
        density=0,
        grouping_data="a7-9",
        strategy=STRATEGY,
    ),
    ActivityByStrategy(
        id=3,
        suggested_start_time=HOUR10_00,
        suggested_end_time=HOUR13_00,
        max_start_time=HOUR10_00,
        min_end_time=HOUR13_00,
        events=[EVENT1000_1030, EVENT1030_1130, EVENT1130_1300],
        density=0,
        grouping_data="a10-13",
        strategy=STRATEGY,
    ),
    ActivityByStrategy(
        id=4,
        suggested_start_time=HOUR14_00,
        suggested_end_time=HOUR16_00,
        max_start_time=HOUR14_00,
        min_end_time=HOUR16_00,
        events=[EVENT1400_1600],
        density=0,
        grouping_data="a14-16",
        strategy=STRATEGY,
    ),
    ActivityByStrategy(
        id=5,
        suggested_start_time=HOUR17_00,
        suggested_end_time=HOUR18_00,
        max_start_time=HOUR17_00,
        min_end_time=HOUR18_00,
        events=[EVENT1700_1800],
        density=0,
        grouping_data="a17-18",
        strategy=STRATEGY,
    ),
]
TREE = intervaltree.IntervalTree(
    [
        intervaltree.Interval(HOUR4_00, HOUR7_00),
        intervaltree.Interval(HOUR8_00, HOUR10_00),
        intervaltree.Interval(HOUR11_00, HOUR12_00),
        intervaltree.Interval(HOUR13_00, HOUR15_00),
        intervaltree.Interval(HOUR19_00, HOUR21_00),
    ]
)


class TestAnalyzer(unittest.TestCase):
    @parameterized.expand(
        [
            # activities:   5.6 7..9 10.......13  14...16 17..18
            # intervals : 4.....7 8..10 11.12 13....15           19..21
            #   expected:       7.8  10.11 12.13    15.16 17..18
            (
                "few_activities",
                ACTIVITIES,
                TREE,
                [
                    ActivityByStrategy(
                        2, HOUR7_00, HOUR8_00, HOUR7_00, HOUR8_00, [EVENT700_730, EVENT730_800], 0, "a7-9", STRATEGY
                    ),
                    ActivityByStrategy(
                        3,
                        HOUR10_00,
                        HOUR11_00,
                        HOUR10_00,
                        HOUR11_00,
                        [EVENT1000_1030, EVENT1030_1130],
                        0,
                        "a10-13",
                        STRATEGY,
                    ),
                    ActivityByStrategy(
                        1001, HOUR12_00, HOUR13_00, HOUR12_00, HOUR13_00, [EVENT1130_1300], 0, "a10-13", STRATEGY
                    ),
                    ActivityByStrategy(
                        4, HOUR15_00, HOUR16_00, HOUR15_00, HOUR16_00, [EVENT1400_1600], 0, "a14-16", STRATEGY
                    ),
                    ActivityByStrategy(
                        5, HOUR17_00, HOUR18_00, HOUR17_00, HOUR18_00, [EVENT1700_1800], 0, "a17-18", STRATEGY
                    ),
                ],
                1001,
                {
                    "activities with head cut by tt": Metric(1, float(3600)),  # a7-9
                    "activities with middle cut by tt": Metric(1, 0.0),  # a10-13
                    "activities with tail cut by tt": Metric(1, float(3600)),  # a14-16
                    "activities removed by tt": Metric(1, float(3600)),  # a5-6
                },
            ),
            (
                "one_big_activity",
                [
                    ActivityByStrategy(
                        id=0,
                        suggested_start_time=HOUR5_00,
                        suggested_end_time=HOUR18_00,
                        max_start_time=HOUR5_00,
                        min_end_time=HOUR18_00,
                        events=[
                            EVENT500_600,
                            EVENT700_730,
                            EVENT730_800,
                            EVENT830_900,
                            EVENT900_1000,
                            EVENT1000_1030,
                            EVENT1030_1130,
                            EVENT1130_1300,
                            EVENT1300_1400,
                            EVENT1400_1600,
                            EVENT1600_1700,
                            EVENT1700_1800,
                        ],
                        density=0,
                        grouping_data="a7-18",
                        strategy=STRATEGY,
                    ),
                ],
                TREE,
                [
                    ActivityByStrategy(
                        0, HOUR7_00, HOUR8_00, HOUR7_00, HOUR8_00, [EVENT700_730, EVENT730_800], 0, "a7-18", STRATEGY
                    ),
                    ActivityByStrategy(
                        1003,
                        HOUR10_00,
                        HOUR11_00,
                        HOUR10_00,
                        HOUR11_00,
                        [EVENT1000_1030, EVENT1030_1130],
                        0,
                        "a7-18",
                        STRATEGY,
                    ),
                    ActivityByStrategy(
                        1002, HOUR12_00, HOUR13_00, HOUR12_00, HOUR13_00, [EVENT1130_1300], 0, "a7-18", STRATEGY
                    ),
                    ActivityByStrategy(
                        1001,
                        HOUR15_00,
                        HOUR18_00,
                        HOUR15_00,
                        HOUR18_00,
                        [EVENT1400_1600, EVENT1600_1700, EVENT1700_1800],
                        0,
                        "a7-18",
                        STRATEGY,
                    ),
                ],
                1003,
                {
                    "activities with head cut by tt": Metric(1, float(7200)),
                    "activities with middle cut by tt": Metric(3, 0.0),  # I.e. there were 3 cuts.
                },
            ),
        ]
    )
    def test_exclude_tree_intervals(
        self,
        test_name: str,
        activities: List[Activity],
        tree: intervaltree.IntervalTree,
        expected_activities: List[Activity],
        expected_last_id: int,
        expected_metrics: Dict[str, Metric],
    ):
        self.maxDiff = None
        metrics = Metrics({})
        # Act
        actual_activities, actual_last_id = _exclude_tree_intervals(activities, tree, 1000, metrics, "tt")
        # Assert
        self.assertListEqual(expected_activities, actual_activities, "wrong activities")
        self.assertEqual(expected_last_id, actual_last_id, "wrong next id")
        self.assertDictEqual(
            {k: v for (k, v) in expected_metrics.items()},
            {k: v for (k, v) in metrics.metrics.items() if v.cnt > 0},
            "wrong metrics",
        )

    @parameterized.expand(
        [
            # activities:   5.6 7...9 10.......13  14....16 17..18
            #  intervals: 4.....7 8...10 11.12 13.....15           19..21
            # expected-------------------------------------
            #     strict:   5.6
            #      start:   5.6                    14.15
            #        end:   5.6   8.9
            #        dim:   5.6   8.9    11.12     14.15
            (
                "strict_boundaries",
                ACTIVITIES,
                IntervalBoundaries.STRICT,
                TREE,
                [
                    ActivityByStrategy(1, HOUR5_00, HOUR6_00, HOUR5_00, HOUR6_00, [EVENT500_600], 0, "a5-6", STRATEGY),
                ],
                {
                    "activities completely covered by tt": Metric(1, float(3600)),  # a5-6
                    "activities not completely covered by tt": Metric(3, float(25200)),  # all remaining
                    "activities removed because are out of tt": Metric(1, float(3600)),  # a17-18
                    "activities with IntervalBoundaries.STRICT removed by tt": Metric(3, float(25200)),
                },
            ),
            (
                "start_boundaries",
                ACTIVITIES,
                IntervalBoundaries.START,
                TREE,
                [
                    ActivityByStrategy(1, HOUR5_00, HOUR6_00, HOUR5_00, HOUR6_00, [EVENT500_600], 0, "a5-6", STRATEGY),
                    ActivityByStrategy(
                        4, HOUR14_00, HOUR15_00, HOUR14_00, HOUR15_00, [EVENT1400_1600], 0, "a14-16", STRATEGY
                    ),  # Note that duration is measured by event-s length.
                ],
                {
                    "activities completely covered by tt": Metric(1, float(3600)),  # a5-6
                    "activities not completely covered by tt": Metric(3, 25200.0),  # all remaining
                    "activities removed because are out of tt": Metric(1, float(3600)),  # a17-18
                    "activities removed with IntervalBoundaries.START started before tt": Metric(2, float(18000)),
                    "activities with tail cut by tt": Metric(1, float(3600)),  # a14-16
                },
            ),
            (
                "end_boundaries",
                ACTIVITIES,
                IntervalBoundaries.END,
                TREE,
                [
                    ActivityByStrategy(1, HOUR5_00, HOUR6_00, HOUR5_00, HOUR6_00, [EVENT500_600], 0, "a5-6", STRATEGY),
                    ActivityByStrategy(2, HOUR8_00, HOUR9_00, HOUR8_00, HOUR9_00, [EVENT830_900], 0, "a7-9", STRATEGY),
                ],
                {
                    "activities completely covered by tt": Metric(1, float(3600)),  # a5-6
                    "activities not completely covered by tt": Metric(3, 25200.0),  # all remaining
                    "activities removed because are out of tt": Metric(1, float(3600)),  # a17-18
                    "activities removed with IntervalBoundaries.END ended before tt": Metric(2, float(18000)),
                    "activities with start cut by tt": Metric(1, float(3600)),  # a8-9
                },
            ),
            (
                "dim_boundaries",
                ACTIVITIES,
                IntervalBoundaries.DIM,
                TREE,
                [
                    ActivityByStrategy(1, HOUR5_00, HOUR6_00, HOUR5_00, HOUR6_00, [EVENT500_600], 0, "a5-6", STRATEGY),
                    ActivityByStrategy(2, HOUR8_00, HOUR9_00, HOUR8_00, HOUR9_00, [EVENT830_900], 0, "a7-9", STRATEGY),
                    ActivityByStrategy(
                        3,
                        HOUR11_00,
                        HOUR12_00,
                        HOUR11_00,
                        HOUR12_00,
                        [EVENT1030_1130, EVENT1130_1300],
                        0,
                        "a10-13",
                        STRATEGY,
                    ),  # Note that duration is measured by event-s length.
                    ActivityByStrategy(
                        4, HOUR14_00, HOUR15_00, HOUR14_00, HOUR15_00, [EVENT1400_1600], 0, "a14-16", STRATEGY
                    ),  # Note that duration is measured by event-s length.
                ],
                {
                    "activities completely covered by tt": Metric(1, float(3600)),  # a5-6
                    "activities not completely covered by tt": Metric(3, 25200.0),  # all remaining
                    "activities removed because are out of tt": Metric(1, float(3600)),  # a17-18
                    "activities with IntervalBoundaries.DIM cut by tt": Metric(3, 0.0),
                },
            ),
            (
                "dim_boundaries_big_activity",
                [
                    ActivityByStrategy(
                        id=0,
                        suggested_start_time=HOUR5_00,
                        suggested_end_time=HOUR18_00,
                        max_start_time=HOUR5_00,
                        min_end_time=HOUR18_00,
                        events=[
                            EVENT500_600,
                            EVENT700_730,
                            EVENT730_800,
                            EVENT830_900,
                            EVENT900_1000,
                            EVENT1000_1030,
                            EVENT1030_1130,
                            EVENT1130_1300,
                            EVENT1300_1400,
                            EVENT1400_1600,
                            EVENT1600_1700,
                            EVENT1700_1800,
                        ],
                        density=0,
                        grouping_data="a5-18",
                        strategy=STRATEGY,
                    ),
                ],
                IntervalBoundaries.DIM,
                TREE,
                [
                    ActivityByStrategy(0, HOUR5_00, HOUR7_00, HOUR5_00, HOUR7_00, [EVENT500_600], 0, "a5-18", STRATEGY),
                    ActivityByStrategy(
                        0, HOUR8_00, HOUR10_00, HOUR8_00, HOUR10_00, [EVENT830_900, EVENT900_1000], 0, "a5-18", STRATEGY
                    ),
                    ActivityByStrategy(
                        0,
                        HOUR11_00,
                        HOUR12_00,
                        HOUR11_00,
                        HOUR12_00,
                        [EVENT1030_1130, EVENT1130_1300],
                        0,
                        "a5-18",
                        STRATEGY,
                    ),  # Note that duration is measured by event-s length.
                    ActivityByStrategy(
                        0,
                        HOUR13_00,
                        HOUR15_00,
                        HOUR13_00,
                        HOUR15_00,
                        [EVENT1300_1400, EVENT1400_1600],
                        0,
                        "a5-18",
                        STRATEGY,
                    ),  # Note that duration is measured by event-s length.
                ],
                {
                    "activities not completely covered by tt": Metric(1, 46800.0),
                    "activities with IntervalBoundaries.DIM cut by tt": Metric(4, 0.0),  # Cut parts of 1 activity.
                },
            ),
        ]
    )
    def test_include_tree_intervals(
        self,
        test_name: str,
        activities: List[ActivityByStrategy],
        boundaries: IntervalBoundaries,
        tree: intervaltree.IntervalTree,
        expected_activities: List[ActivityByStrategy],
        expected_metrics: Dict[str, Metric],
    ):
        self.maxDiff = None
        metrics = Metrics({})
        # Act
        actual: List[ActivityByStrategy] = _include_tree_intervals(
            activities=activities, boundaries=boundaries, tree=tree, metrics=metrics, name_of_tree="tt"
        )
        # Assert
        self.assertListEqual(expected_activities, actual, "wrong activities")
        self.assertDictEqual(
            {k: v for (k, v) in expected_metrics.items()},
            {k: v for (k, v) in metrics.metrics.items() if v.cnt > 0},
            "wrong metrics",
        )

    @parameterized.expand(
        [
            (
                "start:empty_result_tree",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                    ]
                ),
                intervaltree.IntervalTree(),
                None,
                HOUR9_00,
                HOUR10_00,
            ),
            (
                "middle:empty_result_tree",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR11_00),
                    ]
                ),
                intervaltree.IntervalTree(),
                HOUR10_00,
                HOUR10_00,
                HOUR11_00,
            ),
            (
                "end:empty_result_tree",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR11_00),
                    ]
                ),
                intervaltree.IntervalTree(),
                HOUR11_00,
                None,
                None,
            ),
            (
                "start:result_tree_1_interval_starts_before",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR8_00, HOUR9_00),
                    ]
                ),
                None,
                HOUR9_00,
                HOUR12_00,
            ),
            (
                "end:result_tree_1_interval_starts_before",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR8_00, HOUR9_00),
                    ]
                ),
                HOUR12_00,
                None,
                None,
            ),
            (
                "start1:result_tree_1_interval_starts_at_start",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                    ]
                ),
                None,
                HOUR10_00,
                HOUR12_00,
            ),
            (
                "start2:result_tree_1_interval_starts_at_start",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                    ]
                ),
                HOUR9_00,
                HOUR10_00,
                HOUR12_00,
            ),
            (
                "middle:result_tree_1_interval_starts_at_start",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                    ]
                ),
                HOUR11_00,
                HOUR11_00,
                HOUR12_00,
            ),
            (
                "end:result_tree_1_interval_starts_at_start",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                    ]
                ),
                HOUR12_00,
                None,
                None,
            ),
            (
                "start:result_tree_1_interval_starts_at_middle",
                intervaltree.IntervalTree(  # Multiple overlapping intervals on 9..14.
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                        intervaltree.Interval(HOUR10_00, HOUR13_00),
                        intervaltree.Interval(HOUR9_00, HOUR14_00),
                        intervaltree.Interval(HOUR11_00, HOUR13_00),
                        intervaltree.Interval(HOUR13_00, HOUR14_00),
                    ]
                ),
                intervaltree.IntervalTree(  # Covers few intervals in "candidates_tree".
                    [
                        intervaltree.Interval(HOUR10_00, HOUR11_00),
                        intervaltree.Interval(HOUR12_00, HOUR13_00),
                        intervaltree.Interval(HOUR14_00, HOUR15_00),
                    ]
                ),
                None,
                HOUR9_00,
                HOUR10_00,
            ),
            (
                "middle_before:result_tree_1_interval_starts_at_middle",
                intervaltree.IntervalTree(  # Multiple overlapping intervals on 9..14.
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                        intervaltree.Interval(HOUR10_00, HOUR13_00),
                        intervaltree.Interval(HOUR9_00, HOUR14_00),
                        intervaltree.Interval(HOUR11_00, HOUR13_00),
                        intervaltree.Interval(HOUR13_00, HOUR14_00),
                    ]
                ),
                intervaltree.IntervalTree(  # Covers 11..12 interval.
                    [
                        intervaltree.Interval(HOUR11_00, HOUR12_00),
                    ]
                ),
                HOUR10_00,
                HOUR10_00,
                HOUR11_00,
            ),
            (
                "middle_on:result_tree_1_interval_starts_at_middle",
                intervaltree.IntervalTree(  # Multiple overlapping intervals on 9..14.
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                        intervaltree.Interval(HOUR10_00, HOUR13_00),
                        intervaltree.Interval(HOUR9_00, HOUR14_00),
                        intervaltree.Interval(HOUR11_00, HOUR13_00),
                        intervaltree.Interval(HOUR13_00, HOUR14_00),
                    ]
                ),
                intervaltree.IntervalTree(  # Covers 11..12 interval.
                    [
                        intervaltree.Interval(HOUR11_00, HOUR12_00),
                    ]
                ),
                HOUR11_00,
                HOUR12_00,
                HOUR14_00,
            ),
            (
                "middle_inside:result_tree_1_interval_starts_at_middle",
                intervaltree.IntervalTree(  # Multiple overlapping intervals on 9..14.
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                        intervaltree.Interval(HOUR10_00, HOUR13_00),
                        intervaltree.Interval(HOUR9_00, HOUR14_00),
                        intervaltree.Interval(HOUR11_00, HOUR13_00),
                        intervaltree.Interval(HOUR13_00, HOUR14_00),
                    ]
                ),
                intervaltree.IntervalTree(  # Covers 11..13 interval.
                    [
                        intervaltree.Interval(HOUR11_00, HOUR13_00),
                    ]
                ),
                HOUR12_00,
                HOUR13_00,
                HOUR14_00,
            ),
            (
                "middle_after:result_tree_1_interval_starts_at_middle",
                intervaltree.IntervalTree(  # Multiple overlapping intervals on 9..14.
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                        intervaltree.Interval(HOUR10_00, HOUR13_00),
                        intervaltree.Interval(HOUR9_00, HOUR14_00),
                        intervaltree.Interval(HOUR11_00, HOUR13_00),
                        intervaltree.Interval(HOUR13_00, HOUR14_00),
                    ]
                ),
                intervaltree.IntervalTree(  # Covers 11..12 interval.
                    [
                        intervaltree.Interval(HOUR11_00, HOUR12_00),
                    ]
                ),
                HOUR12_00,
                HOUR12_00,
                HOUR14_00,
            ),
            (
                "end:result_tree_1_interval_starts_at_middle",
                intervaltree.IntervalTree(  # Multiple overlapping intervals on 9..14.
                    [
                        intervaltree.Interval(HOUR9_00, HOUR10_00),
                        intervaltree.Interval(HOUR10_00, HOUR13_00),
                        intervaltree.Interval(HOUR9_00, HOUR14_00),
                        intervaltree.Interval(HOUR11_00, HOUR13_00),
                        intervaltree.Interval(HOUR13_00, HOUR14_00),
                    ]
                ),
                intervaltree.IntervalTree(  # Covers 11..12 interval.
                    [
                        intervaltree.Interval(HOUR11_00, HOUR12_00),
                    ]
                ),
                HOUR14_00,
                None,
                None,
            ),
            (
                "start:result_tree_1_interval_starts_after",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR12_00, HOUR13_00),
                    ]
                ),
                None,
                HOUR9_00,
                HOUR12_00,
            ),
            (
                "end:result_tree_1_interval_starts_after",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR12_00, HOUR13_00),
                    ]
                ),
                HOUR12_00,
                None,
                None,
            ),
            (
                "later:result_tree_1_interval_starts_after",
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR9_00, HOUR12_00),
                    ]
                ),
                intervaltree.IntervalTree(
                    [
                        intervaltree.Interval(HOUR12_00, HOUR13_00),
                    ]
                ),
                HOUR13_00,
                None,
                None,
            ),
        ]
    )
    def test_find_next_uncovered_intervals(
        self,
        test_name,
        candidates_tree: intervaltree.IntervalTree,
        result_tree: intervaltree.IntervalTree,
        start_point: datetime.datetime,
        expected_start_point: datetime.datetime,
        expected_end_point: datetime.datetime,
    ):
        self.maxDiff = None
        # Act
        actual_start_point, actual_end_point = find_next_uncovered_intervals(
            candidates_tree=candidates_tree,
            result_tree=result_tree,
            start_point=start_point,
        )
        # Assert
        self.assertEqual(expected_start_point, actual_start_point, "wrong start_point")
        self.assertEqual(expected_end_point, actual_end_point, "wrong end_point")

    @parameterized.expand(
        [
            ("empty_tree", intervaltree.IntervalTree(), intervaltree.Interval(5, 9), intervaltree.Interval(5, 9)),
            (
                "no_overlap",
                intervaltree.IntervalTree([intervaltree.Interval(1, 4)]),
                intervaltree.Interval(5, 9),
                intervaltree.Interval(5, 9),
            ),
            (
                "partial_overlap_begin",
                intervaltree.IntervalTree([intervaltree.Interval(1, 2), intervaltree.Interval(3, 6)]),
                intervaltree.Interval(5, 9),
                intervaltree.Interval(6, 9),
            ),
            (
                "partial_overlap_end",
                intervaltree.IntervalTree([intervaltree.Interval(6, 7), intervaltree.Interval(8, 9)]),
                intervaltree.Interval(5, 9),
                intervaltree.Interval(5, 6),
            ),
            (
                "full_overlap_within",
                intervaltree.IntervalTree([intervaltree.Interval(1, 9)]),
                intervaltree.Interval(5, 8),
                None,
            ),
        ]
    )
    def test_fit_interval_to_tree_to_dont_overlap(
        self,
        test_name,
        tree: intervaltree.IntervalTree,
        interval: intervaltree.Interval,
        expected_result: Optional[intervaltree.Interval],
    ):
        # Act
        result = fit_interval_to_tree_to_dont_overlap(tree, interval)

        # Assert
        if expected_result is None:
            self.assertIsNone(result, f"Expected result is None, but got {result}")
        else:
            self.assertIsNotNone(result, f"Expected a [{expected_result.begin}, {expected_result.end}], but got None")
            self.assertEqual(
                result.begin, expected_result.begin, f"Expected begin: {expected_result.begin}, but got {result.begin}"
            )
            self.assertEqual(
                result.end, expected_result.end, f"Expected end: {expected_result.end}, but got {result.end}"
            )
