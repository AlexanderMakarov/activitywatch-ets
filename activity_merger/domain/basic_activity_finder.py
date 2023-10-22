from collections import namedtuple
import datetime
from typing import List, Optional, Set, Tuple
import intervaltree
from sklearn.linear_model import LogisticRegression
import numpy as np
from activity_merger.domain.input_entities import IntervalBoundaries

from activity_merger.domain.metrics import Metrics

from ..config.config import (
    LOG,
    MIN_ACTIVITY_DURATION_SEC,
)


IntervalFeatures = namedtuple(
    "IntervalFeatures",
    [
        "matches_start_point",
        "proximity_start_point",
        "matches_end_point",
        "proximity_end_point",
        "overlap_ratio",
        "is_fits_max",
        "duration_above_min",
        "density",
        "is_strict_boundaries",
        "is_start_boundaries",
        "is_end_boundaries",
        "is_dim_boundaries",
        "is_not_start_end_boundaries",
        # TODO add feature "amount of keys"
    ],
)


class BAFinder:
    """
    Dedicated finder of "basic activities-by-strategy". "Basic" it is when time interval under it fits
    good into dedicated slot and the activity-by-strategy itself may be used as a good representative of
    activity made by user during interesting time interval.
    Uses Logistic Regression (aka logit, MaxEnt) classifier inside and requires in coefficients and "intercept" values.
    """

    def __init__(self) -> None:
        self.model: LogisticRegression = None

    def set_coefs(self, coefs, intercept):
        # assert len(coefs) == len(
        #     IntervalFeatures.count
        # ), f"Wrong number of coefs {len(coefs)} it should match number of fields in IntervalFeatures."
        self.model = LogisticRegression()
        self.model.coef_ = [coefs]
        self.model.intercept_ = [intercept]

    def train(self, data: List[Tuple[IntervalFeatures, int]]):
        X = []
        y = []
        for item in data:
            X.append(item[0])
            y.append(item[1])
        self.model = LogisticRegression().fit(X, y)

    def calculate_features(
        self,
        candidates: List[intervaltree.Interval],
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: int,
    ) -> List[IntervalFeatures]:
        result: List[IntervalFeatures] = []
        min_duration_sec = MIN_ACTIVITY_DURATION_SEC
        max_possible_duration_sec = (end_point - start_point).total_seconds()
        perfect_duration_sec = min(max_duration_seconds, max_possible_duration_sec)
        for candidate in candidates:
            # Set auxiliary variables.
            boundaries: IntervalBoundaries = candidate.data.strategy.in_trustable_boundaries
            duration_sec = (candidate.end - candidate.begin).seconds
            proximity_start_sec = abs(candidate.begin - start_point).seconds
            proximity_end_sec = abs(candidate.end - end_point).seconds
            overlap_sec = candidate.overlap_size(start_point, end_point).total_seconds()
            duration_above_min_sec = duration_sec - min_duration_sec
            # Set features. Keep max score = 1.0 to simplify coefficients understanding.
            matches_start_point = 1.0 if proximity_start_sec == 0 else 0
            proximity_start_point = (
                abs(1.0 - (proximity_start_sec / max_possible_duration_sec))
                if proximity_start_sec < max_possible_duration_sec
                else 0
            )
            matches_end_point = 1.0 if proximity_end_sec == 0 else 0
            proximity_end_point = (
                abs(1.0 - (proximity_end_sec / max_possible_duration_sec))
                if proximity_end_sec < max_possible_duration_sec
                else 0
            )
            overlap_ratio = 1.0 - abs(perfect_duration_sec - overlap_sec) / perfect_duration_sec
            is_fits_max = 1.0 if duration_sec <= max_duration_seconds else 0.0
            duration_above_min = (
                1.0 - (duration_above_min_sec / perfect_duration_sec)
                if 0 < duration_above_min_sec < perfect_duration_sec
                else 0
            )
            density = candidate.data.density
            is_strict_boundaries = 1.0 if boundaries == IntervalBoundaries.STRICT else 0.0
            is_start_boundaries = 1.0 if boundaries == IntervalBoundaries.START else 0.0
            is_end_boundaries = 1.0 if boundaries == IntervalBoundaries.END else 0.0
            is_dim_boundaries = 1.0 if boundaries == IntervalBoundaries.DIM else 0.0
            is_not_start_end_boundaries = (
                1.0
                if boundaries
                not in [
                    IntervalBoundaries.START,
                    IntervalBoundaries.END,
                ]
                else 0.0
            )
            result.append(
                IntervalFeatures(
                    matches_start_point=matches_start_point,
                    proximity_start_point=proximity_start_point,
                    matches_end_point=matches_end_point,
                    proximity_end_point=proximity_end_point,
                    overlap_ratio=overlap_ratio,
                    is_fits_max=is_fits_max,
                    duration_above_min=duration_above_min,
                    density=density,
                    is_strict_boundaries=is_strict_boundaries,
                    is_start_boundaries=is_start_boundaries,
                    is_end_boundaries=is_end_boundaries,
                    is_dim_boundaries=is_dim_boundaries,
                    is_not_start_end_boundaries=is_not_start_end_boundaries,
                )
            )
        return result

    def find_top(
        self,
        candidates: List[intervaltree.Interval],
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
    ) -> Tuple[intervaltree.Interval, float, float]:
        """
        Finds 2 top candidates for the position and returns in order: top candidate, score of it, score of 2nd candidate.
        Assumes that len of candidates and features are equal and more than one.
        :param candidates: List of candidates.
        :param features: List of features ordered in the same way as candidates.
        :return: Tuple of (top_candidate, top_candidate_score, pre_top_candidate_score).
        """
        # Calculate features for all intersecting intervals.
        features = self.calculate_features(candidates, start_point, end_point, max_duration_seconds)

        # Ask model scores for "is_chosen" probability for each candidate.
        decision_values = self.model.decision_function(features)
        positive_class_probs = 1 / (1 + np.exp(-decision_values))

        # Take 2 candidates with highest score.
        top_two_indices = positive_class_probs.argsort()[-2:][::-1]
        return (
            candidates[top_two_indices[0]],
            positive_class_probs[top_two_indices[0]],
            positive_class_probs[top_two_indices[1]],
        )

    def find_basic_activity_interval(
        self,
        candidates_tree: intervaltree.IntervalTree,
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
        metrics: Metrics,
    ) -> Optional[intervaltree.Interval]:
        """
        Finds "basic" activity among tree of candidates.
        """
        candidates: List[intervaltree.Interval] = list(candidates_tree.overlap(start_point, end_point))
        if not candidates:
            return None

        ba_interval = candidates[0]
        ba_score = 1.0
        closest_candidate_score = None
        if len(candidates) > 1:
            ba_interval, ba_score, closest_candidate_score = self.find_top(
                candidates, start_point, end_point, max_duration_seconds
            )

        # Provide metric and log about new basic activity found.
        ba_score_description = self._score_to_desc(ba_score)
        metrics.incr(
            f"basic activities with {ba_score_description} score", (ba_interval.end - ba_interval.begin).total_seconds()
        )
        LOG.info("Found 'basic activity' with %.2f '%s' score: %s", ba_score, ba_score_description, ba_interval)
        # Provide metric about "how distinguishable basic activity was".
        if closest_candidate_score is not None:
            distance_desc = self._score_to_desc(ba_score - closest_candidate_score)
            metrics.incr(
                f"basic activities with {distance_desc} distance from other candidates",
                (ba_interval.end - ba_interval.begin).total_seconds(),
            )
        else:
            metrics.incr(
                "basic activities without other candidates on interval",
                (ba_interval.end - ba_interval.begin).total_seconds(),
            )
        return ba_interval

    @staticmethod
    def _score_to_desc(score: float) -> str:
        score_description = ""
        if score == 1.0:
            score_description = "highest"
        elif score > 0.6:
            score_description = "high"
        elif score > 0.3:
            score_description = "good"
        else:
            score_description = "low"
        return score_description
