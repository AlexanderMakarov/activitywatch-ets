import datetime
from collections import namedtuple
from typing import List, Optional, Set, Tuple

import intervaltree
import numpy as np
from sklearn.linear_model import LogisticRegression

from activity_merger.domain.input_entities import IntervalBoundaries
from activity_merger.domain.metrics import Metrics
from activity_merger.helpers.event_helpers import activity_by_strategy_to_str

from ..config.config import (LOG, MIN_ACTIVITY_DURATION_SEC)


class BIFinder:
    """
    Base class to search "basic interval" for resulting activity starting from specified "start point".
    By-default finds nothing.
    """

    def find_top(
        self,
        candidates: List[intervaltree.Interval],
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
        metrics: Metrics,
    ) -> Tuple[intervaltree.Interval, float, float]:
        """
        Finds 2 top candidates of "basic interval" and returns in order: top candidate, score of it, score of 2nd candidate.
        Assumes that len of candidates and features are equal and more than one.
        :param candidates: List of candidates.
        :param features: List of features ordered in the same way as candidates.
        :return: Tuple of (top_candidate, top_candidate_score, pre_top_candidate_score).
        """
        raise NotImplementedError("Not implemented")

    def score_to_desc(self, score: float) -> str:
        """
        Converts score to human readable representation with 4 levels.
        """
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


class FromCandidateActivitiesByScoreBIFinder(BIFinder):
    """
    Finds basic interval in 'candidates_tree' with calculating score by custom features and hardcoded weights.
    """

    def find_top(
        self,
        candidates: List[intervaltree.Interval],
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
        metrics: Metrics,
    ) -> Tuple[intervaltree.Interval, float, float]:
        perfect_duration_sec = min(max_duration_seconds, (end_point - start_point).total_seconds())
        candidate_scores: List[Tuple[int, intervaltree.Interval]] = []
        for candidate in candidates:
            score: float = 0.0
            # NOTE: keep max score = 1 to translate into percentage.
            boundaries: IntervalBoundaries = candidate.data.strategy.in_trustable_boundaries
            # Reward start point proximity.
            top = 0.5
            proximity_sec = abs(candidate.begin - start_point).seconds
            if proximity_sec < 10:
                score += top
            elif proximity_sec < 60:
                score += top * 0.75
            elif proximity_sec < MIN_ACTIVITY_DURATION_SEC:
                score += top * 0.5
            # Reward density.
            top = 0.1
            score += candidate.data.density * top
            # Reward duration on the intersected interval. Only if overlap is big enough.
            top = 0.2
            overlap_sec = candidate.overlap_size(start_point, end_point).total_seconds()
            if overlap_sec > MIN_ACTIVITY_DURATION_SEC:
                overlap_ratio = 1.0 - abs(perfect_duration_sec - overlap_sec) / perfect_duration_sec
                if overlap_ratio > 0:
                    score += top * overlap_ratio
            # Reward being in the [MIN_ACTIVITY_DURATION_SEC..max_duration_seconds].
            top = 0.2
            if MIN_ACTIVITY_DURATION_SEC <= overlap_sec <= max_duration_seconds and boundaries not in [
                IntervalBoundaries.START,
                IntervalBoundaries.END,
            ]:
                score += top
            # Store score in the list.
            candidate_scores.append((score, candidate))

        # Take candidate with highest score.
        sorted_results = sorted(candidate_scores, key=lambda x: x[0], reverse=True)
        # Check that interval is valid.
        ba_interval = sorted_results[0][1]
        assert ba_interval.end > ba_interval.begin, f"Inner error with choosing as basic: {ba_interval.data}"
        score = sorted_results[0][0]
        # Provide metric and log about new basic interval found.
        score_description = self.score_to_desc(score)
        metrics.incr(
            f"basic intervals with {score_description} score", (ba_interval.end - ba_interval.begin).total_seconds()
        )
        LOG.info(
            "Found 'basic interval' with %.0f '%s' score: %s",
            score,
            score_description,
            activity_by_strategy_to_str(ba_interval.data),
        )
        # Provide metric about "how distinguishable basic interval was".
        if len(sorted_results) > 1:
            closest_candidate_score = sorted_results[1][0]
            distance_desc = self.score_to_desc(score - closest_candidate_score)
            metrics.incr(
                f"basic intervals with {distance_desc} distance from other candidates",
                (ba_interval.end - ba_interval.begin).total_seconds(),
            )
        else:
            metrics.incr(
                "basic intervals without other candidates on interval",
                (ba_interval.end - ba_interval.begin).total_seconds(),
            )
        return (ba_interval, score, closest_candidate_score)


IntervalFeatures = namedtuple(
    "IntervalFeatures",
    [
        # Interval-dependent features.
        "matches_start_point",
        "proximity_start_point",
        "matches_end_point",
        "proximity_end_point",
        "overlap_ratio",
        "is_fits_max",
        "duration_above_min",
        # Constant features.
        "density",
        "is_strict_boundaries",
        "is_start_boundaries",
        "is_end_boundaries",
        "is_dim_boundaries",
        "is_not_start_end_boundaries",
        # TODO add feature "amount of keys" (need to know max number of keys for the strategy or events)
        # TODO add features per strategy
        # TODO add feature "support from windows"
        # TODO add feature "windows application"
    ],
)


class FromCandidatesByLogisticRegressionBIFinder(BIFinder):
    """
    Dedicated finder of "basic interval". "Basic" it is when time interval under it fits
    good into dedicated slot and the activity-by-strategy itself may be used as a good representative of
    activity made by user during interesting time interval.
    Uses Logistic Regression (aka logit, MaxEnt) classifier inside and requires in coefficients and "intercept" values.
    """

    def __init__(self) -> None:
        self.model: LogisticRegression = None

    def with_coefs(self, coefs: List[np.double], intercept: np.double) -> "FromCandidatesByLogisticRegressionBIFinder":
        num_of_features = len(IntervalFeatures._fields)
        assert len(coefs) == num_of_features, (
            f"Wrong number of coefs {len(coefs)}:"
            f" it should match number of fields {num_of_features} in IntervalFeatures."
        )
        self.model = LogisticRegression()
        self.model.coef_ = np.array([coefs])
        self.model.intercept_ = np.array([intercept])
        return self

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
        max_duration_seconds: float,
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
