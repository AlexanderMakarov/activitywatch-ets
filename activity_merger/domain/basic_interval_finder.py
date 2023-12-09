import datetime
from collections import namedtuple
import re
from typing import List, Optional, Set, Tuple

import intervaltree
import numpy as np
from sklearn.linear_model import LogisticRegression

from activity_merger.domain.input_entities import ActivityByStrategy, IntervalBoundaries, Strategy
from activity_merger.domain.metrics import Metrics
from activity_merger.helpers.event_helpers import activity_by_strategy_to_str
from activity_merger.helpers.helpers import datetime_to_time_str, from_start_to_end_to_str

from ..config.config import (
    BIFINDER_SIMPLE_DENSITY,
    BIFINDER_SIMPLE_DURATION_BETWEEN_MIN_AND_MAX,
    BIFINDER_SIMPLE_DURATION_ON_INTERSECTION_INTERVAL,
    BIFINDER_SIMPLE_START_POINT_PROXIMITY,
    LOG,
    MIN_ACTIVITY_DURATION_SEC,
)


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
        Finds 2 top candidates of "basic interval" and returns in order: top candidate, score of it,
        score of 2nd candidate.
        Assumes that len of candidates and features are equal and more than one.
        :param candidates: List of candidates where "data" property contains `ActivityByStrategy`.
        :param start_point: Time need to start search from.
        :param end_point: Time where activity should stop because next either other activity/interval is already found
        or no more data.
        :param max_duration_seconds: Maximum duration for interval.
        :param metrics: `Metrics` instance to accumulate metrics.
        :return: Tuple of (top_candidate, top_candidate_score, pre_top_candidate_score).
        'top_candidate' is never `None`.
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
            top = BIFINDER_SIMPLE_START_POINT_PROXIMITY
            proximity_sec = abs(candidate.begin - start_point).seconds
            if proximity_sec < 10:
                score += top
            elif proximity_sec < 60:
                score += top * 0.75
            elif proximity_sec < MIN_ACTIVITY_DURATION_SEC:
                score += top * 0.5
            # Reward density.
            top = BIFINDER_SIMPLE_DENSITY
            score += candidate.data.density * top
            # Reward duration on the intersected interval. Only if overlap is big enough.
            top = BIFINDER_SIMPLE_DURATION_ON_INTERSECTION_INTERVAL
            overlap_sec = candidate.overlap_size(start_point, end_point).total_seconds()
            if overlap_sec > MIN_ACTIVITY_DURATION_SEC:
                overlap_ratio = 1.0 - abs(perfect_duration_sec - overlap_sec) / perfect_duration_sec
                if overlap_ratio > 0:
                    score += top * overlap_ratio
            # Reward being in the [MIN_ACTIVITY_DURATION_SEC..max_duration_seconds].
            top = BIFINDER_SIMPLE_DURATION_BETWEEN_MIN_AND_MAX
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
        metrics.incr(f"basic intervals with {score_description} score", ba_interval.data.duration())
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
                f"basic intervals with {distance_desc} distance from other candidates", ba_interval.data.duration()
            )
        else:
            metrics.incr("basic intervals without other candidates on interval", ba_interval.data.duration())
        return (ba_interval, score, closest_candidate_score)


class FromCandidatesByProximityAndDurationBIFinder(BIFinder):
    """
    Tries to find base interval by "closest to start_point and longest" properties of candidates.
    """

    max_rewarded_start_point_proximity_sec = 120  # TODO make this configurable

    def _check_interval_starts_too_far(self, candidate_start_point, start_point: datetime.datetime):
        start_point_proximity = (candidate_start_point - start_point).total_seconds()
        return start_point_proximity > self.max_rewarded_start_point_proximity_sec

    @staticmethod
    def _calculate_score(
        interval: intervaltree.Interval,
        start_point: datetime.datetime,
        end_point: datetime.datetime,
    ) -> float:
        if not interval:
            return 0.0
        # Proximity score calculation
        proximity_score = 0.0
        if interval.begin <= start_point:
            proximity_score = 0.5
        else:
            seconds_after_start = (interval.begin - start_point).total_seconds()
            if seconds_after_start <= 120:  # Decrease score for up to 120 seconds
                proximity_score = 0.5 - 0.1 * (seconds_after_start // 60)
        # Overlap score calculation
        overlap_start = max(interval.begin, start_point)
        overlap_end = min(interval.end, end_point)
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        total_duration = (end_point - start_point).total_seconds()
        overlap_score = 0.5 * (overlap_duration / total_duration) if total_duration > 0 else 0.0
        # Total score is sum.
        return proximity_score + overlap_score

    def _find_2_longest_candidates(
        self, candidates: List[ActivityByStrategy], start_point: datetime.datetime, max_duration_seconds: float
    ):
        top = pre_top = None
        top_duration = pre_top_duration = 0
        for candidate in candidates:
            if self._check_interval_starts_too_far(candidate.begin, start_point):
                break
            duration = (candidate.end - candidate.begin).total_seconds()
            if duration > max_duration_seconds:
                # Skip too long candidates because they usually means "too broad traits to group by".
                continue
            elif duration > top_duration:
                # Update second-longest candidate
                pre_top, pre_top_duration = top, top_duration
                # Update longest candidate
                top, top_duration = candidate, duration
            elif duration > pre_top_duration:
                # Update second-longest candidate
                pre_top, pre_top_duration = candidate, duration
        return top, pre_top

    def _find_2_longest_candidates_flexible(
        self, candidates: List[ActivityByStrategy], start_point: datetime.datetime, max_duration_seconds: float
    ):
        top, pre_top = self._find_2_longest_candidates(candidates, start_point, max_duration_seconds)
        if top is None:
            # If no suitable longest candidate found then probably there is AFK interval and only very long
            # (aka too broad) intervals makes jump over it. So adjust start_point and make one more pass.
            new_start_point = None
            # Find where remained "not too broad" candidates start.
            for candidate in candidates:
                duration = (candidate.end - candidate.begin).total_seconds()
                if duration <= max_duration_seconds:
                    new_start_point = candidate.begin
                    break
            top, pre_top = self._find_2_longest_candidates(candidates, new_start_point, max_duration_seconds)
        return top, pre_top

    def find_top(
        self,
        candidates: List[intervaltree.Interval],
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
        metrics: Metrics,
    ) -> Tuple[intervaltree.Interval, float, float]:
        candidates = sorted(candidates, key=lambda x: x.begin)
        corrected_end_point = min(end_point, start_point + datetime.timedelta(max_duration_seconds))
        top_candidate = second_candidate = None
        top_candidate_score = second_candidate_score = 0.0
        top_candidate, second_candidate = self._find_2_longest_candidates_flexible(
            candidates, start_point, max_duration_seconds
        )
        if not top_candidate:
            candidates_intervals = [from_start_to_end_to_str(x.begin, x.end) for x in candidates]
            raise AssertionError(
                f"Top candidate was not found with start_point={datetime_to_time_str(start_point)}, "
                f"max_rewarded_start_point_proximity_sec={self.max_rewarded_start_point_proximity_sec}, "
                f" {len(candidates_intervals)} candidates=" + ", ".join(candidates_intervals)
            )
        top_candidate_score = self._calculate_score(top_candidate, start_point, corrected_end_point)
        second_candidate_score = self._calculate_score(second_candidate, start_point, corrected_end_point)
        return top_candidate, top_candidate_score, second_candidate_score


class JiraIdBIFinder(FromCandidatesByProximityAndDurationBIFinder):
    """
    Finds basic interval in 'candidates_tree' by looking for Jira ID in activity-by-strategy-es and trying
    to make consequent intervals by same Jira ID.
    Rollbacks to simple "closes to start_point and longest" strategy if there are no "Jira IDs" near start_point.
    """

    jira_id_pattern = r"\b[A-Z]+-[0-9]+\b"  # Regular expression for Jira ID

    def find_top(
        self,
        candidates: List[intervaltree.Interval],
        start_point: datetime.datetime,
        end_point: datetime.datetime,
        max_duration_seconds: float,
        metrics: Metrics,
    ) -> Tuple[intervaltree.Interval, float, float]:
        candidates = sorted(candidates, key=lambda x: x.begin)
        # Search Jira ID-s in candidates.
        jira_intervals = dict()  # Data is Jira ID.
        last_end_time = start_point + datetime.timedelta(seconds=self.max_rewarded_start_point_proximity_sec)
        for candidate in candidates:
            activitybs: ActivityByStrategy = candidate.data
            # 1. Search Jira ID in "grouping_data".
            found_jira_ids = re.findall(self.jira_id_pattern, str(activitybs.grouping_data.get_data()))
            # 2. Search Jira ID in the whole body.
            # TODO found_jira_ids = re.findall(self.jira_id_pattern, str(activitybs.events)))
            if found_jira_ids:
                metrics.incr("candidates with Jira ID", activitybs.duration())
                for jira_id in found_jira_ids:
                    new_interval = intervaltree.Interval(candidate.begin, candidate.end, jira_id)
                    # Merge new interval into jira_intervals if Jira ID matches.
                    same_jira_id_interval = jira_intervals.get(jira_id, None)
                    if same_jira_id_interval:
                        # Check we may merge new interval with existing (if adjacent or overlap).
                        if new_interval.begin <= same_jira_id_interval.end:
                            new_interval = intervaltree.Interval(
                                new_interval.begin, max(new_interval.end, same_jira_id_interval.end), jira_id
                            )
                            metrics.incr(
                                "adjacent candidates with same Jira ID",
                                (new_interval.end - new_interval.begin).total_seconds(),
                            )
                            jira_intervals[jira_id] = new_interval
                    elif self._check_interval_starts_too_far(candidate.begin, start_point):
                        # Candidate is very far from start_point, so we won't find good candidates anymore, stop.
                        break
                    else:
                        # Add new candidate with Jira ID.
                        jira_intervals[jira_id] = new_interval
                        last_end_time = max(last_end_time, new_interval.end)
                    # Shift last_end_time.
                    last_end_time = max(last_end_time, new_interval.end)
            elif candidate.begin > last_end_time:
                # New candidates event with Jira ID won't expand existing candidates, stop.
                break

        # If Jira ID-s were found then choose top candidates from them.
        if jira_intervals:
            # Search 2 top scored candidates.
            top_candidate = None
            top_score = pre_top_score = 0
            for candidate in jira_intervals:
                score = self._calculate_score(candidate, start_point, end_point)
                if score > top_score:
                    pre_top_score = top_score
                    top_candidate, top_score = candidate, score
            return top_candidate, top_score, pre_top_score

        # If Jira ID-s weren't found then rollback to "super" implementation.
        return super().find_top(candidates, start_point, end_point, max_duration_seconds, metrics)


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
        metrics: Metrics,
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
