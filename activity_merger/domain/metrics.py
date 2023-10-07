import collections
import datetime
from typing import Dict, List, Optional, Set, Callable


Metric = collections.namedtuple("Metric", ["cnt", "duration"])
"""
One entry in `Metrics` object.
:param cnt: Number of occurences.
:param duration: Sum of seconds logged for this metric.
"""


class Metrics:
    """
    Object to track metrics related to some time intervals.
    Each metric has name, counter of occurrences and total duration. Duration is optional and 0 by default.
    Also ther is an ability to suppress some metrics in "reporting" methods.
    """

    def __init__(self, handler_per_metric: Dict[str, Callable], skip_metrics: Set[str] = None):
        """
        :param handler_per_metric: Dictionary with name of metric and associated handler to call in case if
        `increment_and_call_handler` is used. Shouldn't be None.
        :param skip_metrics: Set of metric names to don't handle.
        """
        self.handler_per_metric = handler_per_metric
        self.metrics: Dict[str, Metric] = dict((k, Metric(0, 0.0)) for k, _ in handler_per_metric.items())
        self.skip_metrics = skip_metrics if isinstance(skip_metrics, set) else set()

    @staticmethod
    def from_dict(metrics: Dict[str, Metric]) -> "Metrics":
        """
        Builds `Metrics` instance directly, with specified metrics inside. Useful for tests.
        :param metrics: Dict of metrics and values expected to be presented inside.
        """
        self = Metrics({})
        self.metrics = metrics
        return self

    def incr(self, metric_name: str, duration: float = 0.0) -> Metric:
        """
        Increment metric on one event with given duration. May add new metrics and skip None intervals.
        :param metric_name: Name of metric to increment.
        :param duration: Duration in seconds to add. May be 0.
        """
        if metric_name in self.skip_metrics:
            return
        metric = self.metrics.get(metric_name, Metric(0, 0.0))
        metric = Metric(metric.cnt + 1, metric.duration + duration)
        self.metrics[metric_name] = metric
        return metric

    def incr_and_call_handler(self, metric_name: str, duration: float = 0.0, *args):
        """
        Increment metric on one event with given duration and reports it with specified handler.
        Rejects new metrics.
        :param metric_name: Name of metric to increment.
        :param duration: Duration in seconds to add. May be 0.
        :param kwargs: Extra arguments to "handler" method.
        """
        assert metric_name in self.handler_per_metric, f"Unsupported metric name '{metric_name}'."
        if self.incr(metric_name, duration):
            handler = self.handler_per_metric[metric_name]
            handler(*args)

    def override(self, metric_name: str, cnt: int, duration: float):
        """
        Overwrites value of metric directly ignoring `skip_metrics`.
        Aka 'hack' - please prefer to use `increment` method(s).
        :param metric_name: Name of metric to set.
        :param cnt: Number of occurences to set.
        :param duration: Total duration in seconds. Set 0 if duration is not applicable/measurable for metric.
        """
        self.metrics[metric_name] = Metric(cnt, duration)

    def get_metric(self, metric_name: str) -> Optional[Metric]:
        """
        Returns one metric by name.
        :param metric_name: Name of metric to return.
        :return: The metric if extists, else None.
        """
        return self.metrics.get(metric_name)

    def to_strings(
        self, is_exclude_empty: bool = True, is_exclude_duration=False, ignore_with_substrings: List[str] = None
    ) -> List[str]:
        """
        Returns generator of sorted (first by duration, next by count) metric descriptions.
        :param is_exclude_zero: Flag to return only metrics with only positive count.
        :param is_exclude_duration: Flag to don't print duration at all (useful for cases when we now that all
            metrics inside doesn't provide duration).
        :param ignore_with_substrings: List of substrings to don't return metrics with.
        :return: Ready to use generator of metrics converted to strings and sorted by duration.
        """
        sorted_metric_entries = sorted(self.metrics.items(), key=lambda x: (x[1].duration, x[1].cnt), reverse=True)

        def generate():
            for metric in sorted_metric_entries:
                if (not is_exclude_empty or metric[1].cnt > 0) and (
                    not ignore_with_substrings or not any(map(metric[0].__contains__, ignore_with_substrings))
                ):
                    yield metric

        if is_exclude_duration:
            return (f"{x[1].cnt:4} - {x[0]}" for x in generate())
        else:
            return (
                f"{x[1].cnt:4} on {str(datetime.timedelta(seconds=int(x[1].duration))).rjust(8, '0')} - {x[0]}"
                for x in generate()
            )

    def __repr__(self) -> str:
        return "\n  " + "\n  ".join(self.to_strings())
