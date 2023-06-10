import collections
import datetime
from typing import Dict, List, Optional, Set, Callable
from activity_merger.domain.interval import Interval


Metric = collections.namedtuple('Metric', ['cnt', 'duration'])
"""
One entry in `Metrics` object.
:param cnt: Number of occurences.
:param duration: Sum of seconds logged for this metric.
"""


class Metrics:
    """
    Object to track metrics of actions related to `Interval`-s.
    Each metric has name, counter of occurrences and total duration.
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
        self.skip_metrics = skip_metrics

    @staticmethod
    def from_dict(metrics: Dict[str, Metric]) -> 'Metrics':
        """
        Builds `Metrics` instance directly, with specified metrics inside. Useful for tests.
        :param metrics: Dict of metrics and values expected to be presented inside.
        """
        self = Metrics({})
        self.metrics = metrics
        return self

    def increment(self, metric_name: str, interval: Optional[Interval] = None) -> Metric:
        """
        Increment metric on one event with given `Interval` duration. May add new metrics and skip None intervals.
        :param metric_name: Name of metric to increment.
        :param interval: Interval to increment metric with. If `None` then metric loses duration part forever.
        """
        if self.skip_metrics and metric_name in self.skip_metrics:
            return
        metric = self.metrics.get(metric_name, Metric(0, 0.0))
        metric = Metric(
            metric.cnt + 1,
            metric.duration + interval.get_duration() if interval else 0
        )
        self.metrics[metric_name] = metric
        return metric

    def increment_and_call_handler(self, metric_name: str, interval: Interval, *args):
        """
        Increment metric on one event with given `Interval` duration and reports it with specified handler.
        Rejects new metrics.
        :param metric_name: Name of metric to increment.
        :param interval: Interval to increment metric with.
        :param kwargs: Extra arguments to "handler" method.
        """
        assert metric_name in self.handler_per_metric, f"Unsupported metric name '{metric_name}'."
        if self.increment(metric_name, interval):
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

    def to_strings(self, is_exclude_empty: bool = True) -> List[str]:
        """
        Returns generator of sorted by duration metric descriptions except `suppressed_problems`.
        :param is_exclude_zero: Flag to return only not empty metrics.
        :return: Ready to use generator of metrics converted to strings and sorted by duration.
        """
        sorted_metric_entries = sorted(self.metrics.items(), key=lambda x: x[1].duration, reverse=True)
        return (f"{x[1].cnt:4} on {datetime.timedelta(seconds=int(x[1].duration))} - {x[0]}"
                for x in sorted_metric_entries if not is_exclude_empty or x[1].cnt > 0)
