import collections
import datetime
from typing import Dict, Any, List, Optional, Set, Callable
from activity_merger.domain.interval import Interval


Metric = collections.namedtuple('Metric', ['cnt', 'duration'])
"""
One entry in `Metrics` object.
:param cnt: Number of occurences.
:param duration: Sum of seconds.
"""


class Metrics:
    """
    Object to track metrics of actions related to `Interval`-s.
    Each metric has name, counter of occurrences and total duration.
    """

    def __init__(self, handler_per_metric: Dict[str, Callable], suppressed_problems: Set[str], **kwargs) -> None:
        self.handler_per_metric = handler_per_metric
        self.metrics: Dict[str, Metric] = dict((k, Metric(0, 0.0)) for k, v in handler_per_metric.items())
        self.suppressed_problems = suppressed_problems
        self.kwargs: Dict[str, Any] = kwargs

    def increment(self, metric_name: str, interval: Optional[Interval] = None):
        """
        Increment metric on one event with given `Interval` duration. May add new metrics and skip None intervals.
        :param metric_name: Name of metric to increment.
        :param interval: Interval to increment metric with. If `None` then metric loses duration part forever.
        """
        metric = self.metrics.get(metric_name, Metric(0, 0.0))
        self.metrics[metric_name] = Metric(
            metric.cnt + 1,
            metric.duration + interval.get_duration() if interval else 0
        )

    def increment_and_report(self, metric_name: str, interval: Interval, **kwargs):
        """
        Increment metric on one event with given `Interval` duration and reports it with specified handler.
        Rejects new metrics.
        :param metric_name: Name of metric to increment.
        :param interval: Interval to increment metric with.
        :param kwargs: Extra arguments to "handler" method.
        """
        assert metric_name in self.handler_per_metric, f"Unsupported metric name '{metric_name}'."
        self.increment(metric_name, interval)
        handler = self.handler_per_metric[metric_name]
        handler(*kwargs)

    def override(self, metric_name: str, cnt: int, duration: float):
        """
        Sets value of metric directly. Aka 'hack' - please prefer to use `increment` method(s).
        :param metric_name: Name of metric to set.
        :param cnt: Number of occurences to set.
        :param duration: Total duration in seconds. Set 0 if duration is not applicable/measurable for metric.
        """
        self.metrics[metric_name] = Metric(cnt, duration)

    def get_metric(self, metric_name: str) -> Optional[Metric]:
        """
        Returns one metric by name.
        :param metric_name: Name of metric to return.
        :return: The metric.
        """
        return self.metrics[metric_name]

    def to_str(self, is_exclude_zero: bool = True) -> List[str]:
        """
        Returns list of strings with metrics.
        :param is_exclude_zero: Flag to return not all metrics but only not 0.
        :return: Ready to print metrics in strings list.
        """
        sorted_metric_entries = sorted(self.metrics.items(), key=lambda x: x[1][1], reverse=True)
        return (f"{x[1].cnt:4} on {datetime.timedelta(seconds=x[1].duration)} - {x[0]}" for x in sorted_metric_entries
                if not is_exclude_zero or x[1].cnt > 0)
