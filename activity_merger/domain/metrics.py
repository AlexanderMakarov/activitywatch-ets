import collections
import datetime
from typing import Dict, Any, List, Set, Callable
from activity_merger.domain.interval import Interval


Metric = collections.namedtuple('Metric', ['cnt', 'duration'])


class Metrics:

    def __init__(self, handler_per_metric: Dict[str, Callable], suppressed_problems: Set[str], **kwargs) -> None:
        self.handler_per_metric = handler_per_metric
        self.metrics: Dict[str, Metric] = dict((k, Metric(0, 0.0)) for k, v in handler_per_metric)
        self.suppressed_problems = suppressed_problems
        self.kwargs: Dict[str, Any] = kwargs

    def report(self, metric_name: str, interval: Interval, **kwargs):
        assert metric_name in self.handler_per_metric, f"Unsupported metric name '{metric_name}'."
        self.increment_metric(metric_name, interval)
        handler = self.handler_per_metric[metric_name]
        handler(kwargs)

    def increment_metric(self, metric_name: str, interval: Interval):
        metric = self.metrics[metric_name]
        self.metrics[metric_name] = (metric[0] + 1, metric[1] + interval.get_duration())

    def to_str(self, is_exclude_zero: bool) -> List[str]:
        """
        Returns list of strings with metrics.
        :param is_exclude_zero: Flag to return not all metrics but only not 0.
        :return: Ready to print metrics in strings list.
        """
        sorted_metric_entries = sorted(self.metrics.items(), key=lambda x: x[1][1], reverse=True)
        return (f"{x[1].cnt:4} on {datetime.timedelta(seconds=x[1].duration)} - {x[0]}" for x in sorted_metric_entries
                if not is_exclude_zero or x[1].cnt > 0)
