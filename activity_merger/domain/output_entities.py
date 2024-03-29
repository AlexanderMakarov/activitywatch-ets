import collections
import dataclasses
import datetime
from typing import Dict, List, Optional

from ..helpers.helpers import from_start_to_end_to_str, seconds_to_timedelta
from .input_entities import Event
from .metrics import Metrics


@dataclasses.dataclass
class Activity:
    """
    User activity assembled basing on ActivityWatch events.
    """

    start_time: datetime.datetime
    """Start time of the activity."""
    end_time: datetime.datetime
    """End time of the activity."""
    events: List[Event]
    """List of (dominant) events the activity consists of."""
    description: str
    """Human-friendly description of the activity."""

    @property
    def duration(self) -> datetime.timedelta:
        """Duration of the activity."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return f"{self.duration} ({from_start_to_end_to_str(self.start_time, self.end_time)}) {self.description}"


@dataclasses.dataclass
class AnalyzerResult:
    """
    General result of one call to "analyze" method. See description per field.
    """

    activities: List[Activity]
    """List of activities."""
    events_counter: collections.Counter
    """Counter of seconds per event."""
    metrics: Metrics
    """Dictionary of metrics, where each metric is represented by number of intervals and duration."""
    debug_dict: Optional[Dict[str, List[Event]]]
    """Map of debug bucket name to list of events to report into."""

    def to_str(
        self, append_equal_intervals_longer_that: float = -1.0, ignore_metrics_by_substrings: Optional[List[str]] = None
    ) -> str:
        """
        Converts not debug data into human-friendly representation.
        :param append_equal_intervals_longer_that: If equal or more than 0 then result would contain 'equal'
        (i.e. with the same description) activities longer than specified value.
        Otherwise if would be skipped in the output completely - only activities.
        :param ignore_metrics_by_substrings: List of substrings to ignore metrics with them in the output.
        :return: String with metrics, 'equal' activities if configured, activities.
        """
        desc = ""
        if self.metrics:
            sorted_metrics_strings = list(self.metrics.to_strings(ignore_with_substrings=ignore_metrics_by_substrings))
            desc += "Metrics from intervals analysis (total %s):\n  %s\n" % (
                len(sorted_metrics_strings),
                "\n  ".join(sorted_metrics_strings),
            )
        # Print "less than MIN_DURATION_SEC" values from 'activity_counter'.
        if append_equal_intervals_longer_that > 0.0:
            dumb_activities = [
                f"{seconds_to_timedelta(v)} {k}"
                for k, v in self.events_counter.most_common()
                if v >= append_equal_intervals_longer_that
            ]
            desc += "There were %d 'equal' events with %d longer than %d seconds:\n  %s\n" % (
                len(self.events_counter),
                len(dumb_activities),
                append_equal_intervals_longer_that,
                "\n  ".join(dumb_activities),
            )
        # Print resulting activities as is. Order is important here.
        activities_string = "\n  ".join(str(x) for x in self.activities)
        total_duration = sum((x.duration for x in self.activities), start=datetime.timedelta()).total_seconds()
        desc += "Assembled activities:\n  %s\n---- Total %d activities on %s. ----" % (
            activities_string,
            len(self.activities),
            seconds_to_timedelta(total_duration),
        )
        return desc

    def __repr__(self):
        return self.to_str()
