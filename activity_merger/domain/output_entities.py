import dataclasses
import datetime
import collections
from typing import Dict, List

from .metrics import Metrics
from .input_entities import Event, Rule2
from .interval import Interval
from ..helpers.helpers import seconds_to_int_timedelta, from_start_to_end_to_str


@dataclasses.dataclass
class RuleResult:
    """
    Structure to represent rule output like covered intervals, source event and information why all of them were
    chosen to be connected under one `Rule`.
    :param rule: `Rule` which produced this instance.
    :param event: `Event` choosen by `Rule`.
    :param description: Description of underlying time span obtained by event.
    :param intervals: List of `Interval`-s covering by this rule.
    """

    rule: Rule2
    event: Event
    description: str
    intervals: List[Interval]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(description={self.description}, rule={self.rule})"


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
        return (self.end_time - self.start_time)

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
    debug_dict: Dict[str, List[Event]]
    """Map of debug bucket name to list of events to report into."""

    def to_str(self, append_equal_intervals_longer_that: float = -1.0) -> str:
        """
        Converts not debug data into human-friendly representation.
        :param append_equal_intervals_longer_that: If equal or more than 0 then result would contain 'equal'
        (i.e. with the same description) activities longer than specified value.
        Otherwise if would be skipped in the output completely - only activities.
        :return: String with metrics, 'equal' activities if configured, activities.
        """
        desc = ""
        sorted_metrics_strings = list(self.metrics.to_strings())
        desc += "Metrics from intervals analysis (total %s):\n  %s\n" % \
                (len(sorted_metrics_strings), "\n  ".join(sorted_metrics_strings))
        # Print "less than MIN_DURATION_SEC" values from 'activity_counter'.
        if append_equal_intervals_longer_that > 0.0:
            dumb_activities = [f"{seconds_to_int_timedelta(v)} {k}" for k, v in self.events_counter.most_common()
                               if v >= append_equal_intervals_longer_that]
            desc += "There were %d 'equal' events with %d longer than %d seconds:\n  %s\n" %\
                    (len(self.events_counter), len(dumb_activities), append_equal_intervals_longer_that,
                     "\n  ".join(dumb_activities))
        # Print resulting activities as is. Order is important here.
        activities_string = "\n  ".join(str(x) for x in self.activities)
        total_duration = sum((x.duration for x in self.activities), start=datetime.timedelta()).total_seconds()
        desc += "Assembled %d activities on %s:\n  %s" % (len(self.activities), total_duration, activities_string)
        return desc

    def __repr__(self):
        return self.to_str()
