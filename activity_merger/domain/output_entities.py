import dataclasses
import datetime
import collections
from typing import List

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
    One or few `RuleResult`-s separated as independent activity. See description per field.
    """

    start_time: datetime.datetime
    """Start time of the activity."""
    end_time: datetime.datetime
    """End time of the activity."""
    rule_results: List[RuleResult]
    """List of (dominant) events the activity consists of."""
    description: str
    """Human-friendly description of the activity."""
    duration: float
    """
    Total duration of the activity in seconds.
    Note that 'end_time - start_time' doesn't work due to possible gaps between intervals.
    """

    def __repr__(self) -> str:
        return f"{seconds_to_int_timedelta((self.duration))} "\
               f"({from_start_to_end_to_str(self)}) {self.description}"


@dataclasses.dataclass
class AnalyzerResult:
    """
    General result of one call to "analyze" method. See description per field.
    """

    activities: List[Activity]
    """List of activities."""
    rule_results_counter: collections.Counter
    """Duration of intervals per rule."""
    metrics: Metrics
    """Dictionary of metrics, where each metric is represented by number of intervals and duration."""
    raw_rule_result_debug_events: List[Event]
    """List of 'Event'-s which represents 'RuleResult' chosen for each interval."""
    final_rule_result_debug_events: List[Event]
    """List of 'Event'-s which represents 'RuleResult' after applying 'skip', 'merge_next', etc. features."""
    activity_debug_events: List[Event]
    """List of 'Event'-s which represent resulting 'Activity'-es."""
