import dataclasses
import datetime
from typing import List

from .input_entities import Event, Rule
from .interval import Interval
from ..helpers.helpers import seconds_to_int_timedelta, from_start_to_end_to_str


@dataclasses.dataclass
class RuleResult:
    """
    Structure to represent rule output like covered intervals, source event and information why all of them were
    chosen to be connected under one `Rule`.
    :param rule: `Rule` which produced this instance.
    :param event: `Event` choosen by `Rule`.
    :param description: Description of underlying interval.
    :param intervals: List of `Interval`-s covering by this rule.
    :param values: List of `Event` data pieces which pointed on this rule.
    """

    rule: Rule
    event: Event
    description: str
    intervals: List[Interval]
    values: List[str]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rule={self.rule}, description={self.description})"


@dataclasses.dataclass
class Activity:
    """
    One or few `RuleResult`-s separated as independent activity.
    """

    start_time: datetime.datetime
    end_time: datetime.datetime
    rule_results: List[RuleResult]
    description: str
    # Note that 'end_time minus start_time' doesn't work due to possible gaps between intervals.
    duration: float

    def __repr__(self) -> str:
        return f"{seconds_to_int_timedelta((self.duration))} "\
               f"({from_start_to_end_to_str(self)}) {self.description}"
