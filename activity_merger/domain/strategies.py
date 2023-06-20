from typing import Dict, List, Tuple
import dataclasses

from ..config.config import MIN_DURATION_SEC
from .input_entities import Event, Rule2, Strategy
from .metrics import Metrics
from .output_entities import Activity
from ..helpers.helpers import event_data_to_str, seconds_to_int_timedelta

from .interval import Interval, intervals_duration


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
class RuleResultsWindow:
    """
    Subsidiary class to make sliding window on `Interval`-s linked list which accumulates `RuleResult`-s directly or
    supposingly belonging to one specific `Activity`.
    """
    rule_results: List[RuleResult]
    priority: int
    description: str
    duration: float

    def to_str(self, debug=False) -> str:
        """
        Converts to string.
        :param debug: Flag to make output more detailed.
        :return: String representation of the window.
        """
        prefix = f"{seconds_to_int_timedelta(self.duration)} with priority={self.priority}"
        if debug:
            events = dict((x.event.timestamp, x.event) for x in self.rule_results)
            events_str = "\n  ".join(
                f"{seconds_to_int_timedelta(x.duration.total_seconds())} {event_data_to_str(x)}"
                for x in events.values()
            )
            return f"{prefix}, description='{self.description}' and {len(events)} events:" + "\n  " + events_str
        else:
            return f"{prefix} and {len(self.rule_results)} rule results"

    def append(self, rule_result: RuleResult):
        """
        Appends new RuleResult to the window.
        :param rule_result: RuleResult to add.
        """
        self.rule_results.append(rule_result)
        self.duration += intervals_duration(rule_result.intervals)
        # If new rule has more priority than all existing in window then update windows priority and description.
        if rule_result.rule.priority >= self.priority:
            self.priority = rule_result.rule.priority
            self.description = rule_result.description

    def to_activity(self) -> Activity:
        """
        Converts window to 'Activity'.
        :return: New 'Activity' from current window.
        """
        start_time = self.rule_results[0].intervals[0].timestamp
        end_time = self.rule_results[-1].intervals[-1].end_time
        tmp = set(x.description for x in self.rule_results)
        description = ", ".join(sorted(tmp))
        return Activity(start_time, end_time, list(self.rule_results), description, self.duration)


@dataclasses.dataclass
class ActivitiesByStrategy:
    strategy: Strategy
    activities: List[Activity]
    metrics: Metrics

    def __repr__(self) -> str:
        """
        Converts not debug data into human-friendly representation.
        :param append_equal_intervals_longer_that: If equal or more than 0 then result would contain 'equal'
        (i.e. with the same description) activities longer than specified value.
        Otherwise if would be skipped in the output completely - only activities.
        :return: String with metrics, 'equal' activities if configured, activities.
        """
        metrics = "\n    ".join(self.metrics.to_strings())
        activities = "\n    ".join(str(x) for x in self.activities)
        return str(self.strategy) + "\n  Metrics:\n    " + metrics + "\n  Activities:\n    " + activities


class StrategyHandler:

    @staticmethod
    def handle_events(strategy: Strategy, events: List[Event], metrics: Metrics) -> ActivitiesByStrategy:
        """
        Handles events for specified `Strategy`.
        """

        if strategy.in_each_event_is_activity:  # Watchdog, Outlook.
            return StrategyHandler._handle_events_event_is_activity(strategy, events, metrics)
        if strategy.in_events_density_matters:
            if strategy.in_activities_may_overlap:  # Jira, Window, Web, IDEA, VSCode.
                # TODO good to have events to group different "data" into one activity.
                return StrategyHandler._handle_events_few_sliding_windows_by_density(strategy, events, metrics)
            else:  # No examples yet.
                # TODO support rules for this case. Now there are no such Strategies.
                return StrategyHandler._handle_events_event_is_activity(strategy, events, metrics)
        else:
            if strategy.in_activities_may_overlap:  # No examples yet.
                return StrategyHandler._handle_events_few_sliding_windows(strategy, events, metrics)
            else:  # AFK
                # TODO support rules for this case.
                # Now Outlook and AKF in this category and "activity per event" is right for them.
                return StrategyHandler._handle_events_one_sliding_window(strategy, events, metrics)

    @staticmethod
    def _handle_events_event_is_activity(strategy: Strategy, events: List[Event], metrics: Metrics) -> ActivitiesByStrategy:
        # Produces Activity per each event.
        activities: List[Activity] = []
        for event in events:
            activity = Activity(event.timestamp, event.timestamp + event.duration, None, str(event.data),
                                event.duration.seconds)
            metrics.incr('activities', event.duration.seconds)
            activities.append(activity)
        return ActivitiesByStrategy(strategy, activities, metrics)

    @staticmethod
    def _make_activity_between_events(first_event: Event, last_event: Event, activities: List[Activity],\
                                      metrics: Metrics):
        """
        Makes activity starting on start of first event and ending at the end of last event.
        Next adds it to provided list and updates `Metrics` with it.
        Simplifications:
        - Doesn't add `RuleResults`.
        - As "description" uses "first event data to string".
        - Measures duration as difference between start and end time points.
        """
        start_time = first_event.timestamp
        end_time = last_event.timestamp + last_event.duration
        duration =  end_time - start_time  # TODO calculate by "each event duration".
        activity = Activity(start_time, end_time, None, str(first_event.data), duration.seconds)
        activities.append(activity)
        metrics.incr('activities', duration.seconds)

    @staticmethod
    def _handle_events_one_sliding_window(strategy: Strategy, events: List[Event], metrics: Metrics)\
            -> ActivitiesByStrategy:
        # Produces Activity for all consequint events with the same data.
        activities: List[Activity] = []
        window: List[Event] = []
        current_data = None
        for event in events:
            # If event equal to previous then just add event to window.
            if current_data == str(event.data):
                window.append(event)
                metrics.incr('consequient events with same data', event.duration.seconds)
                continue
            # Create Activity from window.
            if window:
                StrategyHandler._make_activity_between_events(window[0], window[-1], activities, metrics)
                window = []
            # Prepare next window.
            current_data = str(event.data)
            window.append(event)
        # Handle last window.
        StrategyHandler._make_activity_between_events(window[0], window[-1], activities, metrics)
        return ActivitiesByStrategy(strategy, activities, metrics)

    @staticmethod
    def _separate_events_per_windows(events: List[Event], metrics: Metrics) -> ActivitiesByStrategy:
        windows: Dict[str, List[Event]] = {}
        for event in events:
            key = str(event.data)
            window = windows.get(key, [])
            metrics.incr(f'data {key}', event.duration.seconds)
            window.append(event)
        return windows

    @staticmethod
    def _handle_events_few_sliding_windows(strategy: Strategy, events: List[Event], metrics: Metrics) -> ActivitiesByStrategy:
        # Produces Activities covering all "same data" events.
        windows: Dict[str, List[Event]] = StrategyHandler._separate_events_per_windows(events, metrics)
        # Make activities from the each window.
        activities = []
        for _, events in windows.items():
            StrategyHandler._make_activity_between_events(events[0], events[-1], activities, metrics)
        return ActivitiesByStrategy(strategy, activities, metrics)

    @staticmethod
    def _handle_events_few_sliding_windows_by_density(strategy: Strategy, events: List[Event], metrics: Metrics) -> ActivitiesByStrategy:
        # Produces not overlapping Activities basing on theirs density or data.
        windows: Dict[str, List[Event]] = StrategyHandler._separate_events_per_windows(events, metrics)
        # Analyse gaps in each window.
        gaps = []
        activities = []
        for _, events in windows.items():
            # Measure all gaps.
            gaps: List[Tuple[int, float]] = []
            prev_event = None
            for i, event in enumerate(events):
                if not prev_event:
                    continue
                gap = event.timestamp - prev_event.timestamp + prev_event.duration
                if gap > 0:
                    gaps.append((i, gap))
            if len(gaps) <= 0:
                # If there are no gaps then just create one activity from all events.
                StrategyHandler._make_activity_between_events(events[0], events[-1], activities, metrics)
                continue
            # Otherwise iterate gaps to find out those which are bigger than minimal activity duration.
            # Make activities between them.
            last_activity_event_index = 0
            for gap in gaps:
                # If gap bigger than minimal activity duration then decicide that it is separate activity.
                if gap[1] >= MIN_DURATION_SEC:
                    first_event = events[last_activity_event_index]
                    StrategyHandler._make_activity_between_events(first_event, events[gap[0]], activities, metrics)
                    last_activity_event_index = gap[0]
            # Here we have activities made up to the last big gap. Make activity from remained events.
            if last_activity_event_index < len(events) - 1:
                first_event = events[last_activity_event_index]
                StrategyHandler._make_activity_between_events(first_event, events[-1], activities, metrics)
        return ActivitiesByStrategy(strategy, activities, metrics)
