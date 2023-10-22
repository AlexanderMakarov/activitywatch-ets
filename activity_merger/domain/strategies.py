import dataclasses
import datetime
from typing import Dict, List, Set, Tuple

import intervaltree

from ..config.config import LOG, MIN_ACTIVITY_DURATION_SEC, TOO_SMALL_INTERVAL_SEC
from ..helpers.helpers import event_to_str, from_start_to_end_to_str, seconds_to_timedelta
from .input_entities import Event, IntervalBoundaries, Strategy
from .metrics import Metrics

BUCKET_AFK_PREFIX = "aw-watcher-afk"
BUCKET_STOPWATCH_PREFIX = "aw-stopwatch"
BUCKET_WINDOW_PREFIX = "aw-watcher-window"


class GroupingDescriptior:
    """
    Object representing way of grouping a list of events into a single activity.
    Is used as a key for dictionary.
    """

    def __hash__(self):
        raise NotImplementedError("__hash__ is not implemented")

    def __eq__(self, other):
        raise NotImplementedError("__eq__ is not implemented")

    def get_kv_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns a list of key-value pairs representing ActivityWatch event's 'data' parts
        which were common for the events in the relevant group of events, aka Activity.
        """
        raise NotImplementedError("get_kv_pairs is not implemented")


@dataclasses.dataclass
class ListOfPairsDescriptor(GroupingDescriptior):
    """
    Group desctriptor from a list of key-value pairs.
    """

    pairs: List[Tuple[str, str]]

    def __hash__(self):
        return hash(frozenset(self.pairs))

    def __eq__(self, other):
        if not isinstance(other, ListOfPairsDescriptor):
            return False
        return frozenset(self.pairs) == frozenset(other.pairs)

    def get_kv_pairs(self) -> List[Tuple[str, str]]:
        return self.pairs


@dataclasses.dataclass(frozen=True)
class ActivityByStrategy:
    """
    Group of events aggregated for the specific strategy.
    """

    id: int
    """Identifier for the activity-by-strategy"""
    suggested_start_time: datetime.datetime
    """Suggested start time of the activity-by-strategy."""
    suggested_end_time: datetime.datetime
    """Suggested end time of the activity-by-strategy."""
    max_start_time: datetime.datetime
    """Maximum start time of the activity-by-strategy possible. Equal or later then 'start time'."""
    min_end_time: datetime.datetime
    """Minimum end time of the activity-by-strategy possible. Equal or earlier then 'end time'."""
    events: List[Event]
    """List of (dominant) events the activity-by-strategy consists of."""
    density: float
    """
    Density of the activity-by-strategy in [0..1] measured by duration of the events inside and taking into account
    boundaries of the parent strategy.
    I.e. if strategy's 'in_trustable_boundaries' is "START" or "END" then value will be zero.
    """
    grouping_data: GroupingDescriptior
    """Object describing why enclosed events are aggregated into activity-by-strategy."""
    strategy: Strategy
    """Strategy used to create this activity-by-strategy."""

    def duration(self) -> float:
        """Returns duration from suggest start to suggest end in seconds."""
        return (self.suggested_end_time - self.suggested_start_time).total_seconds()

    def __repr__(self) -> str:
        return (
            f"{self.id:>4}: {seconds_to_timedelta(self.duration())} x{self.density:.2f},"
            f" {from_start_to_end_to_str(self.suggested_start_time, self.suggested_end_time)}"
            f" (min {from_start_to_end_to_str(self.max_start_time, self.min_end_time)}),"
            f" {len(self.events):>3} {self.strategy.name} events grouped by {self.grouping_data}."
        )


@dataclasses.dataclass
class StrategyApplyResult:
    """Result of applying one strategy on list of ActivityWatch events."""

    strategy: Strategy
    activities: List[ActivityByStrategy]
    metrics: Metrics
    last_id: int

    def to_string(self, ignore_metrics_by_substrings: List[str] = None) -> str:
        """
        Converts content into human-friendly representation.
        :param append_equal_intervals_longer_that: If equal or more than 0 then result would contain 'equal'
        (i.e. with the same description) activities longer than specified value.
        Otherwise if would be skipped in the output completely - only activities.
        :param ignore_metrics_by_substrings: List of substrings to ignore metrics with them in the output.
        :return: String with metrics and activities inside.
        """
        # Note that Metrics.to_strings append 2 spaces indent.
        metrics_strings = list(self.metrics.to_strings(ignore_with_substrings=ignore_metrics_by_substrings))
        metrics = "\n  ".join(metrics_strings)
        activities = "\n    ".join(str(x) for x in sorted(self.activities, key=lambda x: x.suggested_start_time))
        return (
            f"{self.strategy}\n  Metrics ({len(metrics_strings)}):\n  {metrics}"
            f"\n  ActivityByStrategy-es ({len(self.activities)}):\n    {activities}"
        )

    def __repr__(self) -> str:
        return self.to_string()


def cut_event(
    event: Event,
    tree: intervaltree.IntervalTree,
    boundaries: IntervalBoundaries,
    tree_name: str,
    metrics: Metrics,
    strategy_name: str = None,
    is_fail_incompatible: bool = False,
) -> Event:
    """
    Cuts event to be on the intervals in the given tree taking into account boundaries.
    If boundaries and AFK are incompatible then either returns None or raises ValueError. List of incompatible cases:
    - "strict" boundaries and there is no overlaps in the tree at least for one piece of event.
    - "start" boundaries and there is no overlapping with event start.
    - "end" boundaries and there is no overlapping with event end.
    """
    event_start = event.timestamp
    event_end = event_start + event.duration

    # Get overlapping intervals
    overlaps: List[intervaltree.Interval] = sorted(tree[event_start:event_end])
    metrics.incr(f"events overlapped by {len(overlaps)} {tree_name} events", event.duration.total_seconds())
    if not overlaps:
        return event

    first_overlap = overlaps[0]
    last_overlap = overlaps[-1]

    # Handle according to boundaries
    if boundaries == IntervalBoundaries.STRICT:
        event_interval = intervaltree.Interval(event_start, event_end, event)
        for overlap in overlaps:
            # Check that all overlaps are bigger than the event.
            if not overlap.contains_interval(event_interval):
                if is_fail_incompatible:
                    raise ValueError(
                        f"Strategy '{strategy_name}' boundaries '{boundaries}' are incompatible with AFK events for "
                        + event_to_str(event)
                    )
                else:
                    return None
    elif boundaries == IntervalBoundaries.START:
        if first_overlap.begin > event_start:
            if is_fail_incompatible:
                raise ValueError(
                    f"Strategy '{strategy_name}' boundaries '{boundaries}' are incompatible with AFK events for "
                    + event_to_str(event)
                )
            else:
                return None
        event_end = min(first_overlap.end, event_end)
    elif boundaries == IntervalBoundaries.END:
        if last_overlap.end < event_end:
            if is_fail_incompatible:
                raise ValueError(
                    f"Strategy '{strategy_name}' boundaries '{boundaries}' are incompatible with AFK events for "
                    + event_to_str(event)
                )
            else:
                return None
        event_start = max(first_overlap.begin, event_start)
    elif boundaries == IntervalBoundaries.DIM:
        # Just use the first overlap to cut event by.
        event_start = max(first_overlap.begin, event_start)
        event_end = min(first_overlap.end, event_end)

    return Event(bucket_id=event.bucket_id, timestamp=event_start, duration=event_end - event_start, data=event.data)


def calculate_interval_density(
    events: List[Event], start_time: datetime.datetime, end_time: datetime.datetime
) -> float:
    """
    Calculates density of interval by the given events and bounds.
    Assumes that start_time and end_time are placed in the range of some events, probably on the first and last
    accordingly. Also events are sorted in time order. And start_time less than end_time.
    """
    duration: float = 0
    for event in events:
        event_end = event.timestamp + event.duration
        if event_end > start_time and event.timestamp < end_time:
            duration += (min(event_end, end_time) - max(event.timestamp, start_time)).total_seconds()
    return duration / (end_time - start_time).total_seconds()


class InStrategyPropertiesHandler:
    """
    Statefull object to convert list of events by building `ActivitiesByStrategy` from them
    using "in" properties of relevant `Strategy`.
    """

    def __init__(self):
        self.current_strategy: Strategy = None
        self.current_strategy_skip_kv = []
        self.current_metrics: Metrics = None
        self.current_events: List[Event] = []
        self.afk_tree: intervaltree.IntervalTree = None
        self.window_events: List[Event] = None
        self.current_window_tree: intervaltree.IntervalTree = None
        self.current_id: int = 0

    @staticmethod
    def _build_tree(events: List[Event]) -> intervaltree.IntervalTree:
        tree = intervaltree.IntervalTree()
        for event in events:
            tree.addi(event.timestamp, event.timestamp + event.duration, event)
        return tree

    def _transform_event(self, event: Event) -> Event:
        """
        Both checks that event may be used for building `ActivityByStrategy` and cuts it accordingly to
        "in_only_not_afk" and "in_only_if_window_app" parameters of the strategy.
        """
        # Check if need to skip event because data kv.
        metrics = self.current_metrics
        for k, v in self.current_strategy_skip_kv:
            if event.data.get(k) == v:
                metrics.incr(f"events skipped because have {k}={v}", event.duration.total_seconds())
                return None
        # If need then cut event by AFK.
        strategy = self.current_strategy
        if strategy.in_only_not_afk:
            # Cut event by to be only in "not-AFK" intervals.
            initial_duration = event.duration
            event = cut_event(
                event, self.afk_tree, strategy.in_trustable_boundaries, "AFK", self.current_metrics, strategy.name, True
            )
            assert event.duration <= initial_duration, f"Issue with chopping by 'AFK' event: {event_to_str(event)}"
            metrics.incr("events chopped by AFK", (initial_duration - event.duration).total_seconds())
        # If need then cut event by windows watcher "app=" events.
        if strategy.in_only_if_window_app:
            # Cut event by to be only in "relevant app active" intervals.
            initial_duration = event.duration
            event = cut_event(
                event,
                self.current_window_tree,
                strategy.in_trustable_boundaries,
                "relevant app active",
                metrics,
                strategy.name,
                True,
            )
            assert (
                event.duration <= initial_duration
            ), f"Issue with chopping by 'relevant app active' event: {event_to_str(event)}"
            metrics.incr("events chopped by 'relevant app active'", (initial_duration - event.duration).total_seconds())
        return event

    def _check_wrong_duration(self, duration: datetime.timedelta) -> bool:
        duration_sec = duration.total_seconds()
        if duration_sec <= TOO_SMALL_INTERVAL_SEC:
            self.current_metrics.incr(
                f"activity-by-strategy-es not created because interval < {TOO_SMALL_INTERVAL_SEC} seconds",
                duration_sec,
            )
            return True
        return False

    def handle_events(self, strategy: Strategy, events: List[Event], metrics: Metrics) -> StrategyApplyResult:
        """
        Handles list of events for the specified `Strategy` to provide `ActivitiesByStrategy` instance.
        """

        # Fill up "current" properties.
        self.current_strategy = strategy
        self.current_metrics = metrics
        self.current_events = events
        self.current_strategy_skip_kv = []  # Reset from the previous strategy.
        if strategy.in_skip_key_values:
            self.current_strategy_skip_kv = list(strategy.in_skip_key_values.items())

        # Copy "standard watchers" events.
        if strategy.bucket_prefix.startswith(BUCKET_AFK_PREFIX):
            # Build tree from "not AFK" events here because only these events will be used.
            tree = intervaltree.IntervalTree()
            for event in events:
                status = event.data["status"]
                if status == "not-afk":
                    tree.addi(event.timestamp, event.timestamp + event.duration, event)
                    metrics.incr("not-afk events", event.duration.total_seconds())
                else:
                    metrics.incr("afk events", event.duration.total_seconds())
            self.afk_tree = tree
        if strategy.bucket_prefix.startswith(BUCKET_WINDOW_PREFIX):
            # Just save events because for each strategy will be specified different events.
            self.window_events = events

        # Build "window_tree" if need for the strategy.
        if strategy.in_only_if_window_app:
            tree = intervaltree.IntervalTree()
            for event in self.window_events:
                app_name = event.data["app"]
                if app_name in strategy.in_only_if_window_app:
                    tree.addi(event.timestamp, event.timestamp + event.duration, event)
            self.current_window_tree = tree

        # TODO (impr) refactor to pluggable handlers.
        # Hanldle events depending on strategy parameters.
        if strategy.in_each_event_is_activity:  # Watchdog, Outlook.
            return self._convert_each_event_into_activitybs()
        if strategy.in_events_density_matters:
            if strategy.in_activities_may_overlap:  # Jira, Window, Web, IDEA, VSCode, Git.
                return self._aggregate_events_with_few_sliding_windows_by_density()
            else:  # No examples yet.
                return self._aggregate_events_with_one_sliding_window()
        else:
            if strategy.in_activities_may_overlap:  # No examples yet.
                return self._aggregate_events_with_few_sliding_windows()
            else:  # AFK
                return self._aggregate_events_with_one_sliding_window()

    def _convert_each_event_into_activitybs(self) -> StrategyApplyResult:
        # Produces Activity per each event.
        activities: List[ActivityByStrategy] = []
        density = (
            0
            if self.current_strategy.in_trustable_boundaries in [IntervalBoundaries.START, IntervalBoundaries.END]
            else 1
        )
        for event in self.current_events:
            event = self._transform_event(event)
            if event is None:
                continue
            end_time = event.timestamp + event.duration
            if self._check_wrong_duration(end_time - event.timestamp):
                continue
            self.current_id += 1
            activitybs = ActivityByStrategy(
                id=self.current_id,
                suggested_start_time=event.timestamp,
                suggested_end_time=end_time,
                max_start_time=event.timestamp,
                min_end_time=end_time,
                events=[event],
                density=density,
                grouping_data=ListOfPairsDescriptor([(k, str(v)) for k, v in event.data.items()]),
                strategy=self.current_strategy,
            )
            self.current_metrics.incr("activities", event.duration.seconds)
            activities.append(activitybs)
        return StrategyApplyResult(self.current_strategy, activities, self.current_metrics, self.current_id)

    def _make_activitybs_between_events(
        self,
        events: List[Event],
        grouping_data: GroupingDescriptior,
        activities: List[ActivityByStrategy],
    ):
        """
        Makes activity-by-strategy starting on start of first event and ending at the end of last event.
        Next adds it to provided list and updates `Metrics` with it.
        """
        # Use start of first event and end of last event as "suggested" bounds of the activity-by-strategy.
        start_time = events[0].timestamp
        end_time = events[-1].timestamp + events[-1].duration
        # Skip too small intervals. Don't do it on "suggested" values because they may provide duration=0.
        if self._check_wrong_duration(end_time - start_time):
            return
        # For max and min time check 'in_trustable_boundaries' and put start/end time of first/last event respectively.
        # Note that for STRICT and DIM boundaries max and min time points aren't used.
        # Because any overlap is either removes activity-by-strategy or splits it.
        max_start_time = start_time
        min_end_time = end_time
        in_trustable_boundaries = self.current_strategy.in_trustable_boundaries
        density_zero = False
        if in_trustable_boundaries is IntervalBoundaries.END:
            max_start_time += events[0].duration
            density_zero = True
        elif in_trustable_boundaries is IntervalBoundaries.START:
            min_end_time = events[-1].timestamp
            density_zero = True
        # Make a new activity-by-strategy. Increment ID-s counter.
        self.current_id += 1
        activitybs = ActivityByStrategy(
            id=self.current_id,
            suggested_start_time=start_time,
            suggested_end_time=end_time,
            max_start_time=max_start_time,
            min_end_time=min_end_time,
            events=events,
            density=0 if density_zero else calculate_interval_density(events, start_time, end_time),
            grouping_data=grouping_data,
            strategy=self.current_strategy,
        )
        activities.append(activitybs)
        self.current_metrics.incr("activity-by-strategy-es", activitybs.duration())

    def _aggregate_events_with_one_sliding_window(self) -> StrategyApplyResult:
        # Produces Activity for all consequint events with the same data.
        activities: List[ActivityByStrategy] = []
        window: List[Event] = []
        # Prepare variable to store keys in event's 'data' by which search similar events.
        window_kv_pairs: Set[Tuple[str, str]] = set()
        # Start iterate over events with collecting windows.
        strategy = self.current_strategy
        metrics = self.current_metrics
        for event in self.current_events:
            event = self._transform_event(event)
            if event is None:
                continue
            kv_pairs: Set[Tuple[str, str]] = set()
            if strategy.in_group_by_keys:
                for key_tuple in strategy.in_group_by_keys:
                    if all(key in event.data for key in key_tuple):
                        kv_pairs.update(
                            (
                                k,
                                v,
                            )
                            for k, v in event.data.items()
                            if k in key_tuple
                        )
                        break
            else:
                kv_pairs.update(
                    (
                        k,
                        v,
                    )
                    for k, v in event.data.items()
                )
            # If kv_pairs wasn't found then it is either warning about misconfiguration or warning about bad event data.
            if not kv_pairs:
                metrics.incr("events without data containing any in_group_by_keys", event.duration.total_seconds())
                continue
            # If event has keys and corresponding values equal to window's then add event to the window.
            if kv_pairs == window_kv_pairs:
                window.append(event)
                metrics.incr("consecutive events with same data", event.duration.seconds)
                continue
            # Otherwise if window exists then create Activity from it.
            if window:
                window_key = ListOfPairsDescriptor(list(window_kv_pairs))
                self._make_activitybs_between_events(
                    window,
                    window_key,
                    activities,
                )
                window = []
            # In any case prepare next window.
            window_kv_pairs = kv_pairs
            window.append(event)
        # Handle last window.
        if window:
            self._make_activitybs_between_events(
                window,
                ListOfPairsDescriptor(list(window_kv_pairs)),
                activities,
            )
        return StrategyApplyResult(self.current_strategy, activities, self.current_metrics, self.current_id)

    def _add_event_to_window(
        self, event: Event, key: GroupingDescriptior, windows: Dict[GroupingDescriptior, List[Event]]
    ):
        window = windows.setdefault(key, [])
        self.current_metrics.incr(f"events with data {key}", event.duration.seconds)
        window.append(event)

    def _separate_events_per_windows(self, events: List[Event]) -> Dict[GroupingDescriptior, List[Event]]:
        """
        Separates list of events into windows basing on the data inside and 'group_by_keys'.
        :param events: List of events to separate.
        :param group_by_keys: Set of keys in event's "data" field to use for making windows.
        :return: Resulting windows with keys equal to event's "data" field values chosen as window identifiers
        and values as correspondings lists of event's.
        """
        # Collect all possible windows.
        windows: Dict[GroupingDescriptior, List[Event]] = {}
        strategy = self.current_strategy
        metrics = self.current_metrics
        if strategy.in_group_by_keys:
            for event in events:
                event = self._transform_event(event)
                if event is None:
                    continue
                # If way to make windows is specified they make window per each group of keys.
                event_data = event.data
                added_times = 0
                for key_tuple in strategy.in_group_by_keys:
                    try:
                        window_key = ListOfPairsDescriptor([(key, str(event_data[key])) for key in key_tuple])
                        self._add_event_to_window(event, window_key, windows)
                        added_times += 1
                    except KeyError:
                        continue  # There is no such key.
                # If wasn't added then it is either warning about misconfiguration or warning about bad event data.
                if added_times > 0:
                    metrics.incr(
                        f"events with data containing {added_times} in_group_by_keys", event.duration.total_seconds()
                    )
        else:
            # If way to make windows is not specified then build window key as tuple of data key-value pairs.
            for event in events:
                event = self._transform_event(event)
                if event is None:
                    continue
                event_data = event.data
                window_key = ListOfPairsDescriptor([(k, str(v)) for k, v in event_data])
                self._add_event_to_window(event, window_key, windows)

        # Some windows may have the same interval/events because differnt "descriptor"-s build from the same events.
        # Need to remove such duplicates with leaving windows with maximum keys in "descriptor".
        # If length of keys is the same then it is some misconfiguration in "in_group_by_keys".
        windows_with_interval: Dict[Tuple[datetime.datetime, datetime.timedelta], GroupingDescriptior] = {}
        window_events: List[Event]
        for descriptor, window_events in windows.items():
            last_event = window_events[-1]
            window_start = window_events[0].timestamp
            window_end = last_event.timestamp + last_event.duration
            key = (
                window_start,
                window_end,
            )
            existing_descriptior = windows_with_interval.get(key, None)
            if existing_descriptior is not None:
                length_diff = len(descriptor.get_kv_pairs()) - len(existing_descriptior.get_kv_pairs())
                if length_diff > 0:
                    # Current descriptor is longer, replace.
                    windows_with_interval[key] = descriptor
                    metrics.incr(
                        "removed activity-by-strategy-es with same intervals but less keys",
                        (window_end - window_start).total_seconds(),
                    )
                elif length_diff < 0:
                    # Current descriptor is shorter, do nothing.
                    continue
                else:
                    LOG.warning(
                        "Wrong configuration for %s strategy: "
                        + "got the same interval/events grouped by 'same length' %s and %s",
                        self.current_strategy.name,
                        existing_descriptior,
                        descriptor,
                    )
            else:
                windows_with_interval[key] = descriptor
        # Check that duplicates were found. If yes, then rebuild `windows`.
        if len(windows_with_interval) < len(windows):
            tmp = {}
            for descriptor in windows_with_interval.values():
                tmp[descriptor] = windows[descriptor]
            windows = tmp
        return windows

    def _aggregate_events_with_few_sliding_windows(self) -> StrategyApplyResult:
        """Produces Activities covering all "same data" events."""
        windows: Dict[GroupingDescriptior, List[Event]] = self._separate_events_per_windows(events)
        # Make activities from the each window.
        activities = []
        for key, events in windows.items():
            self._make_activitybs_between_events(events, key, activities)
        return StrategyApplyResult(self.current_strategy, activities, self.current_metrics, self.current_id)

    def _aggregate_events_with_few_sliding_windows_by_density(self) -> StrategyApplyResult:
        """Produces overlapping Activities basing on theirs density on time scale."""
        windows: Dict[Tuple, List[Event]] = self._separate_events_per_windows(self.current_events)
        # Analyse gaps in each window.
        activities: List[ActivityByStrategy] = []
        for key, window_events in windows.items():
            # Measure all gaps.
            gaps: List[Tuple[int, float]] = []
            prev_event = None
            for i, event in enumerate(window_events):
                if prev_event:
                    gap = (event.timestamp - (prev_event.timestamp + prev_event.duration)).seconds
                    if gap > 0:
                        gaps.append((i, gap))
                prev_event = event
            if len(gaps) <= 0:
                # If there are no gaps then just create one activity-by-strategy from all events.
                self._make_activitybs_between_events(window_events, key, activities)
                continue
            # Otherwise iterate gaps to find out those which are bigger than minimal activity duration.
            # Make activities between them.
            last_activitybs_event_index = 0
            for gap in gaps:
                # If gap bigger than minimal activity duration then it is a separate activity-by-strategy.
                if gap[1] >= MIN_ACTIVITY_DURATION_SEC:
                    self._make_activitybs_between_events(
                        window_events[last_activitybs_event_index : gap[0]],
                        key,
                        activities,
                    )
                    last_activitybs_event_index = gap[0]
            # Here we have activity-by-strategy-es made up to the last big gap.
            # Make activity-by-strategy-es from the remained events.
            if last_activitybs_event_index < len(window_events) - 1:
                self._make_activitybs_between_events(
                    window_events[last_activitybs_event_index:-1],
                    key,
                    activities,
                )
        return StrategyApplyResult(self.current_strategy, activities, self.current_metrics, self.current_id)
