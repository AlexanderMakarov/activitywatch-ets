import dataclasses
import datetime
from typing import Dict, List, Set, Tuple

import intervaltree

from ..config.config import MIN_DURATION_SEC
from ..helpers.helpers import event_to_str, from_start_to_end_to_str, seconds_to_timedelta
from .input_entities import IntervalBoundaries, Event, Strategy
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

    suggested_start_time: datetime.datetime
    """Suggested start time of the activity."""
    suggested_end_time: datetime.datetime
    """Suggested end time of the activity."""
    max_start_time: datetime.datetime
    """Maximum start time of the activity possible. Equal or later then 'start time'."""
    min_end_time: datetime.datetime
    """Minimum end time of the activity possible. Equal or earlier then 'end time'."""
    events: List[Event]
    """List of (dominant) events the activity consists of."""
    grouping_data: GroupingDescriptior
    """Object describing why enclosed events are aggregated into activity."""
    strategy: Strategy
    """ Strategy used to create this activity."""

    def duration(self) -> float:
        """Returns duration from suggest start to suggest end in seconds."""
        return (self.suggested_end_time - self.suggested_start_time).total_seconds()

    def __repr__(self) -> str:
        return (
            f"{seconds_to_timedelta(self.duration())},"
            f" {from_start_to_end_to_str(self.suggested_start_time, self.suggested_end_time)}"
            f" (min {from_start_to_end_to_str(self.max_start_time, self.min_end_time)}),"
            f" {len(self.events):>3} {self.strategy.name} events grouped by {self.grouping_data}."
        )


@dataclasses.dataclass
class ActivitiesByStrategy:
    """List of activities aggregated by a strategy."""

    strategy: Strategy
    activities: List[ActivityByStrategy]
    metrics: Metrics

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


# def cut_event(event: Event, interval: intervaltree.Interval) -> Event:
#     """
#     Makes new event with part which is overlapped by the given interval.
#     """
#     return Event(
#         event.bucket_id,
#         max(event.timestamp, interval.begin),
#         min(event.timestamp + event.duration, interval.end),
#         event.data
#     )


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
        event_end = last_overlap.end
    elif boundaries == IntervalBoundaries.END:
        if last_overlap.end < event_end:
            if is_fail_incompatible:
                raise ValueError(
                    f"Strategy '{strategy_name}' boundaries '{boundaries}' are incompatible with AFK events for "
                    + event_to_str(event)
                )
            else:
                return None
        event_start = first_overlap.begin
    elif boundaries == IntervalBoundaries.DIM:
        # Just use the first overlap to cut event by.
        event_start = max(first_overlap.begin, event_start)
        event_end = min(first_overlap.end, event_end)

    return Event(bucket_id=event.bucket_id, timestamp=event_start, duration=event_end - event_start, data=event.data)


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
            metrics.incr("events chopped by 'relevant app active'", (initial_duration - event.duration).total_seconds())
        return event

    def handle_events(self, strategy: Strategy, events: List[Event], metrics: Metrics) -> ActivitiesByStrategy:
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

        # TODO: refactor to pluggable handlers.
        # Hanldle events depending on strategy parameters.
        if strategy.in_each_event_is_activity:  # Watchdog, Outlook.
            return self._convert_each_event_into_activity()
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

    def _convert_each_event_into_activity(self) -> ActivitiesByStrategy:
        # Produces Activity per each event.
        activities: List[ActivityByStrategy] = []
        for event in self.current_events:
            event = self._transform_event(event)
            if event is None:
                continue
            end_time = event.timestamp + event.duration
            activity = ActivityByStrategy(
                suggested_start_time=event.timestamp,
                suggested_end_time=end_time,
                max_start_time=event.timestamp,
                min_end_time=end_time,
                events=[event],
                grouping_data=ListOfPairsDescriptor([(k, str(v)) for k, v in event.data.items()]),
                strategy=self.current_strategy,
            )
            self.current_metrics.incr("activities", event.duration.seconds)
            activities.append(activity)
        return ActivitiesByStrategy(self.current_strategy, activities, self.current_metrics)

    def _make_activity_between_events(
        self,
        events: List[Event],
        grouping_data: GroupingDescriptior,
        activities: List[ActivityByStrategy],
    ):
        """
        Makes activity starting on start of first event and ending at the end of last event.
        Next adds it to provided list and updates `Metrics` with it.
        """
        # Use start of first event and end of last event as "suggested" bounds of the activity.
        start_time = events[0].timestamp
        end_time = events[-1].timestamp + events[-1].duration
        # For max and min time check 'in_trustable_boundaries' and put start/end time of first/last event respectively.
        # Note that for STRICT and DIM boundaries max and min time points aren't used.
        # Because any overlap is either removes activity or splits it.
        max_start_time = start_time
        min_end_time = end_time
        in_trustable_boundaries = self.current_strategy.in_trustable_boundaries
        if in_trustable_boundaries is IntervalBoundaries.END:
            max_start_time += events[0].duration
        elif in_trustable_boundaries is IntervalBoundaries.START:
            min_end_time = events[-1].timestamp
        activity = ActivityByStrategy(
            suggested_start_time=start_time,
            suggested_end_time=end_time,
            max_start_time=max_start_time,
            min_end_time=min_end_time,
            events=events,
            grouping_data=grouping_data,
            strategy=self.current_strategy,
        )
        activities.append(activity)
        self.current_metrics.incr("activities", activity.duration())

    def _aggregate_events_with_one_sliding_window(self) -> ActivitiesByStrategy:
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
                self._make_activity_between_events(
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
            self._make_activity_between_events(
                window,
                ListOfPairsDescriptor(list(window_kv_pairs)),
                activities,
            )
        return ActivitiesByStrategy(self.current_strategy, activities, self.current_metrics)

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
        # First collect all possible windows.
        windows: Dict[GroupingDescriptior, List[Event]] = {}
        strategy = self.current_strategy
        metrics = self.current_metrics
        if strategy.in_group_by_keys:
            for event in events:
                event = self._transform_event(event)
                if event is None:
                    continue
                # If way to make windows is specified they make window per each group of keys.
                is_added = False
                for key_tuple in strategy.in_group_by_keys:
                    if all(key in event.data for key in key_tuple):
                        window_key = ListOfPairsDescriptor([(key, str(event.data[key])) for key in key_tuple])
                        self._add_event_to_window(event, window_key, windows)
                        is_added = True
                        break
                # If wasn't added then it is either warning about misconfiguration or warning about bad event data.
                if not is_added:
                    metrics.incr("events without data containing any in_group_by_keys", event.duration.total_seconds())
        else:
            # If way to make windows is not specified then build window key as tuple of data key-value pairs.
            for event in events:
                event = self._transform_event(event)
                if event is None:
                    continue
                window_key = ListOfPairsDescriptor([(k, str(v)) for k, v in event.data])
                self._add_event_to_window(event, window_key, windows)
        # Next check windows for the "same events" entries which may appear if group_by_keys contains few entries
        # and some set of events have the same value for both keys.
        # Step 1: build "inverted windows" dict with all "same events" windows keys grouped.
        inverted_windows: Dict[int, List[Tuple]] = {}
        keys_to_remove = set()
        for key, window_events in windows.items():
            events_hash = hash(tuple(str(x) for x in window_events))
            same_events_window_keys = inverted_windows.setdefault(events_hash, [])
            if same_events_window_keys:
                # If the window with the same events exists then add to keys_to_remove all these keys.
                if len(same_events_window_keys) < 2:
                    keys_to_remove.add(same_events_window_keys[0])
                    metrics.incr("windows with similar events")
                keys_to_remove.add(key)
                metrics.incr("windows with similar events")
            same_events_window_keys.append(key)
        # Step 2: iterate over inverted_windows and create new windows with "merged" keys for duplicates.
        for same_events_window_keys in inverted_windows.values():
            if len(same_events_window_keys) > 1:
                new_window_key = tuple(x for key in same_events_window_keys for x in key)
                window_events = windows[same_events_window_keys[0]]
                windows[new_window_key] = window_events
                metrics.incr("windows with combined keys due to similar events in different groups")
        # Step 3: remove duplicated windows with old keys.
        for key in keys_to_remove:
            del windows[key]
        return windows

    def _aggregate_events_with_few_sliding_windows(self) -> ActivitiesByStrategy:
        """Produces Activities covering all "same data" events."""
        windows: Dict[GroupingDescriptior, List[Event]] = self._separate_events_per_windows(events)
        # Make activities from the each window.
        activities = []
        for key, events in windows.items():
            self._make_activity_between_events(events, key, activities)
        return ActivitiesByStrategy(self.current_strategy, activities, self.current_metrics)

    def _aggregate_events_with_few_sliding_windows_by_density(self) -> ActivitiesByStrategy:
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
                # If there are no gaps then just create one activity from all events.
                self._make_activity_between_events(window_events, key, activities)
                continue
            # Otherwise iterate gaps to find out those which are bigger than minimal activity duration.
            # Make activities between them.
            last_activity_event_index = 0
            for gap in gaps:
                # If gap bigger than minimal activity duration then it is a separate activity.
                if gap[1] >= MIN_DURATION_SEC:
                    self._make_activity_between_events(
                        window_events[last_activity_event_index : gap[0]],
                        key,
                        activities,
                    )
                    last_activity_event_index = gap[0]
            # Here we have activities made up to the last big gap. Make activity from the remained events.
            if last_activity_event_index < len(window_events) - 1:
                self._make_activity_between_events(
                    window_events[last_activity_event_index:-1],
                    key,
                    activities,
                )
        return ActivitiesByStrategy(self.current_strategy, activities, self.current_metrics)
