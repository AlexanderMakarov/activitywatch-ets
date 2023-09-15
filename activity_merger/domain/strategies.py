import dataclasses
import datetime
from typing import Dict, List, Set, Tuple

from ..config.config import MIN_DURATION_SEC
from ..helpers.helpers import from_start_to_end_to_str, seconds_to_timedelta
from .input_entities import ActivityBoundaries, Event, Strategy
from .metrics import Metrics


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


@dataclasses.dataclass
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
    duration: float
    """
    Total duration of the activity in seconds measured by events.
    Note that events may be placed with gaps between.
    """
    events: List[Event]
    """List of (dominant) events the activity consists of."""
    grouping_data: GroupingDescriptior
    """Object describing why enclosed events are aggregated into activity."""
    strategy: Strategy
    """ Strategy used to create this activity."""

    def __repr__(self) -> str:
        return (
            f"{seconds_to_timedelta(self.duration)},"
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
        activities = "\n    ".join(str(x) for x in self.activities)
        return (
            f"{self.strategy}\n  Metrics ({len(metrics_strings)}):\n  {metrics}"
            f"\n  ActivityByStrategy-es ({len(self.activities)}):\n    {activities}"
        )

    def __repr__(self) -> str:
        return self.to_string()


def handle_events(strategy: Strategy, events: List[Event], metrics: Metrics) -> ActivitiesByStrategy:
    """
    Handles events for the specified `Strategy` to provide `ActivitiesByStrategy` instance.
    """

    if strategy.in_each_event_is_activity:  # Watchdog, Outlook.
        return _convert_each_event_into_activity(strategy, events, metrics)
    if strategy.in_events_density_matters:
        if strategy.in_activities_may_overlap:  # Jira, Window, Web, IDEA, VSCode, Git.
            # TODO good to have events to group different "data" into one activity.
            return _aggregate_events_with_few_sliding_windows_by_density(strategy, events, metrics)
        else:  # No examples yet.
            # TODO support rules for this case. Now there are no such Strategies.
            return _convert_each_event_into_activity(strategy, events, metrics)
    else:
        if strategy.in_activities_may_overlap:  # No examples yet.
            return _aggregate_events_with_few_sliding_windows(strategy, events, metrics)
        else:  # AFK
            # TODO support rules for this case.
            return _aggregate_events_with_one_sliding_window(strategy, events, metrics)


def _check_skip_event(event: Event, strategy: Strategy, metrics: Metrics) -> bool:
    if strategy.in_skip_key_value:
        for k, v in strategy.in_skip_key_value.items():
            if event.data.get(k) == v:
                metrics.increment(f"events skipped because have {k}={v}")
                return True
    return False


def _convert_each_event_into_activity(
    strategy: Strategy, events: List[Event], metrics: Metrics
) -> ActivitiesByStrategy:
    # Produces Activity per each event.
    activities: List[ActivityByStrategy] = []
    for event in events:
        if _check_skip_event(event, strategy, metrics):
            continue
        end_time = event.timestamp + event.duration
        activity = ActivityByStrategy(
            suggested_start_time=event.timestamp,
            suggested_end_time=end_time,
            max_start_time=event.timestamp,
            min_end_time=end_time,
            duration=event.duration.seconds,
            events=[event],
            grouping_data=ListOfPairsDescriptor([(k, str(v)) for k, v in event.data.items()]),
            strategy=strategy,
        )
        metrics.incr("activities", event.duration.seconds)
        activities.append(activity)
    return ActivitiesByStrategy(strategy, activities, metrics)


def _make_activity_between_events(
    events: List[Event],
    grouping_data: GroupingDescriptior,
    out_activity_boundaries: ActivityBoundaries,
    activities: List[ActivityByStrategy],
    strategy: Strategy,
    metrics: Metrics,
):
    """
    Makes activity starting on start of first event and ending at the end of last event.
    Next adds it to provided list and updates `Metrics` with it.
    """
    duration = sum(x.duration.seconds for x in events)
    # Use start of first event and end of last event as "suggested" bounds of the activity.
    start_time = events[0].timestamp
    end_time = events[-1].timestamp + events[-1].duration
    # For max and min time check 'out_activity_boundaries' and put start/end time of first/last event respectively.
    # Note that for STRICT and DIM boundaries max and min time points aren't used.
    # Because any overlap is either removes activity or splits it.
    max_start_time = start_time
    min_end_time = end_time
    if out_activity_boundaries == ActivityBoundaries.END:
        max_start_time += events[0].duration
    elif out_activity_boundaries == ActivityBoundaries.START:
        min_end_time = events[-1].timestamp
    activity = ActivityByStrategy(
        suggested_start_time=start_time,
        suggested_end_time=end_time,
        max_start_time=max_start_time,
        min_end_time=min_end_time,
        duration=duration,
        events=events,
        grouping_data=grouping_data,
        strategy=strategy,
    )
    activities.append(activity)
    metrics.incr("activities", duration)


def _aggregate_events_with_one_sliding_window(
    strategy: Strategy, events: List[Event], metrics: Metrics
) -> ActivitiesByStrategy:
    # Produces Activity for all consequint events with the same data.
    activities: List[ActivityByStrategy] = []
    window: List[Event] = []
    # Prepare variable to store keys in event's 'data' by which search similar events.
    window_kv_pairs: Set[Tuple[str, str]] = set()
    # Start iterate over events with collecting windows.
    for event in events:
        if _check_skip_event(event, strategy, metrics):
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
            _make_activity_between_events(
                window,
                window_key,
                strategy.out_activity_boundaries,
                activities,
                strategy,
                metrics,
            )
            window = []
        # In any case prepare next window.
        window_kv_pairs = kv_pairs
        window.append(event)
    # Handle last window.
    if window:
        _make_activity_between_events(
            window,
            ListOfPairsDescriptor(list(window_kv_pairs)),
            strategy.out_activity_boundaries,
            activities,
            strategy,
            metrics,
        )
    return ActivitiesByStrategy(strategy, activities, metrics)


def _add_event_to_window(
    event: Event, key: GroupingDescriptior, windows: Dict[GroupingDescriptior, List[Event]], metrics: Metrics
):
    window = windows.setdefault(key, [])
    metrics.incr(f"events with data {key}", event.duration.seconds)
    window.append(event)


def _separate_events_per_windows(
    events: List[Event], strategy: Strategy, metrics: Metrics
) -> Dict[GroupingDescriptior, List[Event]]:
    """
    Separates list of events into windows basing on the data inside and 'group_by_keys'.
    :param events: List of events to separate.
    :param group_by_keys: Set of keys in event's "data" field to use for making windows.
    :param metrics: Metrics object to fill with actions inside.
    :return: Resulting windows with keys equal to event's "data" field values chosen as window identifiers
    and values as correspondings lists of event's.
    """
    # First collect all possible windows.
    windows: Dict[GroupingDescriptior, List[Event]] = {}
    if strategy.in_group_by_keys:
        for event in events:
            if _check_skip_event(event, strategy, metrics):
                continue
            # If way to make windows is specified they make window per each group of keys.
            is_added = False
            for key_tuple in strategy.in_group_by_keys:
                if all(key in event.data for key in key_tuple):
                    window_key = ListOfPairsDescriptor([(key, str(event.data[key])) for key in key_tuple])
                    _add_event_to_window(event, window_key, windows, metrics)
                    is_added = True
                    break
            # If wasn't added then it is either warning about misconfiguration or warning about bad event data.
            if not is_added:
                metrics.incr("events without data containing any in_group_by_keys", event.duration.total_seconds())
    else:
        # If way to make windows is not specified then build window key as tuple of data key-value pairs.
        for event in events:
            if _check_skip_event(event, strategy, metrics):
                continue
            window_key = ListOfPairsDescriptor([(k, str(v)) for k, v in event.data])
            _add_event_to_window(event, window_key, windows, metrics)
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


def _aggregate_events_with_few_sliding_windows(
    strategy: Strategy, events: List[Event], metrics: Metrics
) -> ActivitiesByStrategy:
    """Produces Activities covering all "same data" events."""
    windows: Dict[GroupingDescriptior, List[Event]] = _separate_events_per_windows(events, strategy, metrics)
    # Make activities from the each window.
    activities = []
    for key, events in windows.items():
        _make_activity_between_events(events, key, strategy.out_activity_boundaries, activities, strategy, metrics)
    return ActivitiesByStrategy(strategy, activities, metrics)


def _aggregate_events_with_few_sliding_windows_by_density(
    strategy: Strategy, events: List[Event], metrics: Metrics
) -> ActivitiesByStrategy:
    """Produces overlapping Activities basing on theirs density on time scale."""
    windows: Dict[Tuple, List[Event]] = _separate_events_per_windows(events, strategy, metrics)
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
            _make_activity_between_events(
                window_events, key, strategy.out_activity_boundaries, activities, strategy, metrics
            )
            continue
        # Otherwise iterate gaps to find out those which are bigger than minimal activity duration.
        # Make activities between them.
        last_activity_event_index = 0
        for gap in gaps:
            # If gap bigger than minimal activity duration then it is a separate activity.
            if gap[1] >= MIN_DURATION_SEC:
                _make_activity_between_events(
                    window_events[last_activity_event_index : gap[0]],
                    key,
                    strategy.out_activity_boundaries,
                    activities,
                    strategy,
                    metrics,
                )
                last_activity_event_index = gap[0]
        # Here we have activities made up to the last big gap. Make activity from the remained events.
        if last_activity_event_index < len(window_events) - 1:
            _make_activity_between_events(
                window_events[last_activity_event_index:-1],
                key,
                strategy.out_activity_boundaries,
                activities,
                strategy,
                metrics,
            )
    return ActivitiesByStrategy(strategy, activities, metrics)
