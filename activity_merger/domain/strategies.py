import dataclasses
import datetime
import re
from typing import Dict, List, Optional, Set, Tuple

import intervaltree
import numpy as np
from sklearn.cluster import DBSCAN

from activity_merger.helpers.event_helpers import activity_by_strategy_to_str

from ..config.config import (
    EVENTS_COMPARE_TOLERANCE_TIMEDELTA,
    LOG,
    MIN_ACTIVITY_DURATION_SEC,
    MIN_DENSITY_FOR_SPARCE_INTERVALS,
    TOO_SMALL_INTERVAL_SEC,
)
from ..helpers.helpers import datetime_to_time_str
from .input_entities import (
    ActivityByStrategy,
    DictGroupingDescriptor,
    Event,
    GroupingDescriptior,
    IntervalBoundaries,
    Strategy,
)
from .metrics import Metrics

BUCKET_AFK_PREFIX = "aw-watcher-afk"
BUCKET_STOPWATCH_PREFIX = "aw-stopwatch"
BUCKET_WINDOW_PREFIX = "aw-watcher-window"


@dataclasses.dataclass
class StrategyApplyResult:
    """Result of applying one strategy on list of ActivityWatch events."""

    strategy: Strategy
    activities: List[ActivityByStrategy]
    metrics: Metrics
    last_id: int

    def to_string(self, ignore_metrics_by_substrings: Optional[List[str]] = None) -> str:
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
        activities = "\n    ".join(activity_by_strategy_to_str(x) for x in sorted(self.activities, key=lambda x: x.suggested_start_time))
        return (
            f"{self.strategy}\n  Metrics ({len(metrics_strings)}):\n  {metrics}"
            f"\n  ActivityByStrategy-es ({len(self.activities)}):\n    {activities}"
        )

    def __repr__(self) -> str:
        return self.to_string()


def cut_by_interval_tree(
    event_start: datetime.datetime,
    event_end: datetime.datetime,
    tree: intervaltree.IntervalTree,
    boundaries: IntervalBoundaries,
    tree_name: str,
    metrics: Metrics,
    strategy_name: str = None,
    is_fail_incompatible: bool = False,
) -> Tuple[bool, Optional[Tuple[datetime.datetime, datetime.datetime]]]:
    """
    Calculates event boundaries to be on the intervals in the given tree taking into account boundaries.
    If boundaries and way of overlapping with tree are incompatible then either returns `None` or raises `ValueError`.
    List of incompatible cases:
    - "strict" boundaries and there is no overlaps in the tree at least for one piece of event.
    - "start" boundaries and there is no overlapping with event start.
    - "end" boundaries and there is no overlapping with event end.
    Intervals in tree are expected to be not overlapping.
    May return 3 types of result:
    1. Event interval is OK as is - in this case first item in result is False.
    2. Event should disappear completely - in this case second item in result is None.
    3. Need to cut event to [start, end] - in this case first item is True and second item is new begin and end.
    """
    initial_duration = event_end - event_start
    is_changed = False

    # Get overlapping intervals
    overlaps: List[intervaltree.Interval] = sorted(tree[event_start:event_end])
    if not overlaps:
        return (False, None)
    metrics.incr(f"events overlapped by {len(overlaps)} {tree_name} events", initial_duration.total_seconds())

    first_overlap = overlaps[0]
    last_overlap = overlaps[-1]

    # Handle according to boundaries
    if boundaries == IntervalBoundaries.STRICT:
        # Check that no one from tree's intervals ends inside event.
        event_interval = intervaltree.Interval(event_start, event_end)
        for overlap in overlaps:
            if not overlap.contains_interval(event_interval):
                if is_fail_incompatible:
                    raise ValueError(
                        f"Strategy '{strategy_name}' boundaries '{boundaries}' are incompatible with {tree_name}"
                        f" intervals for event in [{event_start}-{event_end}]"
                    )
                else:
                    return (False, None)
    elif boundaries == IntervalBoundaries.START:
        # Check that first overlapping interval doesn't overlap event start.
        if first_overlap.begin > event_start:
            if is_fail_incompatible:
                raise ValueError(
                    f"Strategy '{strategy_name}' boundaries '{boundaries}' are incompatible with {tree_name}"
                    f" intervals for event in [{event_start}-{event_end}]"
                )
            else:
                return (False, None)
        # If there is a gap in "tree overlapping" then need to cut event to the shortest interval from the start.
        if first_overlap.end < event_end:
            is_changed = True
            event_end = first_overlap.end
    elif boundaries == IntervalBoundaries.END:
        # Check that last overlapping interval doesn't overlap event end.
        if last_overlap.end < event_end:
            if is_fail_incompatible:
                raise ValueError(
                    f"Strategy '{strategy_name}' boundaries '{boundaries}' are incompatible with {tree_name}"
                    f" intervals for event in [{event_start}-{event_end}]"
                )
            else:
                return (False, None)
        # If there is a gap in "tree overlapping" then need to cut event to the shortest interval from the end.
        if last_overlap.begin > event_start:
            is_changed = True
            event_start = last_overlap.begin
    elif boundaries == IntervalBoundaries.DIM:
        # In a case of DIM boundaries we may get into situation when need to split event on few.
        # But few events from one is a thing which need to avoid because it doesn't make sense - event begin or end
        # or both should be supported by some interval in tree (assuming that strategy config is right).
        # But due to author of strategy chose DIM they are not sure in exact boundaries.
        # It happens when:
        # 1. Events are generated without real support. For example Outlook events may end up earlier or later.
        #    In this case need cut event to shortest overlapping interval from the start - extending event
        #    duration is job of more high level logic.
        # 2. Events are sometimes generated wronlgy. For example IDEA generates events on app start or
        #    "wake up from sleep" motivators. Or "today's first" GIT/Jira events start from the start of the day.
        #    But due to these activities may happen over night then it may accidentally intersect with real activity.
        #    Anywat we need to cut by shortest overlapping interval from the end.
        # What to choose? The best idea is to find shortest overlaps for start and end then choose longest from them.

        first_interval_start = event_start if first_overlap.begin <= event_start else None
        first_interval_end = min(first_overlap.end, event_end)
        last_interval_start = max(last_overlap.begin, event_start) if last_overlap.end >= event_end else None
        last_interval_end = event_end
        # Compare overlapped intervals. Check by *_start variables!
        if first_interval_start is None and last_interval_start is None:
            # Skip event entirely - it is overlapped only somewhere in the middle.
            metrics.incr(
                f"DIM events skipped because edges aren't overlapped by {tree_name} events",
                initial_duration.total_seconds(),
            )
            return (False, None)
        elif first_interval_start and last_interval_start:
            # Choose longest overlapped interval.
            if (first_interval_end - first_interval_start) >= (last_interval_end - last_interval_start):
                metrics.incr(
                    f"DIM events overlapped by {tree_name} events better at start",
                    (first_interval_end - first_interval_start).total_seconds(),
                )
                event_start = first_interval_start
                event_end = first_interval_end
            else:
                metrics.incr(
                    f"DIM events overlapped by {tree_name} events better at end",
                    (last_interval_end - last_interval_start).total_seconds(),
                )
                event_start = last_interval_start
                event_end = last_interval_end
        elif first_interval_start:
            metrics.incr(
                f"DIM events overlapped by {tree_name} events only at start",
                (first_interval_end - first_interval_start).total_seconds(),
            )
            event_start = first_interval_start
            event_end = first_interval_end
        else:
            metrics.incr(
                f"DIM events overlapped by {tree_name} events better at end",
                (last_interval_end - last_interval_start).total_seconds(),  # type: ignore
            )
            event_start = last_interval_start  # type: ignore
            event_end = last_interval_end

    # Check that the resulting duration is not greater than initial as simple sanity check.
    if event_end - event_start > initial_duration:
        raise AssertionError(
            f"Issue with calculating points to chop by '{tree_name}' tree: initial duration {initial_duration}"
            f" is less than resulting [{datetime_to_time_str(event_start)}..{datetime_to_time_str(event_end)}]"
        )
    return (is_changed, (event_start, event_end))


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


class EventChangeHandler:
    """
    Manages single event tranformation persistence and decoding.
    """

    __slots__ = ["changes"]

    def __init__(self):
        self.changes = []

    def add_change(self, desc, is_start_cut=False, is_end_cut=False, is_data_changed=False):
        """
        Adds description of a change.
        :param desc: The change description.
        :param is_start_cut: Whether event interval start was cut.
        :param is_end_cut: Whether event interval end was cut.
        :param is_data_changed: Whether event data was changed.
        """
        fields = ""
        if is_start_cut:
            fields += "-start"
        if is_end_cut:
            fields = "-end"
        if is_data_changed:
            fields += "*data"
        self.changes.append(f"{desc}:{fields}")

    def is_changed(self) -> bool:
        """
        Checks if there were any changes.
        """
        return bool(self.changes)

    def __repr__(self) -> str:
        return ", ".join(self.changes)

    def update_event_data(self, event_data: dict):
        """
        Updates provided event data dictionary with accumulated changes description.
        """
        if self.changes:
            event_data["changes"] = ", ".join(self.changes)

    @staticmethod
    def get_event_boundaries_changes(event_data: dict) -> Tuple[bool, bool]:
        """
        Returns tuple of 2 booleans indicating whether event interval was cut on start or/and end correspondingly.
        """
        changes = event_data.get("changes")
        if not changes:
            return False, False
        return "-start" in changes, "-end" in changes


class InStrategyPropertiesHandler:
    """
    Statefull object to convert list of events by building `ActivitiesByStrategy` from them
    using "in" properties of relevant `Strategy`.
    """

    def __init__(self):
        self.current_strategy: Strategy = None
        self.current_strategy_skip_key_pattern_list = []
        self.current_strategy_only_key_pattern_list = []
        self.current_metrics: Metrics = None
        self.current_events: List[Event] = []
        self.any_afk_tree: intervaltree.IntervalTree = intervaltree.IntervalTree()
        self.not_afk_tree: intervaltree.IntervalTree = intervaltree.IntervalTree()
        self.window_events: List[Event] = []
        self.current_window_tree: intervaltree.IntervalTree = None
        self.current_id: int = 0

    @staticmethod
    def _build_tree(events: List[Event]) -> intervaltree.IntervalTree:
        tree = intervaltree.IntervalTree()
        for event in events:
            tree.addi(event.timestamp, event.timestamp + event.duration, event)
        return tree

    def _transform_event(self, event: Event) -> Optional[Event]:
        """
        Both checks that event may be used for building `ActivityByStrategy` and cuts it accordingly to
        "in_only_not_afk" and "in_only_if_window_app" parameters of the strategy.
        Adds information where and how event was transformed if it was.
        """
        # Skip event entirely by "black list" or "white list".
        metrics = self.current_metrics
        event_start = event.timestamp
        event_duration = event.duration
        event_end = event_start + event_duration
        event_data = event.data
        change_handler = EventChangeHandler()
        for key, pattern in self.current_strategy_skip_key_pattern_list:
            key_value = event_data.get(key, None)
            if pattern.match(key_value):
                metrics.incr(
                    f"events skipped because have '{key}' value matches '{pattern}'",
                    event_duration.total_seconds(),
                )
                return None
        for key, pattern in self.current_strategy_only_key_pattern_list:
            value = event_data.get(key, None)
            match = pattern.match(value)
            if not match:
                metrics.incr(
                    f"events skipped because have '{key}' value doesn't match '{pattern}'",
                    event_duration.total_seconds(),
                )
                return None
            else:
                change_handler.add_change("in_only_key_regexp", is_data_changed=True)
                event_data[key] = match.group(1) if match.groups() else match.group(0)

        strategy = self.current_strategy
        boundaries = strategy.in_trustable_boundaries

        # Cut event by windows watcher "app=" events if need.
        # Should be before AFK chopping because we don't want to get random piece of event.
        if strategy.in_only_if_window_app:
            # Cut event by to be only in "relevant app active" intervals.
            is_changed, new_interval = cut_by_interval_tree(
                event_start,
                event_end,
                self.current_window_tree,
                boundaries,
                "relevant app active",
                metrics,
                strategy.name,
                True,
            )
            if new_interval is None:
                metrics.incr("events cut out by 'relevant app active'", event_duration.total_seconds())
                return None
            elif is_changed:
                change_handler.add_change(
                    "in_only_if_window_app",
                    is_start_cut=(event_start != new_interval[0]),
                    is_end_cut=(event_end != new_interval[1]),
                )
                event_start = new_interval[0]
                event_end = new_interval[1]
                metrics.incr(
                    "events chopped by 'relevant app active'",
                    (event_duration - (event_end - event_start)).total_seconds(),
                )

        # Cut event by AFK if need.
        # Note that if we need "only within not-AFK" then "only within any AFK" is applied automatically.
        if strategy.in_only_not_afk:
            # Cut event by to be only in "not-AFK" intervals.
            is_changed, new_interval = cut_by_interval_tree(
                event_start, event_end, self.not_afk_tree, boundaries, "not-AFK", metrics, strategy.name, False
            )
            if new_interval is None:
                metrics.incr("events cut out by 'not-AFK'", event_duration.total_seconds())
                return None
            elif is_changed:
                change_handler.add_change(
                    "in_only_not_afk",
                    is_start_cut=(event_start != new_interval[0]),
                    is_end_cut=(event_end != new_interval[1]),
                )
                event_start = new_interval[0]
                event_end = new_interval[1]
                metrics.incr(
                    "events chopped by 'not-AFK'", (event_duration - (event_end - event_start)).total_seconds()
                )
        elif not strategy.in_may_be_offline:
            is_changed, new_interval = cut_by_interval_tree(
                event_start, event_end, self.any_afk_tree, boundaries, "any-AFK", metrics, strategy.name, False
            )
            if new_interval is None:
                metrics.incr("events cut out by 'any-AFK'", event_duration.total_seconds())
                return None
            elif is_changed:
                change_handler.add_change(
                    "in_may_be_offline",
                    is_start_cut=(event_start != new_interval[0]),
                    is_end_cut=(event_end != new_interval[1]),
                )
                event_start = new_interval[0]
                event_end = new_interval[1]
                metrics.incr(
                    "events chopped by 'any-AFK'", (event_duration - (event_end - event_start)).total_seconds()
                )

        # Transform event if need.
        if change_handler.is_changed():
            change_handler.update_event_data(event_data)
            event = Event(
                bucket_id=event.bucket_id, timestamp=event_start, duration=event_end - event_start, data=event.data
            )
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

        # Reset values from the previous strategy.
        self.current_strategy_skip_key_pattern_list = []
        self.current_strategy_only_key_pattern_list = []
        # Fill up "current" properties (sometimes conditionally).
        self.current_strategy = strategy
        self.current_metrics = metrics
        self.current_events = events
        if strategy.in_skip_key_regexp:
            self.current_strategy_skip_key_pattern_list = list(
                (k, re.compile(regexp)) for k, regexp in strategy.in_skip_key_regexp.items()
            )
        if strategy.in_only_key_regexp:
            self.current_strategy_only_key_pattern_list = list(
                (k, re.compile(regexp)) for k, regexp in strategy.in_only_key_regexp.items()
            )

        # Copy "standard watchers" events.
        if strategy.bucket_prefix.startswith(BUCKET_AFK_PREFIX):
            # Build 2 AFK trees - to filter by "any_afk" and to sort by "not-afk".
            for event in events:
                event_begin = event.timestamp
                event_duration = event.duration
                status = event.data["status"]
                self.any_afk_tree.addi(event_begin, event_begin + event_duration, event)
                if status == "not-afk":
                    self.not_afk_tree.addi(event_begin, event_begin + event_duration, event)
                    metrics.incr("not-afk events", event_duration.total_seconds())
                else:
                    metrics.incr("afk events", event_duration.total_seconds())
        if strategy.bucket_prefix.startswith(BUCKET_WINDOW_PREFIX):
            assert self.current_window_tree is None, (
                "Wrong order of buckets handling, 'current_window_tree' already exists"
                + f" while strategy with '{BUCKET_WINDOW_PREFIX}' bucket prefix is received."
                + f" Keep all strategies handling '{BUCKET_WINDOW_PREFIX}' bucket just after '{BUCKET_AFK_PREFIX}'."
            )
            # Just save events because for each strategy will be specified different events.
            self.window_events.extend(events)

        # Build "window_tree" if need for the strategy and wasn't build before.
        if strategy.in_only_if_window_app and self.current_window_tree is None:
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
                return self._aggregate_events_with_few_sliding_windows_by_density_threshold()
                # A-b-s-es span throught AFK events which leads to low density.
                # return self._aggregate_events_with_few_sliding_windows_by_recursive_gaps_comparison()
                # Inconsistent between small and large windows.
                # return self._aggregate_events_with_few_sliding_windows_by_optics()
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
                grouping_data=DictGroupingDescriptor(event.data.copy()),
                strategy=self.current_strategy,
            )
            self.current_metrics.incr("activities", event.duration.seconds)
            activities.append(activitybs)
        return StrategyApplyResult(self.current_strategy, activities, self.current_metrics, self.current_id)

    def _make_activitybs_between_events(
        self,
        events: List[Event],
        grouping_data: GroupingDescriptior,
    ) -> Optional[ActivityByStrategy]:
        """
        Makes activity-by-strategy starting on start of first event and ending at the end of last event.
        Updates `Metrics` with it.
        """
        # Use start of first event and end of last event as "suggested" bounds of the activity-by-strategy.
        start_time = events[0].timestamp
        end_time = events[-1].timestamp + events[-1].duration
        # Skip too small intervals. Don't do it on "suggested" values because they may provide duration=0.
        if self._check_wrong_duration(end_time - start_time):
            return None
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
        self.current_metrics.incr("activity-by-strategy-es", activitybs.duration())
        return activitybs

    def _aggregate_events_with_one_sliding_window(self) -> StrategyApplyResult:
        """
        Runs one sliding window on events list and creates new activity-by-strategy each time as new set of
        key-value pairs is found.
        In result produces activity-by-strategy for all consequent events with the same data.
        """
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
            event_data = event.data
            # Collect key-value pairs from event data.
            kv_pairs: Set[Tuple[str, str]] = set()
            if strategy.in_group_by_keys:
                # For each tuple of keys in the list check if all are placed in event's data and
                # if so then add key-value pairs to `kv_pairs` and stop. We need only 1 window.
                for key_tuple in strategy.in_group_by_keys:
                    if all(key in event_data for key in key_tuple):
                        kv_pairs.update(
                            (
                                k,
                                event_data[k],
                            )
                            for k in event_data
                            if k in key_tuple
                        )
                        break
            else:
                kv_pairs.update(
                    (
                        k,
                        event_data[k],
                    )
                    for k in event_data
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
                window_key = DictGroupingDescriptor(dict(window_kv_pairs))
                activitybs = self._make_activitybs_between_events(window, window_key)
                if activitybs is not None:
                    activities.append(activitybs)
                window = []
            # Prepare next window.
            window_kv_pairs = kv_pairs
            window.append(event)
        # Handle last window.
        if window:
            activitybs = self._make_activitybs_between_events(window, DictGroupingDescriptor(dict(window_kv_pairs)))
            if activitybs is not None:
                activities.append(activitybs)
        return StrategyApplyResult(self.current_strategy, activities, self.current_metrics, self.current_id)

    def _add_event_to_window(
        self, event: Event, key: GroupingDescriptior, windows: Dict[GroupingDescriptior, List[Event]]
    ):
        window = windows.setdefault(key, [])
        # TODO revert self.current_metrics.incr(f"events with data {key}", event.duration.seconds)
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
                # For each tuple of keys in the list try to build GroupingDescriptor, skip and continue if
                # exception was raised. We need in as many windows as possible.
                for key_tuple in strategy.in_group_by_keys:
                    # Check if all keys in key_tuple are in event_data
                    if not all(key in event_data for key in key_tuple):
                        continue
                    # Create a dictionary for DictGroupingDescriptor using the keys from key_tuple
                    window_key_dict = {key: event_data[key] for key in key_tuple}
                    window_key = DictGroupingDescriptor(window_key_dict)
                    self._add_event_to_window(event, window_key, windows)
                    added_times += 1
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
                window_key = DictGroupingDescriptor([(k, str(v)) for k, v in event_data])
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
                length_diff = len(descriptor.get_data()) - len(existing_descriptior.get_data())
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
            activitybs = self._make_activitybs_between_events(events, key)
            if activitybs is not None:
                activities.append(activitybs)
        return StrategyApplyResult(self.current_strategy, activities, self.current_metrics, self.current_id)

    def _convert_events_with_gaps_to_activitiesbs(
        self,
        key,
        events: List[Event],
        gaps: List[Tuple[int, float]],
        max_gap_sec: float,
        recursion_max_gap_divider: float,
        events_indexes: Tuple[int, int],
        gaps_indexes: Tuple[int, int],
    ) -> List[ActivityByStrategy]:
        """
        Converts list of events and relevant gaps between them (list with length less at least on one item) into
        activities-by-strategy recursiverly (if need). Returns the number of created activity-by-strategy-es.
        Idea is to choose some gap which is too big and iterate over all gaps to clusterise events with it.
        Next in the each cluster choose gap as the fraction (half) of the lesser value from:
        - initial gap,
        - resulting covered by events interval,
        and separate by cluster until left no more big enough gaps to separate events into clusters.
        Remained clusters would be dense enough to make activity-by-strategy-es from them.
        """
        # If there are no gaps then just create single activity-by-strategy.
        activitiesbs = []
        if len(gaps) < 1 or gaps_indexes[1] - gaps_indexes[0] < 1:
            activitybs = self._make_activitybs_between_events(events[events_indexes[0] : events_indexes[1]], key)
            return [activitybs] if activitybs is not None else []
        # Otherwise iterate over all gaps to find gaps bigger than max_gap_sec recursively.
        finish_gaps_index = gaps_indexes[1]
        start_index = events_indexes[0]
        start_gaps_index = gaps_indexes[0]
        # Iterate 1 extra time to write recursion once (check finish by `end_gaps_index == finish_gap_index`).
        for end_gaps_index in range(gaps_indexes[0], finish_gaps_index + 1):
            gap = gaps[end_gaps_index] if end_gaps_index < finish_gaps_index else None
            # Check that gaps are over or current gap is big enough to separate cluster.
            if end_gaps_index >= finish_gaps_index or gap[1] >= max_gap_sec:
                # Note that 'end_index' is exclusive.
                end_index = gap[0] if end_gaps_index < finish_gaps_index else events_indexes[1]
                # Check that need to search clusters recursively.
                if recursion_max_gap_divider > 0 and (start_gaps_index, end_gaps_index) != gaps_indexes:
                    last_event = events[end_index] if end_gaps_index < finish_gaps_index else events[-1]
                    under_events_interval_duration_sec = (
                        last_event.timestamp + last_event.duration - events[start_index].timestamp
                    ).total_seconds()
                    # Run recursion with slice of events and relevant gaps.
                    local_activitiesbs = self._convert_events_with_gaps_to_activitiesbs(
                        key,
                        events,
                        gaps,
                        min(max_gap_sec, under_events_interval_duration_sec) / recursion_max_gap_divider,
                        recursion_max_gap_divider,
                        (start_index, end_index),
                        (start_gaps_index, end_gaps_index),  # Note that gaps list may be much less than events list.
                    )
                    activitiesbs += local_activitiesbs
                else:
                    # If don't need seach clusters recursively then just convert cluster to activity-by-strategy.
                    activitybs = self._make_activitybs_between_events(events[start_index:end_index], key)
                    if activitybs is not None:
                        activitiesbs.append(activitybs)
                start_index = end_index
                start_gaps_index = end_gaps_index + 1  # Need to shift gaps index because we are skipping it.
        return activitiesbs

    def _aggregate_events_with_few_sliding_windows_by_recursive_gaps_comparison(self) -> StrategyApplyResult:
        """
        Produces overlapping activity-by-strategy-es basing on a gaps between events.
        In details it first separates events with "same data" into windows (as big as needed).
        Next in each window iterates events to collect list of gaps between events.
        If there are no gaps then collects acitivity-by-strategy.
        Chooses some gap which is too big and iterate over all gaps to clusterise events with it.
        Next in the each cluster choose gap as the fraction (half) of the lesser value from:
        - initial gap,
        - resulting covered by events interval,
        and separate by cluster until left no more big enough gaps to separate events into clusters.
        """
        metrics = self.current_metrics
        windows: Dict[Tuple, List[Event]] = self._separate_events_per_windows(self.current_events)
        # Analyse gaps in each window.
        activitiesbs: List[ActivityByStrategy] = []
        for key, window_events in windows.items():
            # Measure all gaps in seconds.
            # Because 1 second for human is pretty small interval and may be treated as an continous activity.
            gaps: List[Tuple[int, float]] = []
            prev_event = None
            for i, event in enumerate(window_events):
                if prev_event:
                    gap_seconds = (event.timestamp - (prev_event.timestamp + prev_event.duration)).seconds
                    # Add only big enough gaps - for performance.
                    if gap_seconds > EVENTS_COMPARE_TOLERANCE_TIMEDELTA.seconds:
                        gaps.append((i, gap_seconds))
                prev_event = event
            # If there are no gaps then just create one activity-by-strategy from all events.
            if len(gaps) <= 0:
                activitybs = self._make_activitybs_between_events(window_events, key)
                if activitybs is not None:
                    activitiesbs.append(activitybs)
                metrics.incr("1 separated by density activity-by-strategy-es with same data")
                continue
            # Convert events into activities-by-strategy-es relative to trustable boundaries:
            # - START, END without recursive clusters search - just by MIN_ACTIVITY_DURATION_SEC gap.
            # - remained boundaries with recursive clusters search using divider = 2.
            recursion_max_gap_divider = (
                2
                if self.current_strategy.in_trustable_boundaries
                not in [IntervalBoundaries.START, IntervalBoundaries.END]
                else 0
            )
            # PROBLEM: A-b-s spans throught AFK events and with low density while should be reduced.
            activitiesbs += self._convert_events_with_gaps_to_activitiesbs(
                key=key,
                events=window_events,
                gaps=gaps,
                max_gap_sec=MIN_ACTIVITY_DURATION_SEC,
                recursion_max_gap_divider=recursion_max_gap_divider,
                events_indexes=(0, len(window_events)),
                gaps_indexes=(0, len(gaps)),
            )
            metrics.incr(f"{len(activitiesbs)} separated by density activity-by-strategy-es with same data")
        return StrategyApplyResult(self.current_strategy, activitiesbs, self.current_metrics, self.current_id)

    def _aggregate_events_with_few_sliding_windows_by_density_threshold(self) -> StrategyApplyResult:
        """
        Produces overlapping activity-by-strategy-es basing on theirs density on time scale.
        In details it first separates events with "same data" into windows (as big as needed).
        Next in each window iterates events with calculating density of the resulting events cluster with inlcuding
        the next event. If gap between events is bigger than 'MIN_ACTIVITY_DURATION_SEC' or resulting density is less
        than 'MIN_DENSITY_FOR_SPARCE_INTERVALS' then cuts new activity-by-strategy.
        """
        metrics = self.current_metrics
        windows: Dict[Tuple, List[Event]] = self._separate_events_per_windows(self.current_events)
        activitiesbs: List[ActivityByStrategy] = []
        for key, window_events in windows.items():
            # If it is the only event then make activity-by-strategy from it right now.
            if len(window_events) == 1:
                activitybs = self._make_activitybs_between_events(window_events, key)
                if activitybs is not None:
                    activitiesbs.append(activitybs)
                metrics.incr("1 separated by density activity-by-strategy-es with same data")
                continue
            # Prepare to iterate events to separate a-b-s-es.
            cluster = []
            len_events = len(window_events)

            for i, event in enumerate(window_events):
                cluster.append(event)

                # Calculate gap to next event, if there is a next event.
                if i < len_events - 1:
                    next_event = window_events[i + 1]
                    gap_duration = next_event.timestamp - (event.timestamp + event.duration)
                    # Decide that cluster is big enough if gap is bigger than min activity duration or density is low.
                    is_large_gap = gap_duration.seconds > MIN_ACTIVITY_DURATION_SEC
                    if not is_large_gap:
                        # Calculate projected density of the current cluster i.e. like if the next event added.
                        projected_duration = (
                            sum(x.duration.total_seconds() for x in cluster) + next_event.duration.total_seconds()
                        )
                        projected_time_span = (
                            (next_event.timestamp + next_event.duration) - cluster[0].timestamp
                        ).total_seconds()
                        projected_density = projected_duration / projected_time_span
                        is_large_gap = projected_density < MIN_DENSITY_FOR_SPARCE_INTERVALS
                    # If gap with the next event is too big then convert cluster to a-b-s and start new cluster.
                    if is_large_gap:
                        activitybs = self._make_activitybs_between_events(cluster, key)
                        if activitybs is not None:
                            activitiesbs.append(activitybs)
                        cluster = []
            # Add the last cluster if not empty.
            if cluster:
                activitybs = self._make_activitybs_between_events(cluster, key)
                if activitybs is not None:
                    activitiesbs.append(activitybs)
            metrics.incr(f"{len(activitiesbs)} separated by density activity-by-strategy-es with same data")
        return StrategyApplyResult(self.current_strategy, activitiesbs, self.current_metrics, self.current_id)

    def _aggregate_events_with_few_sliding_windows_by_optics(self) -> StrategyApplyResult:
        """
        Produces overlapping activity-by-strategy-es basing on theirs density on time scale.
        In details it first separates events with "same data" into windows (as big as needed).
        Next in each window it simplifies intervals to middlepoints and
        runs OPTICS algorithm - https://en.wikipedia.org/wiki/OPTICS_algorithm
        to find clusters. Note that it works quite bad because it looses
        """
        metrics = self.current_metrics
        windows: Dict[Tuple, List[Event]] = self._separate_events_per_windows(self.current_events)
        # Analyse gaps in each window.
        activitiesbs: List[ActivityByStrategy] = []
        for key, window_events in windows.items():
            # If it is the only event then make activity-by-strategy from it right now.
            if len(window_events) == 1:
                activitybs = self._make_activitybs_between_events(window_events, key)
                if activitybs is not None:
                    activitiesbs.append(activitybs)
                metrics.incr("1 separated by density activity-by-strategy-es with same data")
                continue
            # Convert list of events into 1D list of midpoints.
            dist_matrix = np.zeros((len(window_events), len(window_events)))
            for i, e1 in enumerate(window_events):
                for j, e2 in enumerate(window_events):
                    if i == j:
                        dist_matrix[i][j] = 0
                        continue
                    if e1.timestamp < e2.timestamp:
                        dist_matrix[i][j] = (e2.timestamp - (e1.timestamp + e1.duration)).total_seconds()
                    else:
                        dist_matrix[i][j] = (e1.timestamp - (e2.timestamp + e2.duration)).total_seconds()
            # FYI: with min_samples bigger we are getting bigger activities.
            clustering = DBSCAN(min_samples=2, eps=MIN_ACTIVITY_DURATION_SEC, metric="precomputed").fit(dist_matrix)
            event_clusters = {}
            for label, event in zip(clustering.labels_, window_events):
                event_clusters.setdefault(label, []).append(event)
            metrics.incr(f"{len(event_clusters)} separated by density activity-by-strategy-es with same data")
            for event_list in event_clusters.values():
                activitybs = self._make_activitybs_between_events(event_list, key)
                if activitybs is not None:
                    activitiesbs.append(activitybs)
        return StrategyApplyResult(self.current_strategy, activitiesbs, self.current_metrics, self.current_id)
