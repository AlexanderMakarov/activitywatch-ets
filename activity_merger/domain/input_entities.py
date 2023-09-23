import dataclasses
import collections
import enum
from typing import Dict, Tuple, Callable, List, Union


Event = collections.namedtuple("Event", ["bucket_id", "timestamp", "duration", "data"])
"""
Lightweight representation of ActivityWatcher event without "id" field but with "bucket_it" field to trach source.
"""


class IntervalBoundaries(enum.Enum):
    """Enum of possible time interval boundaries rules."""

    STRICT = enum.auto()
    """ Interval boundaries is not a subject of change. """

    START = enum.auto()
    """ Interval start is strict, the end may be chopped. """

    END = enum.auto()
    """ Interval end is strict, the start may be chopped. """

    DIM = enum.auto()
    """ Interval boundaries are not strict and may be chopped. """

    @classmethod
    def from_str(cls, name: Union[str, "IntervalBoundaries"]) -> "IntervalBoundaries":
        """Constructs instance from the name if string is given."""
        if isinstance(name, IntervalBoundaries):
            return name
        elif isinstance(name, str):
            return cls[name.upper()]
        else:
            raise ValueError(f"Invalid '{name}' value is provided as an {cls.name}.")


@dataclasses.dataclass(frozen=True)
class Strategy:
    """
    Structure representing how to aggregate events from the one source of events (i.e. watcher or exporter).
    All properties with "in_" prefix affects only logic of grouping events into "activites" inside strategy.
    While properties with "out_" prefix affects only logic of merging "activites" from different strategies,
    i.e. represents "priority" of "features" of this events source comparing with other sources.
    NOTE: required only in a name and bucket prefix, all other parameters have default values like `None` and `False`.
    """

    name: str
    """
    Name of events source to be provided into resulting activity description. Expected to be short.
    """

    bucket_prefix: str
    """
    Prefix for buckets to analyze with this activity.
    """

    in_each_event_is_activity: bool = False
    """
    Flag that each event is the separate activity. Overrides all other "in_*" flags.
    """

    in_trustable_boundaries: IntervalBoundaries = "dim"
    """
    Means which boundaries of events are trustable. Supports: "strict", "start", "end", "dim" values.
    """

    in_events_density_matters: bool = False
    """
    Flag that need to separate events by density.
    Like if there is big time gap between equal (by data) events then they represent different activities,
    but if such events are close to each other then it is the same activity.
    """

    in_activities_may_overlap: bool = False
    """
    Flag that events from different activities may alternate with each other but still represent few
    overlapping activities. Note that "out_*" parameters and logic around them will make activities
    not overlapping later on, but on the "in_*" stage better to have as much as possible guesses about
    probable activities.
    """

    in_group_by_keys: List[Tuple[str]] = None
    """
    Specific keys to separate events into the windows/activities when related values are similar.
    By default events are grouped together if set of keys and values for all keys are identical.
    But some events producers may set different sets of keys.
    If provide list of possible key sets (as Python tuples) then only those keys would be checked and default
    behavior would be applied only not all keys are presented in the event.
    Note that order of key sets/tuples is important - if first set of keys was matched then all remained are ignored.
    """

    in_skip_key_values: Dict[str, any] = None
    """
    Map of key/value pairs (i.e. dictionary) to skip and don't make activities from.
    Useful for "uknown" events which are bad source of information and may be duplicated by more
    meaningful events/activities.
    """

    in_only_not_afk: bool = False
    """
    True means that events will be cut to appear only in not-AFK intervals.
    """

    in_only_if_window_app: List[str] = None
    """
    List of values in "app" key of OS Windows Manager events to cut current events to appear only inside.
    For example IDEA events are reported all time but real work in IDEA may happen
    only if currently active app is "jetbrains-idea".
    NOTE that this setting would work only if there is `("app",)` entry in `in_group_by_keys` property
    and `in_skip_key_values` doesn't filter out required values.
    """

    out_self_sufficient: bool = False
    """
    True means activity from this bucket may overlaps others. Any conflicts leads to errors.
    """

    out_produces_good_activity_name: bool = False
    """
    Flag that strategy events may provide good resulting activity name.
    """

    out_activity_name_sentence_builder: Callable[[List[Tuple[str, str]]], str] = None
    """
    Function which builds one sentence for the resulting activity name from the list of
    key-value pairs aggregated from ActivityWatch event's "data" dictionaries which were used to merge
    multiple events into one activity. Should start from upper case letter and end with the point.
    If `None` then uses simple "dict to string" logic.
    If function returns `None` or empty string then strategy won't contribute to the resulting activity name.
    """

    def __post_init__(self):
        # Convert in_trustable_boundaries from string to `ActivityBoundaries`.
        # https://stackoverflow.com/a/54119384/1535127
        object.__setattr__(self, "in_trustable_boundaries", IntervalBoundaries.from_str(self.in_trustable_boundaries))

    __repr_properties = [  # Used to specify properties for `__repr__(self)`.
        "bucket_prefix",
        "in_each_event_is_activity",
        "in_trustable_boundaries",
        "in_events_density_matters",
        "in_activities_may_overlap",
        "in_group_by_keys",
        "in_skip_key_values",
        "in_only_not_afk",
        "in_only_if_window_app",
        "out_self_sufficient",
        "out_produces_good_activity_name",
        "out_activity_name_sentence_builder",
    ]

    def __repr__(self) -> str:
        desc = [f"Strategy '{self.name}':"]
        for prop in Strategy.__repr_properties:
            desc.append(f"  {prop} = {getattr(self, prop)}")
        return "\n".join(desc)
