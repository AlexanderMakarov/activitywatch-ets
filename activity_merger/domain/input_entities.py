import dataclasses
import collections
import enum
from typing import Tuple, Callable, List, Union


Event = collections.namedtuple("Event", ["bucket_id", "timestamp", "duration", "data"])
"""
Lightweight representation of ActivityWatcher event without "id" field but with "bucket_it" field to trach source.
"""


class ActivityBoundaries(enum.Enum):
    """Enum of possible activity boundary rules."""

    STRICT = enum.auto()
    """ Activity boundaries is not a subject of change. """

    START = enum.auto()
    """ Activity start is strict but the end may be chopped. """

    END = enum.auto()
    """ Activity end is strict but the start may be chopped. """

    DIM = enum.auto()
    """ Activity boundaries are not strict and may be changed. """

    @classmethod
    def from_str(cls, name: Union[str, "ActivityBoundaries"]) -> "ActivityBoundaries":
        """Constructs instance from the name if string is given."""
        if isinstance(name, ActivityBoundaries):
            return name
        elif isinstance(name, str):
            return cls[name.upper()]
        else:
            raise ValueError(f"Invalid '{name}' value is provided as an ActivityBoundaries.")


@dataclasses.dataclass(frozen=True)
class Strategy:
    """
    Structure representing how to aggregate events from the one source of events (i.e. watcher or exporter).
    All properties with "in_" prefix affects only logic of grouping events into "activites" inside strategy.
    While properties with "out_" prefix affects only logic of merging "activites" from different strategies,
    i.e. represents "priority" of "features" of this events source comparing with other sources.
    """

    name: str
    """
    Name of events source to be provided into resulting activity description. Expected to be short.
    """

    bucket_prefix: str
    """
    Prefix for buckets to analyze with this activity.
    """

    in_each_event_is_activity: bool
    """
    Flag that each event is the separate activity. Overrides all other "in_*" flags.
    """

    in_events_density_matters: bool
    """
    Flag that need to separate events by density.
    Like if there is big time gap between equal (by data) events then they represent different activities,
    but if such events are close to each other then it is the same activity.
    """

    in_activities_may_overlap: bool
    """
    Flag that events from different activities may alternate with each other but still represent few
    overlapping activities. Note that "out_*" parameters and logic around them will make activities
    not overlapping later on, but on the "in_*" stage better to have as much as possible guesses about
    probable activities.
    """

    in_group_by_keys: List[Tuple[str]]
    """
    Keys to separte events into the windows/activities.
    Some events may represent one activity if the have the same value in the only one or few `data` keys,
    if take into the account all fields then events will be unique.
    For example for JIRA bucket events with the same value in "ticket" key may represent the same actvity -
    aka "worked on the ticket X".
    Note that order of tuples in the list is important - event will be added to the activity/window with
    values for the key(s) from the 2nd tuple if its "data" doesn't have keys from the 1st tuple and so on.
    """

    in_skip_key_value: List[Tuple[str]]
    """
    Map of key/value pairs (i.e. dictionary) to skip and don't make activities from.
    Useful for "uknown" events which are bad source of information and may be duplicated by more
    meaningful events/activities.
    """

    out_self_sufficient: bool
    """
    True means activity from this bucket may overlaps others. Any conflicts leads to errors.
    """

    out_only_not_afk: bool
    """
    True means that activities will be cut by AFK data on corresponding host.
    """

    out_activity_boundaries: ActivityBoundaries
    """
    Means which boundaries of activity are strict.
    """

    out_produces_good_activity_name: bool
    """
    Flag that strategy events may provide good resulting activity name.
    """

    out_activity_name_sentence_builder: Callable[[List[Tuple[str, str]]], str]
    """
    Function which builds one sentence for the resulting activity name from the list of
    key-value pairs aggregated from ActivityWatch event's "data" dictionaries which were used to merge
    multiple events into one activity. Should start from upper case letter and end with the point.
    If `None` then uses simple "dict to string" logic.
    If function returns `None` or empty string then strategy won't contribute to the resulting activity name.
    """

    def __post_init__(self):
        # Convert out_activity_boundaries from string to `ActivityBoundaries`.
        # https://stackoverflow.com/a/54119384/1535127
        object.__setattr__(self, "out_activity_boundaries", ActivityBoundaries.from_str(self.out_activity_boundaries))

    __properties = [
        "bucket_prefix",
        "in_each_event_is_activity",
        "in_events_density_matters",
        "in_activities_may_overlap",
        "in_group_by_keys",
        "out_self_sufficient",
        "out_only_not_afk",
        "out_activity_boundaries",
        "out_produces_good_activity_name",
        "out_activity_name_sentence_builder",
    ]

    def __repr__(self) -> str:
        desc = [f"Strategy '{self.name}':"]
        for prop in Strategy.__properties:
            desc.append(f"  {prop} = {getattr(self, prop)}")
        return "\n".join(desc)
