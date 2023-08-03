import dataclasses
import re
import collections
import enum
from typing import Tuple, Callable, List, Union


Event = collections.namedtuple('Event', ['bucket_id', 'timestamp', 'duration', 'data'])
"""
Lightweight representation of ActivityWatcher event without "id" field but with "bucket_it" field to trach source.
"""


class Rule2:
    """
    Structure to match ActivityWatch events, tie them with specific behavior, name activities with them.
    Note that `str(rule)` produces rule name, while `find_rule_for_event` and `is_match` produce name of the
    activity based on value (event) given as a parameter.

    :param pattern: Regexp pattern to handle some value. Value is determined in parent rule.
    If there is no parent (i.e. this rule is top level) then pattern should match name of the ActivityWatch bucket.
    :param priority: Priority of the rule, greater value => more chance to describe activity from underlying interval.
    Better to use unique priority values, otherwise conflicts will be solved unpredictably.
    Note that default priority for "afk" watcher 'afk' state is `AFK_RULE_PRIORITY` (500?), 'not-afk' rule - 1,
    for "watchdog" "enabled" state - `WATCHDOG_RULE_PRIORITY` (1000?). This way rule may create custom activity even
    from AFK or "watchdog" event.
    :param to_string: Way to describe activity represented by underlying interval.
    If string then is used as is.
    If lambda then takes `re.match(value)` result and expected to produce string.
    If `None` then uses the full match (0 group).
    Note that resulting activity name is assembled from all rules in the tree joined via space (" ") character.

    Use `with_subrules` to buld tree of rules and make more specific activities.
    See `skip`, `merge_next`, `placeholder` methods to add specific behavior to rules/activities.
    """
    __slots__ = ('__pattern', '__regexp', 'priority', 'to_string',
                 '__is_skip', '__is_merge_next', '__is_placeholder',
                 '__parent', '__data_key', '__subrules')

    def __init__(self, pattern: str, priority: int, to_string: Union[Callable[[str], str], str] = None):
        self.__pattern = pattern
        self.__regexp = re.compile(pattern)
        self.priority = priority
        self.to_string = to_string
        self.__is_skip = False
        self.__is_merge_next = False
        self.__is_placeholder = False
        self.__parent = None
        self.__data_key = None
        self.__subrules = None

    def __hash__(self):
        return hash((self.__pattern, self.priority,
                    self.__is_skip, self.__is_merge_next, self.__is_placeholder,
                    self.__data_key))

    def skip(self) -> 'Rule2':
        """
        Marks rule to skip underlying interval from reports. May be used for "non working" activities.
        """
        if self.__is_merge_next or self.__is_placeholder or self.__subrules:
            raise ValueError(f"{self} can't be skipped - conflicts with other behaviors")
        self.__is_skip = True
        return self

    def merge_next(self) -> 'Rule2':
        """
        Marks rule to merge underlying interval with the next interval to add to it's rule.
        Useful for 'new browser tab' like activities when it is impossible to reveal activity from related event
        because too few data and more details will be in the following event.
        Note that event under this rule amy be converted into decicated activity if it is the last interval in
        the analyzed period.
        """
        if self.__is_skip or self.__is_placeholder or self.__subrules:
            raise ValueError(f"{self} can't be merged with next interval - conflicts with other behaviors")
        self.__is_merge_next = True
        return self

    def placeholder(self) -> 'Rule2':
        """
        Marks "service" rules. They are useless for report and usually 0 priority.
        Causes hints about necessity of new rule to cover related acitivity.
        """
        if self.__is_skip or self.__is_merge_next or self.__subrules:
            raise ValueError(f"{self} can't be a placeholder - conflicts with other behaviors")
        self.__is_placeholder = True
        return self

    def with_subrules(self, key: str, subrules: List['Rule2']):
        """
        Builds tree from rules to handle different aspects of events.
        :param key: Key in `Event.data` dictionary to apply subrules on.
        :param subrules: List of rules to match value by key and configure theirs behavior.
        It is useful to end list of subrules with rule having '.*' pattern - it will catch all
        unmatched events and handle them at least somehow.
        """
        if self.__is_skip or self.__is_merge_next or self.__is_placeholder:
            raise ValueError(f"{self} can't add subrules - conflicts with other behaviors")
        self.__data_key = key
        self.__subrules = subrules
        for rule in subrules:
            rule.__parent = self
        return self

    @property
    def is_skip(self) -> bool:
        return self.__is_skip

    @property
    def is_merge_next(self) -> bool:
        return self.__is_merge_next

    @property
    def is_placeholder(self) -> bool:
        return self.__is_placeholder

    def __repr__(self) -> str:
        desc = ""
        if self.__subrules:
            desc += f", with {len(self.__subrules)} subrules by '{self.__data_key}' key"
        if self.__is_skip:
            desc += ", to skip"
        if self.__is_merge_next:
            desc += ", to merge with next interval"
        if self.__is_placeholder:
            desc += ", placeholder"
        if self.__parent:
            desc += f", parent={self.__parent}"
        return f"Rule '{self.__pattern}', priority={self.priority}{desc}"

    def find_rule_for_event(self, event: Event) -> Tuple['Rule2', str]:
        """
        Find rule for the even in a recursive way passing down to subrules.
        :param event: Event to find "leaf" rule for.
        :return: Tuple with rule which more precisely matches event in the graph
        and description of matched part if rule was found.
        """
        description = None
        result = None
        # If it is "top" level rule then additionally to other checks try match itself with event's bucket ID.
        if self.__parent is None:
            matches, description = self.is_match(event.bucket_id)
            if not matches:
                return None, None
            else:
                result = self
        # Next (or instead) check if there are subrules to redirect matching to.
        if not self.__subrules:
            return result, description  # Nothing to check further.
        # Check if subrules won't be able handle this event.
        data_value = event.data.get(self.__data_key)
        if not data_value:
            return result, description  # Subrules won't be able match specified event - stop to search.
        # Check with subrules.
        for subrule in self.__subrules:
            matches, subrule_description = subrule.is_match(data_value)
            # If subrule matches value then pass next evaluation to it.
            if matches:
                # First update resutls.
                result = subrule
                if description is not None:
                    description += ", " + subrule_description
                else:
                    description = subrule_description
                # Try to search recursively.
                recursive_result, recursive_description = subrule.find_rule_for_event(event)
                # If recursive search found something else then update results with it.
                if recursive_result:
                    result = recursive_result
                    description += " " + recursive_description
                break
        return result, description

    def is_match(self, value: str) -> Tuple[bool, str]:
        """
        Checks if rule matches given value.
        :param value: Value to check.
        :return: Tuple with flag if value is matched
        and if yes then string with description of the part which was matched.
        """
        match = self.__regexp.match(value)
        if not match:
            return False, None
        description = None
        # If matched then need to update inner description.
        if self.to_string:
            # If `to_string` is specified then use it.
            if isinstance(self.to_string, str):
                description = self.to_string
            else:
                description = self.to_string(match)
        else:
            # Otherwise just put the whole value matched by regexp or it's last group.
            description = match.groups()[-1] if match.groups() else match.string
        return True, description

    def get_number_of_rules_in_tree(self) -> int:
        """
        :return: Number of `Rule`-s in a tree started with this rule.
        Returns at least 1 - for this rule.
        """
        result = 1
        if self.__subrules:
            for subrule in self.__subrules:
                if subrule:
                    result += subrule.get_number_of_rules_in_tree()
        return result


class ActivityBoundaries(enum.Enum):
    START = enum.auto()
    END = enum.auto()
    WHOLE = enum.auto()

    @classmethod
    def from_str(cls, name: Union[str, 'ActivityBoundaries']) -> 'ActivityBoundaries':
        """Constructs instance from the name if string is given."""
        if isinstance(name, ActivityBoundaries):
            return name
        elif isinstance(name, str):
            return cls[name.upper()]
        else:
            raise ValueError(f"Invalid '{name}' value is provided as an ActivityBoundaries.")


@dataclasses.dataclass
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
    [whole, start, end] - means which part of activity is trustable.
    """
    out_activity_name: str
    """
    [alone, auxiliary] - means whether activity name is trustable.
    """

    def __post_init__(self):
        self.out_activity_boundaries = ActivityBoundaries.from_str(self.out_activity_boundaries)

    __properties = [
        'bucket_prefix', 
        'in_each_event_is_activity', 
        'in_events_density_matters', 
        'in_activities_may_overlap',
        'in_group_by_keys',
        'out_self_sufficient',
        'out_only_not_afk',
        'out_activity_boundaries',
        'out_activity_name'
    ]

    def __repr__(self) -> str:
        desc = [f"Strategy '{self.name}':"]
        for prop in Strategy.__properties:
            desc.append(f"  {prop} = {getattr(self, prop)}")
        return '\n'.join(desc)
