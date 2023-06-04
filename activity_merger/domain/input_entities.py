import re
import collections
import dataclasses
from typing import Optional, Tuple, Callable, List, Dict, Union


Event = collections.namedtuple('Event', ['bucket_id', 'timestamp', 'duration', 'data'])
"""
Lightweight representation of ActivityWatcher event without "id" field but with "bucket_it" field to trach source.
"""


@dataclasses.dataclass(eq=True, order=True)
class Rule:  # TODO remove
    """
    Structure which represents how to compare related event with others.
    :param key_pattern: Regexp pattern to handle intervals of. Make sense only in scope of specific `EventKeyHandler`.
    :param priority: Priority of the rule, greater value => more chance to be used among other rules on interval.
    Better to use unique priority values, otherwise conflicts will be solved unpredictably.
    Note that default priority for "afk" watcher 'afk' state is `AFK_RULE_PRIORITY` (500?), 'not-afk' rule - 1,
    for "watchdog" "enabled" state - `WATCHDOG_RULE_PRIORITY` (1000?). These constants are also used to compound
    intervals into "activities".
    :param subhandler: `EventKeyHandler` for the different key of event with its own set of rules.
    It allows to take into consideration several keys of the event.
    :param to_string: Lambda which produces rule description from "key pattern" regexp `re.Match`. If `None` then uses
    the full match (0 group), if returns `None` then doesn't count.
    :param skip: Flag to skip this activity from reports. May be used for "non working" activities.
    :param is_placeholder: Flag for "service" rules. They are useless for report and usually 0 priority.
    Will cause some hints about necessity of new rule to cover related acitivity.
    :param merge_next: Flag to merge current interval with the next interval rule. Useful for 'new browser tab' like
    activities (i.e. it is impossible to reveal activity from related event but it is part of next event activity).
    """

    key_pattern: str
    priority: int
    subhandler: 'EventKeyHandler' = None
    to_string: Callable[[str], str] = None
    skip: bool = False
    is_placeholder: bool = False
    merge_next: bool = False

    def get_description(self, match: re.Match) -> Optional[str]:
        """
        :return: Description of activity represented by rule or `None` if description should be set by subhandlers.
        """
        return match.group(0) if self.to_string is None else self.to_string(match)

    def __repr__(self) -> str:
        desc = ""
        if self.subhandler:
            desc += f", with subhandler {self.subhandler}"
        if self.skip:
            desc += ", to skip"
        if self.merge_next:
            desc += ", to merge with next interval"
        if self.is_placeholder:
            desc += ", placeholder"
        return f"Rule '{self.key_pattern}', priority={self.priority}{desc}"

    def __hash__(self) -> int:
        return hash((self.key_pattern, self.skip, self.merge_next, self.is_placeholder, hash(self.subhandler)))


class Rule2:
    __slots__ = ('_pattern', '_regexp', 'priority', 'to_string', '_desc',
                 '_is_skip', '_is_merge_next', '_is_placeholder',
                 '_parent', '_data_key', '_subrules')

    def __init__(self, pattern: str, priority: int, to_string: Union[Callable[[str], str], str] = None):
        self._pattern = pattern
        self._regexp = re.compile(pattern)
        self.priority = priority
        self.to_string = to_string
        self._desc = None
        self._is_skip = False
        self._is_merge_next = False
        self._is_placeholder = False
        self._parent = None
        self._data_key = None
        self._subrules = None

    def __hash__(self):
        return hash(self._pattern, self.priority, self._desc,
                    self._is_skip, self._is_merge_next, self._is_placeholder,
                    self._data_key)

    def skip(self) -> 'Rule2':
        if self._is_merge_next or self._is_placeholder:
            raise ValueError(f"{self} can't be skipped additionally")
        self._is_skip = True
        return self

    def merge_next(self) -> 'Rule2':
        if self._is_skip or self._is_placeholder:
            raise ValueError(f"{self} can't be merged with next interval additionally")
        self._is_merge_next = True
        return self

    def placeholder(self) -> 'Rule2':
        if self._is_skip or self._is_merge_next:
            raise ValueError(f"{self} can't be a placeholder additionally")
        self._is_merge_next = True
        return self

    def with_subrules(self, key: str, subrules: List['Rule2']):
        self._data_key = key
        self._subrules = subrules
        for rule in subrules:
            rule._parent = self
        return self

    @property
    def is_skip(self) -> bool:
        return self._is_skip

    @property
    def is_merge_next(self) -> bool:
        return self._is_merge_next

    @property
    def is_placeholder(self) -> bool:
        return self._is_placeholder

    def __repr__(self) -> str:
        desc = ""
        if self._subrules:
            desc += f", with {len(self._subrules)} subrules by '{self._data_key}' key"
        if self._is_skip:
            desc += ", to skip"
        if self._is_merge_next:
            desc += ", to merge with next interval"
        if self._is_placeholder:
            desc += ", placeholder"
        return f"Rule '{self._pattern}', priority={self.priority}{desc}"

    def find_rule_for_event(self, event: Event) -> 'Rule2':
        """
        Find rule for the even in a recursive way passing down to subrules.
        :param event: Event to find "leaf" rule for.
        :return: Rule which more precisely matches event or `None` if there are no rule matching event.
        """
        # If it is "top" level rule then try match bucket ID.
        if self._parent is None:
            if not self.is_match(event.bucket_id):
                return None
        # Next (or instead) check if there are subrules to redirect matching to.
        if self._subrules:
            data_value = event.data.get(self._data_key)
            # Check that event has data to check by subrules.
            if data_value:
                for rule in self._subrules:
                    # If subrule matches value then pass next evaluation to it.
                    if rule.is_match(data_value):
                        return rule.find_rule_for_event(event)
        # If rule doesn't have subrules or event doesn't have data under required key then it is exact rule to handle this event.
        return self

    def is_match(self, value: str) -> bool:
        match = self._regexp.match(value)
        if not match:
            return False
        # If matched then need to update inner description.
        if self._parent is None:
            # If rule doesn't have parent then this rule is checked for bucket name.
            self._desc = match.group(0) + " bucket,"
        elif self.to_string:
            # If `to_string` is specified then use it.
            if isinstance(self.to_string, str):
                self._desc = self.to_string
            else:
                self._desc = self.to_string(match)
        else:
            # Otherwise just put the whole value matched by regexp.
            self._desc = match.group(0)
        return True

    def get_activity_description(self) -> str:
        """
        Builds description of an activity which this rule is created for.
        Uses `desc` value obtained via `to_string` methods from itself and up on tree of rules it is placed in.
        For example if rule is created as
        `Rule('bucket_x') -> Rule('app', to_string=lambda x: f'app {x.group(1),'}) -> Rule('Zoom.*', to_string='Zoom')`
        then it returns something like "x bucket, app y, Zoom".
        :return: Description of activity represented by rule.
        """
        if self._parent:
            return self._parent.get_activity_description() + " " + self._desc
        return self._desc


class EventKeyHandler:
    """
    Structure which binds multiple `Rule`-s to one `Event`'s field value by regexp.
    """

    def __init__(self, key: str, rules: List[Rule], to_str_keys: Optional[List[str]] = None) -> None:
        """
        Default constructor.
        :param key: Key from ActivityWatch event to choose rules basing on.
        :param rules: List of `Rule` objects to handle differrent "key" values.
        :param to_str_keys: List of keys to make `to_str` implementation basing on. If not specified then base
        event key value is used.
        """
        self.key = key
        self.rules: Dict[re.Pattern, Rule] = dict((re.compile(rule.key_pattern), rule) for rule in rules)
        self.to_str_keys = to_str_keys

    def __repr__(self) -> str:
        return f"EventKeyHandler(key={self.key}, rules_len={len(self.rules)})"

    def get_rule(self, event: Event) -> Optional[Tuple[Rule, List[str]]]:
        """
        Searches rule for specified event.
        :param event: Event to find rule for.
        :return: `Rule` handling specified event and ordered list of matching key value description-s in order of rule
        handlers which point to the rule. If there are no such rule then returns two None values.
        """
        value = event.data[self.key]
        descriptions = []
        matched_rule = None
        for (regex, rule) in self.rules.items():
            match = regex.match(value)
            if match:
                description = rule.get_description(match)
                if description:
                    descriptions.append(description)
                matched_rule = rule
                break
        if matched_rule and matched_rule.subhandler:
            matched_rule, subhandler_descriptions = matched_rule.subhandler.get_rule(event)
            descriptions.extend(subhandler_descriptions)
        return matched_rule, descriptions

    def __hash__(self) -> int:
        return hash((self.key) + (r.__hash__() for r in self.rules))
