import re
import collections
from typing import Optional, Tuple, Callable, List, Dict, Union


Event = collections.namedtuple('Event', ['bucket_id', 'timestamp', 'duration', 'data'])
"""
Lightweight representation of ActivityWatcher event without "id" field but with "bucket_it" field to trach source.
"""


class Rule2:
    """
    Structure to match ActivityWatch events, tie them with specific behavior, name activities with them.
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
        self.__is_merge_next = True
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
        return f"Rule '{self.__pattern}', priority={self.priority}{desc}"

    def find_rule_for_event(self, event: Event) -> Tuple['Rule2', str]:
        """
        Find rule for the even in a recursive way passing down to subrules.
        :param event: Event to find "leaf" rule for.
        :return: Tuple with rule which more precisely matches event in the graph
        and description of matched part if rule was found.
        """
        # If it is "top" level rule then additionally to other checks try match itself with event's bucket ID.
        description = None
        if self.__parent is None:
            matched, description = self.is_match(event.bucket_id)
            if not matched:
                return None, None
        # Next (or instead) check if there are subrules to redirect matching to.
        # Assume that this rule match input event already because someone called it.
        if not self.__subrules:
            return self, description  # It is leaf node.
        # Check if subrules won't be able handle this event.
        data_value = event.data.get(self.__data_key)
        if not data_value:
            return None, None  # Subrules won't be able match specified event - stop to search.
        for rule in self.__subrules:
            matched, subrule_description = self.is_match(data_value)
            # If subrule matches value then pass next evaluation to it.
            if matched:
                subrule, subrule_description = rule.find_rule_for_event(event)
                # Determine result rule.
                rule = subrule if subrule else self
                # Determine result description. Any part may be None.
                if description is not None:
                    if subrule_description is not None:
                        description += " " + subrule_description
                else:
                    description = subrule_description
                return rule, description
        # If subrules weren't able match event then current rule is a "leaf" rule to handle this event.
        return self, description

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
        if self.__parent is None:
            # If rule doesn't have parent then this rule is checked for bucket name.
            description = match.group(0) + " bucket,"
        elif self.to_string:
            # If `to_string` is specified then use it.
            if isinstance(self.to_string, str):
                description = self.to_string
            else:
                description = self.to_string(match)
        else:
            # Otherwise just put the whole value matched by regexp.
            description = match.group(0)
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


class EventKeyHandler:
    """
    Structure which binds multiple `Rule`-s to one `Event`'s field value by regexp.
    """

    def __init__(self, key: str, rules: List[Rule2], to_str_keys: Optional[List[str]] = None) -> None:
        """
        Default constructor.
        :param key: Key from ActivityWatch event to choose rules basing on.
        :param rules: List of `Rule` objects to handle differrent "key" values.
        :param to_str_keys: List of keys to make `to_str` implementation basing on. If not specified then base
        event key value is used.
        """
        self.key = key
        self.rules: Dict[re.Pattern, Rule2] = dict((re.compile(rule.key_pattern), rule) for rule in rules)
        self.to_str_keys = to_str_keys

    def __repr__(self) -> str:
        return f"EventKeyHandler(key={self.key}, rules_len={len(self.rules)})"

    def get_rule(self, event: Event) -> Optional[Tuple[Rule2, List[str]]]:
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
