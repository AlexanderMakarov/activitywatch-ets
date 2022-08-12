import re
import collections
import dataclasses
from typing import Optional, Tuple, Callable, List, Dict


Event = collections.namedtuple('Event', ['bucket_id', 'timestamp', 'duration', 'data'])
"""
Lightweight representation of ActivityWatcher event without "id" field but with "bucket_it" field to trach source.
"""


@dataclasses.dataclass
class Rule:
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

class EventKeyHandler:
    """
    Structure which binds multiple `Rule`-s to one `Event`'s specific key values (by regexp).
    In spite of it supports case with absent key in event data it is better to separate `EventKeyHandler`-s
    per bucket because set of keys differs between differnt bucket events.
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
        handlers which point to the rule.
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
