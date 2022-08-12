import datetime
from ..domain.input_entities import Event
from ..config.config import CURRENT_TIMEZONE


def event_data_to_str(event: Event):
    if not event:
        return 'null'
    return str(event.data)


def event_to_str(event: Event):
    return f"{event.timestamp.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{(event.timestamp + event.duration).astimezone(CURRENT_TIMEZONE):%H:%M:%S}("\
           f"{event_data_to_str(event)})"


def from_start_to_end_to_str(obj) -> str:
    return f"{obj.start_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"\
           f"..{obj.end_time.astimezone(CURRENT_TIMEZONE):%H:%M:%S}"


def seconds_to_int_timedelta(seconds: float) -> str:
    return datetime.timedelta(seconds=int(seconds))
