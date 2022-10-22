#!/usr/bin/env python3
import datetime
from typing import Dict, List, Callable, Any, Tuple
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException
import socket
import aw_client
import aw_core.models as awmodels

from activity_merger.config.config import LOG
from activity_merger.helpers.helpers import setup_logging
from activity_merger.domain.input_entities import Event


NAME = 'outlook_aw_events_scraper'
SCREENSHOT_NAME = f"{NAME}-events-screenshot.png"
SCREENSHOT_FAIL_NAME = f"{NAME}-fail-screenshot.png"
CALENDAR_TODAY_URL_SUFFIX = "/#path=/calendar/view/Day"
BUCKET_ID = f'{NAME}_{socket.gethostname()}'
MAX_SCROLL_BACK = 31


# JS snippet to get location in browser as a hint on mouse move.
# document.onmousemove = function(e){
# let r = e.target.getBoundingClientRect();
# e.target.title = e.target.innerHTML+"#x="+r.x+"y="+r.y;
# };


def start_firefox_under_existing_profile(profile: str, page: str,  headless: bool = True) -> WebDriver:
    options = webdriver.FirefoxOptions()
    firefox_args = [ # moz:firefoxOptions
        '--safe-mode',
        '--new-tab', page,
        '--start-maximized',
    ]
    options._arguments.extend(firefox_args)
    # Note that '--profile' inside options._arguments causes alert "window with such profile is already opened".
    options.profile = profile
    options.headless = headless
    LOG.info(f"Wait few seconds/minutes - new Firefox window is starting with profile '{profile}'")
    return webdriver.Firefox(options=options)  # Note that all other params are deprecated.


def call_web_element_with_fail_handling(description: str, container: WebElement, func: Callable,
        check_result_not_empty: bool = True) -> Any:
    assert container, f"call_web_element_with_fail_handling: Can't get {description} from empty container: {container}"
    try:
        result = None
        if check_result_not_empty:
            result = func(container)
            # Note that is case of 'find_elements' we have to check array so don't rely on NoSuchElementException.
            assert result, f"Can't find {description} on {container}"
        else:
            try:
                result = func(container)
            except NoSuchElementException:
                pass  # We don't care if element wasn't found.
        return result
    except Exception as e:
        container.screenshot(SCREENSHOT_FAIL_NAME)
        LOG.error(f"Can't find {description} on {container}. Container inner HTML:" +
                  "\n" + container.get_attribute('innerHTML'))
        raise e


def _wait_on_page(driver: WebDriver, xpath: str) -> Any:
    try:
        return WebDriverWait(driver, 1*60).until(
            EC.visibility_of_all_elements_located((By.XPATH, xpath)),
            f"Can't find/wait Outlook Web 'Calendar' page elements. Make sure that you've logged in on main window."
        )
    except Exception as e:
        driver.get_screenshot_as_file(SCREENSHOT_FAIL_NAME)
        LOG.error(e.text + ". Page HTML:\n" + driver.page_source)
        raise e


def _scroll_to_day(back_days: int, date_label: str, driver: WebDriver):
    if back_days > 0 or date_label is not None:
        SCROLL_BACK_XPATH_SELECTOR = "span[contains(@class,'ms-Icon--chevronLeft')]"
        DAY_SCROLL_AREA_XPATH = f"//{SCROLL_BACK_XPATH_SELECTOR}/../../../div[count(button)=4]"
        scroll_area = None
        date_label_span = None
        scrolls_back = 0
        while scrolls_back < MAX_SCROLL_BACK:
            if scroll_area:  # Check label only after the first iteration.
                date_label_span: WebElement = call_web_element_with_fail_handling(
                    "'Current day' label",
                    scroll_area,
                    lambda x: x.find_element(
                        By.XPATH,  # Note that page may contains a lot of elements with only one visible.
                        "button[contains(@class,'o365button')]"
                            "/span[contains(@class,'o365buttonLabel') and not(contains(@style,'none'))]"
                    ),
                    False
                )
                if date_label and date_label_span and date_label_span.text == date_label:
                    break
            if scrolls_back == back_days:
                if date_label:  # I.e. if label was checked but doesn't match.
                    raise AssertionError(f"After {scrolls_back} days back '{date_label} wasn't found.")
                else:
                    break  # I.e. just assume/believe that we are on the required date right now.
            scroll_area = _wait_on_page(driver, DAY_SCROLL_AREA_XPATH)[0]
            scroll_back: WebElement = call_web_element_with_fail_handling(
                "'Open previous day' button",
                scroll_area,
                lambda x: x.find_element(By.XPATH, f"button/{SCROLL_BACK_XPATH_SELECTOR}/..")
            )
            current_day_desc = f"'{date_label_span.text}'" if date_label_span else "<first_or_undefined>"
            LOG.info(f"From {current_day_desc} page clicking on '{scroll_back.get_attribute('ariaLabel')}' button"
                     f"(rect={scroll_back.rect}) to shift on previous day.")
            scroll_back.click()
            scrolls_back += 1


def _get_hour_points(container_web_element: WebElement):
    hour_spans = call_web_element_with_fail_handling(
        "containers of hours",
        container_web_element,
        lambda x: x.find_elements(
            By.XPATH,
            ".//span[contains(@class,'ms-font-m')"
            " and contains(@class,'semilight')"
            " and contains(@class,'ms-font-color-neutralPrimary')]"
        )
    )
    # Note that WebElement.rect/location are based on Element.getBoundingClientRect() though are adding padding
    # and border-width. See https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect
    # Also not that "padding-top" is returned as "10px" so need to parse only digits.
    hour_points = [
        (
            int(x.text),
            x.rect['y'] + int(''.join(c for c in x.value_of_css_property("padding-top") if c.isdigit()))
        )
        for x in hour_spans if x.text.isnumeric()
    ]
    if len(hour_points) < 24:
        LOG.error(container_web_element.get_attribute('innerHTML'))
        assert False, "Can't find all 24 'hour' points on the screen."
    return hour_points


def _find_start_and_duration(hour_points: List[Tuple[int, float]], event_div: WebElement, event_name: str,
        event_date: datetime.datetime) -> Tuple[datetime.datetime, datetime.timedelta]:
    """
    Roughly (by web element coordinates) finds out start and duration of event.
    :param hour_points: List of tuples [hour value, relevant div rectangle dict].
    :param event_div: Web element to find hour for.
    :param event_name: Name of the element for logs.
    :param event_date: Date to set for event.
    :return: Tuple [event start, event duration].
    """
    start_y = event_div.rect['y']
    end_y = start_y + event_div.rect['height'] 
    # There are following option here:
    # 1) Event starts on hour start, ends inside it.
    # 2) Event starts on hour start, ends in some following hour.
    # 3) Event starts inside hour, ends inside it.
    # 4) Event starts inside hour,ends in some following hour.
    index_started_in = None
    index_ended_in = None
    is_find_start = True
    for i, hour_point in enumerate(hour_points):  # Here are always 24 elements, it is fast.
        if is_find_start:
            if hour_point[1] < start_y:
                continue  # Just iterate until find hour where event started.
            index_started_in = i if hour_point[1] == start_y else i - 1
            is_find_start = False
            index_ended_in = index_started_in  # Start with assumption that event ends in the same hour.
        if hour_point[1] > end_y:  # Just iterate until find where event ended.
            break
        index_ended_in = i
    if index_started_in is None:
        assert False, f"Can't find start hour for element '{event_name}' with y coordinate {start_y}"\
                      f" among hours: {hour_points}"
    if index_ended_in is None:
        assert False, f"Can't find end hour for element '{event_name}' with bottom y coordinate {end_y}"\
                      f" among hours: {hour_points}"
    # Measure size of hour in pixels.
    hour_height = hour_points[1][1] - hour_points[0][1]
    # Calculate start time.
    hour_started_in = hour_points[index_started_in]
    start_hour_ratio = float(start_y - hour_started_in[1]) / hour_height
    start_time = datetime.datetime(year=event_date.year, month=event_date.month, day=event_date.day,
                                   hour=hour_started_in[0], minute=int(start_hour_ratio * 60), second=0).astimezone()
    # Calculate end time.
    hour_ended_in = hour_points[index_ended_in]
    end_hour_ratio = float(end_y - hour_ended_in[1]) / hour_height
    end_time = datetime.datetime(year=event_date.year, month=event_date.month, day=event_date.day,
                                 hour=hour_ended_in[0], minute=int(end_hour_ratio * 60), second=0).astimezone()
    return start_time, end_time - start_time


def get_events_from_owa(profile_abs_path: str, owa_url: str, headless: bool,
        events_date: datetime.datetime = None, back_days: int = 0, date_label: str = None) -> List[Event]:
    """
    Scapes events from OWA page. Starts Firefox browser with OWA page "Calendar for a day" in headless mode (if need),
    scrolls to requered date if need, finds all events
    and calculates theirs start date and duration by coordinates on the screen.
    Requires Firefox profile to be opened already with OWA page authorized - it doesn't handle authentication.
    :param profile_abs_path: Absolute path to Firefox profile.
    On Linux looks like '/home/{username}/.mozilla/firefox/808favcvs.default-release/'.
    :param owa_url: URL to Web (MS Office Web Apps) Outlook. Page where email is opens.
    May look like 'https://mail.company.com/owa'.
    :param headless: Flag to open Firefox window in the headless mode. Doesn't make it faster at all.
    :param events_date: Date to set for events. Calculated based on "today" if 'back days' parameter is specified.
    Should contain time zone info and match date in "date label" parameter.
    :param back_days: Optional. Number of days back.
    :param date_label: Date label on page for "tap back until open" logic.
    Value depends from language, regions, settings, etc. Replaces or supports 'back_days' parameter.
    Should match 'date' parameter.
    :return: List of `Event`-s parsed for specified date.
    """
    LOG.info(f"Starting Firefox with profile '{profile_abs_path}'."
             " Note that you need to have OWA opened and authenticated under this profile, otherwise it would fail.")
    # In "with UI" mode some parsing on my machine takes 65s, in headless - the same 65s.
    page_url = owa_url + CALENDAR_TODAY_URL_SUFFIX
    driver = start_firefox_under_existing_profile(profile_abs_path, page_url, headless)
    LOG.info(
        f"Firefox started, waiting loading of '{page_url}'. Don't scroll/click on it's window!")
    # Set 'events_date' to search if not specified.
    if events_date is None:
        if back_days > 0:
            events_date = datetime.datetime.today().astimezone() - datetime.timedelta(days=back_days)
    # Check that first need scroll to day and perform scrolls.
    _scroll_to_day(back_days, date_label, driver)
    # Start parse events.
    # Note that page may contian few containers, espectially if t)here are no events this day.
    events_containers = _wait_on_page(
        driver,
        "//div[contains(@class,'scrollContainer') and not(contains(@style,'none'))]"
            "/div[@role='presentation' and not(contains(@style,'none'))]"
    )
    events_container = next(x for x in events_containers if x.size['height'] > 100)
    driver.implicitly_wait(1)  # Wait until all events are rendered on the container. Immediate check returns only first event(s).
    events_container.screenshot(SCREENSHOT_NAME)
    LOG.info(f"Page of required day is opened, scrapping events. See screenshot {SCREENSHOT_NAME}.")
    # Note that these div-s also contains elements with calendar(s) name.
    TYPE_DIV_XPATH_SELECTOR = "div[contains(@class,'calendarBusy') or contains(@class,'calendarTentative')"\
                                  " or contains(@class,'calendarBusy')]"
    probable_event_divs: List[WebElement] = call_web_element_with_fail_handling(
        "probable containers of event rectangles",
        events_container,
        lambda x: x.find_elements(By.XPATH, f".//{TYPE_DIV_XPATH_SELECTOR}/..")
    )
    hour_points: List[int, float] = _get_hour_points(events_container)
    events: List[Event] = []
    for probable_event_div in probable_event_divs:
        # Search for the "time" rectangle of event.
        event_div = call_web_element_with_fail_handling(
            '"time" rectangle',
            probable_event_div,
            lambda x: x.find_element(By.XPATH, f".//{TYPE_DIV_XPATH_SELECTOR}"),
            False
        )
        if not event_div:
            LOG.info(f"Skipping '{probable_event_div.text}' because event rectangle is not placed in it.")
            continue
        # Search for event data container. They are wrapped into 'div' next to 'event_div' with 2 types of structure:
        # 1) Series of events - inside 2 div-s where 1st contains 3 span-s with name, location, sender.
        # 2) One time event - inside 1 span with name, 2 div-s where 1st contains 2 span-s with location, sender.
        event_data_container = call_web_element_with_fail_handling(
            "div with event data",
            probable_event_div,
            lambda x: x.find_element(By.XPATH, ".//div[(count(span)=3) or (count(span)=1 and count(div)=2)]"),
            False
        )
        if not event_data_container:
            LOG.info(f"Skipping '{probable_event_div.text}' because event data is not placed in it.")
            continue
        # Event is found - grab/calculate data from it.
        LOG.info(f"Found event '{event_data_container.text}' in rect {event_div.rect}.")
        event_elements: List[WebElement] = call_web_element_with_fail_handling(
            "elements of event",
            event_data_container,
            lambda x: x.find_elements(By.XPATH, ".//span")  # In both cases elements are placed in span-s in order.
        )
        div_classes = event_div.get_attribute("class")
        event_type = None  # Value of class from TYPE_DIV_XPATH_SELECTOR.
        if "calendarTentative" in div_classes:
            event_type = "tentative"
        elif "calendarFree" in div_classes:
            event_type = "free"
        elif "calendarBusy" in div_classes:
            event_type = "busy"
        start_time, duration = _find_start_and_duration(hour_points, event_div, event_data_container.text, events_date)
        # Assemble event.
        events.append(Event(BUCKET_ID, start_time, duration, {
            'type': event_type,
            'name': event_elements[0].text,
            'location': event_elements[1].text,
            'sender': event_elements[2].text,
        }))
    driver.quit()
    return events


def upload_events(events: List[Event]):
    # Convert into ActivityWatch clients.
    aw_events = [awmodels.Event(timestamp=x.timestamp, duration=x.duration, data=x.data) for x in events]
    # Build client, check than bucket is created and insert events.
    client = aw_client.ActivityWatchClient(NAME)
    client.create_bucket(BUCKET_ID, event_type="owa365.calendar.event")  # Will return 304 if bucket exists.
    return client.insert_events(BUCKET_ID, aw_events)


def main():
    # TODO ask arguments
    profile_abs_path = '/home/i4ellendger/.mozilla/firefox/21357bye.default-release/'
    events = get_events_from_owa(profile_abs_path, "https://mail.akvelon.com/owa/", True, back_days=2)
    # events = []
    # events.append(Event(bucket_id='outlook_aw_events_scraper_i4ellendger-Latitude-5511', timestamp=datetime.datetime(2022, 10, 20, 10, 0).astimezone(), duration=datetime.timedelta(seconds=5280), data={'type': 'busy', 'name': 'Armenian language classes', 'location': 'Zoom', 'sender': 'Ekaterina Cheshuina'}))
    # events.append(Event(bucket_id='outlook_aw_events_scraper_i4ellendger-Latitude-5511', timestamp=datetime.datetime(2022, 10, 20, 12, 0).astimezone(), duration=datetime.timedelta(seconds=840), data={'type': 'tentative', 'name': 'PLAN team stand up', 'location': 'https://intapp.zoom.us/j/96652217786?pwd=MFFodUpnbis0VXdwM1c5STlSK0FtQT09', 'sender': 'Ivan Volkov'}))
    # events.append(Event(bucket_id='outlook_aw_events_scraper_i4ellendger-Latitude-5511', timestamp=datetime.datetime(2022, 10, 20, 15, 0).astimezone(), duration=datetime.timedelta(seconds=3480), data={'type': 'busy', 'name': 'Discuss results of PPT-4898 (Define approach to migrate jobs to SpringBatch)', 'location': 'https://us04web.zoom.us/j/73290816495?pwd=9WAMmHb2rQTVKa1J9ehJsrIeQfXZjP.1', 'sender': 'Alexey Semenov'}))
    # events.append(Event(bucket_id='outlook_aw_events_scraper_i4ellendger-Latitude-5511', timestamp=datetime.datetime(2022, 10, 20, 20, 0).astimezone(), duration=datetime.timedelta(seconds=1680), data={'type': 'busy', 'name': 'FW: [EXTERNAL] Weekly Cloud Release Meeting', 'location': '', 'sender': 'ENG - Cloud Status Team'}))
    LOG.info(f"Parsed {len(events)} events:" + "\n  " + "\n  ".join(str(x) for x in events))
    # Load events into ActivityWatcher
    data = upload_events(events)
    LOG.info(data)


if __name__ == '__main__':
    LOG = setup_logging()
    main()
