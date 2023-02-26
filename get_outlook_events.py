#!/usr/bin/env python3
import datetime
from typing import List, Callable, Any, Tuple
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException
import argparse
import contextlib
import unittest
import parameterized

from activity_merger.config.config import LOG, FIREFOX_PROFILE_PATH, OWA_SCRAPER_NAME, OWA_URL, OWA_BUCKET_ID,\
                                          OWA_MAX_SCROLL_BACK
from activity_merger.helpers.helpers import setup_logging, valid_date, upload_events
from activity_merger.domain.input_entities import Event


SCREENSHOT_NAME = f"{OWA_SCRAPER_NAME}-events-screenshot.png"
SCREENSHOT_FAIL_NAME = f"{OWA_SCRAPER_NAME}-fail-screenshot.png"
CALENDAR_TODAY_URL_SUFFIX = "/#path=/calendar/view/Day"


# JS snippet to get location in browser as a hint on mouse move.
# document.onmousemove = function(e){
# let r = e.target.getBoundingClientRect();
# e.target.title = e.target.innerHTML+"#x="+r.x+"y="+r.y;
# };


@contextlib.contextmanager
def start_firefox_under_existing_profile(profile: str, page: str,  headless: bool = True) -> WebDriver:
    options = webdriver.FirefoxOptions()
    firefox_args = [ # moz:firefoxOptions
        '--new-instance',  # --safe-mode is dangerous - it uninstalls all plugins from profile!
        '--new-tab', page,
        '--start-maximized',
    ]
    options._arguments.extend(firefox_args)
    # Note that '--profile' inside options._arguments causes alert "window with such profile is already opened".
    # But see https://stackoverflow.com/a/71604450/1535127 - it is the only way to open running profile aside.
    options.profile = profile
    options.headless = headless
    LOG.info(f"Wait few seconds/minutes - new Firefox window is starting with profile '{profile}'")
    driver = webdriver.Firefox(options=options)  # Note that all other params are deprecated.
    try:
        yield driver
    finally:
        driver.quit()


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
        current_day_desc = "<first_or_undefined>"
        while scrolls_back < OWA_MAX_SCROLL_BACK:
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
                if date_label_span:
                    current_day_desc = date_label_span.text
                    if date_label and date_label_span.text == date_label:
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
            LOG.info(f"From {current_day_desc} page clicking on '{scroll_back.get_attribute('ariaLabel')}' button"
                     f"(rect={scroll_back.rect}) to shift on previous day.")
            scroll_back.click()
            scrolls_back += 1
    LOG.info(f"Finishing on {current_day_desc} page.")


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
        # Check if meeting started after 23:00. It should be within a page anyway.
        if start_y > hour_points[-1][1]:
            index_started_in = 23
            index_ended_in = 23  # We can't see meetings from the next day so assume it ends at midnight.
        else:
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


def scrape_events_from_page(driver: WebDriver, events_date: datetime.datetime) -> List[Event]:
    """
    Scapes events from the page currently opened in the given WebDriver.
    :param driver: 'WebDriver' to scrape data from page opened in.
    :param events_date: Date to set for the events scraped.
    :return: List of scraped events.
    """
    # Note that page may contian few containers, espectially if t)here are no events this day.
    events_containers = _wait_on_page(
        driver,
        "//div[contains(@class,'scrollContainer') and not(contains(@style,'none'))]"
            "/div[@role='presentation' and not(contains(@style,'none'))]"
    )
    events_container = next(x for x in events_containers if x.size['height'] > 100)
    # Wait until all events are rendered on the container. Immediate check returns only first event(s).
    driver.implicitly_wait(1)
    events_container.screenshot(SCREENSHOT_NAME)
    LOG.info(f"Page of required day is opened, scrapping events. See screenshot {SCREENSHOT_NAME}.")
    # Note that these div-s also contains elements with calendar(s) name.
    TYPE_DIV_XPATH_SELECTOR = "div[contains(@class,'calendarBusy') or contains(@class,'calendarTentative')"\
                              " or contains(@class,'calendarFree')]"
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
        events.append(Event(OWA_BUCKET_ID, start_time, duration, {
            'type': event_type,
            'name': event_elements[0].text,
            'location': event_elements[1].text,
            'sender': event_elements[2].text,
        }))
    return events


def _calculate_events_date_and_scrolls_back(events_date: datetime.datetime, back_days: int, date_label: str)\
        -> Tuple[datetime.datetime, int]:
    # Check parameters for validity.
    if events_date:
        assert (isinstance(events_date, datetime.datetime) or isinstance(events_date, datetime.date))\
                and events_date.tzinfo,\
            f"'events_date' value ({events_date}) should be a date/datetime with timezone info."
    if back_days:
        assert isinstance(back_days, int),\
            f"'back_days' value ({back_days}) should be an integer."
        assert 0 <= back_days < OWA_MAX_SCROLL_BACK,\
            f"'back_days' value ({back_days}) should be positive and not more than limit {OWA_MAX_SCROLL_BACK}."
    if date_label:
        assert isinstance(date_label, str),\
            f"'date_label' value ({date_label}) should be a string with date label."
    # Calculate events_date and back_days for events.
    if events_date is None:
        events_date = datetime.datetime.today().astimezone()
    if back_days and back_days > 0:
        events_date = (datetime.datetime.today() - datetime.timedelta(days=back_days)).astimezone()
    else:
        back_days = (datetime.datetime.today() - events_date.replace(tzinfo=None)).days
    events_date = events_date.date()  # Convert to date because for events need to remove "time" part.
    return events_date, back_days,


def get_events_from_owa(profile_abs_path: str, owa_url: str, headless: bool = False,
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
    :param events_date: Optional. Date to set for events and to scroll on. Should contain time zone info.
    By-default is "today - back_days" (or just "today" if 'back_days' is not provided).
    :param back_days: Optional. Number of days to scroll back from today to reach required date.
    :param date_label: Optional. Date label on page to search with "tap back until open" logic.
    Should match 'date' parameter. If 'back_days' is provided it should match it's value.
    Value can't be calculated because depends on language, region, OWA365 settings, etc.
    :return: List of `Event`-s parsed for specified date.
    """
    # Check required input parameters.
    assert profile_abs_path, "Firefox profile folder path is not specified."
    assert owa_url, "OWA365/Web Outlook URL is not specified."
    events_date, back_days = _calculate_events_date_and_scrolls_back(events_date, back_days, date_label)
    LOG.info(f"Starting Firefox with profile '{profile_abs_path}'."
             " Note that you need to have OWA opened and authenticated under this profile, otherwise it would fail.")
    # In "with UI" mode some parsing on my machine takes 65s, in headless - the same 65s.
    page_url = owa_url + CALENDAR_TODAY_URL_SUFFIX
    with start_firefox_under_existing_profile(profile_abs_path, page_url, headless) as driver:
        LOG.info(f"Firefox started, waiting loading of '{page_url}'. Don't scroll/click on it's window!")
        # Check that first need scroll to day and perform scrolls.
        if back_days or date_label:
            _scroll_to_day(back_days, date_label, driver)
        # Parse events. TODO parse few days.
        return scrape_events_from_page(driver, events_date)


def main():
    parser = argparse.ArgumentParser(
        description="Opens Firefox (headless if need) under specified profile (see '--profile-path' parameter)"
                    " with given Office 365 Email/Calendar page (aka OWA365), scrolls to specified date in Calendar,"
                    " parses all found events in it and loads them into ActivityWatch. Note that you need to have"
                    " Firefox opened in another window and be logged in OWA in order to pass authentication."
    )
    parser.add_argument('events_date', nargs='?', type=valid_date, default=datetime.datetime.now().astimezone(),
                        help="Date to set for OWA365/Web Outlook Calendar events in format 'YYYY-mm-dd'."
                             " By default is today.")
    parser.add_argument('-b', '--back-days', type=int,
                        help="How many days need to scroll back from today to reach day to scrape Calendar events."
                             f" Overwrites EVENTS_DATE if specified. Max to {OWA_MAX_SCROLL_BACK}."
                             " If is not specified then calculated as 'today - EVENTS_DATE'.")
    parser.add_argument('-l', '--date-label', type=str,
                        help="Date label on OWA page for 'scroll days back until open page with this label' logic."
                             " Value depends on the language, region, OWA365 settings, etc. It should match 'date'"
                             " parameter and is just extra check for the day we want to gather events from.")
    parser.add_argument('--headless', action='store_true',
                        help="Flag to open Firefox window in the headless mode. Doesn't make parsing faster"
                             "but eliminates risk to click something on the page and break parsing.")
    parser.add_argument('-p', '--profile-path', type=str, default=FIREFOX_PROFILE_PATH,
                        help="Absolute path to Firefox profile folder. On Linux it looks like "
                             "'/home/{username}/.mozilla/firefox/{some_id}.default-release/'.")
    parser.add_argument('-u', '--owa-url', type=str, default=OWA_URL,
                        help="URL to Web (MS Office Web Apps) Outlook. Page where email box opens."
                             "May look like 'https://mail.company.com/owa'.")
    parser.add_argument('-r', '--replace', dest='is_replace_bucket', action='store_true',
                        help=f"Flag to delete ActivityWatch '{OWA_BUCKET_ID}' bucket first."
                            " Removes all previous events in it, for all time.")
    parser.add_argument('--dry-run', dest='is_dry_run', action='store_true',
                        help="Flag to just parse and log events, don't upload them into ActivityWatch.")
    args = parser.parse_args()
    events = get_events_from_owa(args.profile_path, args.owa_url, args.headless, events_date=args.events_date, 
                                 back_days=args.back_days, date_label=args.date_label)
    LOG.info(f"Ready to upload {len(events)} events:" + "\n  " + "\n  ".join(str(x) for x in events))
    # Load events into ActivityWatcher
    if not args.is_dry_run:
        LOG.info(upload_events(events, OWA_SCRAPER_NAME, "owa365.calendar.event", OWA_BUCKET_ID,
                               args.is_replace_bucket))


if __name__ == '__main__':
    LOG = setup_logging()
    main()


# Tests are placed right here because of Python imports inflexibility.
class TestGetOutlookEvents(unittest.TestCase):
    today = datetime.datetime.now().astimezone()
    day_ago = (today - datetime.timedelta(days=1))

    @parameterized.parameterized.expand([
        (
            "all None-s",
            None, None, None,
            None, today.date(), 0
        ),
        (
            "date_label only",
            None, None, "some",
            None, today.date(), 0
        ),
        (
            "wrong events_date",
            "some", None, None,
            "'events_date' value \(some\) should be a date/datetime with timezone info.", None, None
        ),
        (
            "events_date is without timezone info",
            datetime.datetime.now(), None, None,
            "'events_date' value (.*) should be a date/datetime with timezone info.", None, None
        ),
        (
            "only events_date - use it",
            day_ago, None, None,
            None, day_ago.date(), 1
        ),
        (
            "wrong back_days",
            day_ago, "100500", None,
            "'back_days' value \(100500\) should be an integer.", None, None
        ),
        (
            "too big back_days",
            day_ago, 100500, None,
            f"'back_days' value \(100500\) should be positive and not more than limit {OWA_MAX_SCROLL_BACK}.",
            None, None
        ),
        (
            "no date_label",
            day_ago, None, None,
            None, day_ago.date(), 1
        ),
        (
            "both date and back-days provided",
            day_ago, 2, None,
            None, (today - datetime.timedelta(days=2)).date(), 2
        ),
    ])
    def test_calculate_events_date_and_scrolls_back(self, test_name, events_date, back_days, date_label,
            expected_message, expected_date, expected_back_days):
        if expected_message:
            with self.assertRaisesRegex(AssertionError, expected_message, msg=f"'{test_name}' wrong error."):
                _calculate_events_date_and_scrolls_back(events_date, back_days, date_label)
        else:
            actual_date, actual_back_days = _calculate_events_date_and_scrolls_back(events_date, back_days, date_label)
            self.assertEqual(actual_date, expected_date, f"'{test_name}' wrong actual date.")
            self.assertEqual(actual_back_days, expected_back_days, f"'{test_name}' wrong actual back_days.")
