#!/usr/bin/env python3
import datetime
from typing import Dict, List, Callable, Any, Tuple
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
import aw_client

from activity_merger.config.config import LOG
from activity_merger.helpers.helpers import setup_logging
from activity_merger.domain.input_entities import Event


NAME = 'outlook_aw_events_scraper'
SCREENSHOT_NAME = f"{NAME}-events-screenshot.png"
PAGE_URL = 'https://mail.akvelon.com/owa/#path=/calendar/view/Day'


def start_firefox_under_existing_profile(page: str, headless: bool = True) -> WebDriver:
    PROFILE = '/home/i4ellendger/.mozilla/firefox/21357bye.default-release/'
    options = webdriver.FirefoxOptions()
    firefox_args = [ # moz:firefoxOptions
        '--safe-mode',
        '--new-tab', page,
        '--start-maximized',
    ]
    options._arguments.extend(firefox_args)
    # Note that '--profile' inside options._arguments causes alert "window with such profile is already opened".
    options.profile = PROFILE
    options.headless = headless
    LOG.info(f"Wait few seconds/minutes - new Firefox window is starting with profile '{PROFILE}'")
    return webdriver.Firefox(options=options)  # Note that all other params are deprecated.


def call_web_element_with_fail_handling(description: str, container: WebElement, func: Callable,
        check_result_not_empty: bool = True) -> Any:
    assert container, f"call_web_element_with_fail_handling: Can't get {description} from empty {container}"
    try:
        result = func(container)
        if check_result_not_empty:
            assert result, f"Can't find {description} on {container}"
        return result
    except Exception as e:
        LOG.error(container.get_attribute('innerHTML'))
        raise e


def _find_hour(hour_points: List[Tuple[int, Dict]], element: WebElement) -> Tuple[int, Dict]:
    y = element.rect['y']
    result = None
    for hour_point in hour_points:
        if hour_point[1] > y:
            break
        result = hour_point
    if result is None:
        assert False, f"Can't find corresponding hour for element '{element.text}' with y coordinate {y}"\
                      f" among hours: {hour_points}"
    return result


def get_events_from_owa() -> List[Event]:
    driver = start_firefox_under_existing_profile(PAGE_URL, False)  # TODO True
    LOG.info(f"Firefox started, waiting loading of {PAGE_URL}")
    EVENTS_XPATH_FUNC = "(contains(@class,'calendarBusy') or contains(@class,'calendarTentative'))"

    # Note that page may contians few containers, espectially if there are no events this day.
    events_containers = WebDriverWait(driver, 1*60).until(
        # lambda driver: driver.execute_script('return document.readyState') == 'complete'
        EC.visibility_of_all_elements_located((
            By.XPATH,
            "//div[contains(@class,'scrollContainer') and not(contains(@style,'none'))]"
            "/div[@role='presentation' and not(contains(@style,'none'))]"
        ))
    )
    # events_containers = driver.find_elements(
    #     By.XPATH,
    #     "//div[contains(@class,'scrollContainer')]/div[@role='presentation']"
    # )
    events_container = next(x for x in events_containers if x.size['height'] > 100)
    events_container.screenshot(SCREENSHOT_NAME)
    LOG.info(f"Page loaded, scrapping events. See screenshot {SCREENSHOT_NAME}")
    # Note that these div-s also contains elements with calendar name(s).
    probable_event_divs = call_web_element_with_fail_handling(
        "probable containers of event rectangles",
        events_container,
        lambda x: x.find_elements(By.XPATH, "//div[" + EVENTS_XPATH_FUNC + "]/..")
    )
    events: List[Event] = []
    hour_points: List[Tuple[int, Dict]] = None
    for probable_event_div in probable_event_divs:
        # Search "hour" elements at left column first time.
        if hour_points is None:
            hour_spans = call_web_element_with_fail_handling(
                "containers of event rectangles",
                probable_event_div,
                lambda x: x.find_elements(
                    By.XPATH,
                    "//span[contains(@class,'ms-font-m')"
                    " and contains(@class,'semilight')"
                    " and contains(@class,'ms-font-color-neutralPrimary')]"
                )
            )
            hour_points = [(int(x.text), x.rect['y']) for x in hour_spans if x.text.isnumeric()]
            if len(hour_points) < 24:
                LOG.error(probable_event_div.get_attribute('innerHTML'))
                assert False, "Can't find all 24 'hour' points on the screen."
        # Search for the text of event.
        event_div = call_web_element_with_fail_handling(
            "div with event data",
            probable_event_div,
            lambda x: x.find_element(By.XPATH, "//div[contains(@class,'contentDiv') and count(span)=3]"),
            False
        )
        if event_div:
            hour = _find_hour(hour_points, event_div)
            LOG.info(f"Found event '{event_div.text}' in rect {event_div.rect} started at or later "
                     f"{hour[0]} at {hour[1]}")
            # TODO Search for start and end.


    LOG.info(events_container.page_source)
    # LOG.info("Assembled %d activities:\n  %s" % (len(activities), "\n  ".join(str(x) for x in activities)))


def main():
    events = get_events_from_owa()
    LOG.info(f"Parsed {len(events)} from {PAGE_URL}" + "\n  " + "\n  ".join(str(x) for x in events))
    # load events into ActivityWatcher


if __name__ == '__main__':
    LOG = setup_logging()
    main()
