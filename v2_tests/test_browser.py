"""Opt-in local browser checks: F1ENGINE_E2E_URL=http://127.0.0.1:8766 pytest -m browser."""

import os
from pathlib import Path
import pytest

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(
        not os.environ.get("F1ENGINE_E2E_URL"), reason="Set F1ENGINE_E2E_URL to a running offline demo"
    ),
]


@pytest.fixture
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(os.environ["F1ENGINE_E2E_URL"])
        page.get_by_role("button", name="Generate forecast").wait_for()
        yield page
        assert not errors, errors
        browser.close()


def generate(page):
    page.get_by_role("button", name="Generate forecast").click()
    page.get_by_role("heading", name="Whole field", exact=True).wait_for()


def test_race_distribution_and_evidence(page):
    generate(page)
    assert page.get_by_role("heading", name="Finishing distribution").is_visible()
    page.get_by_text("Forecast evidence & assumptions", exact=True).click()
    page.get_by_role("button", name="Verify reproduction").click()
    page.get_by_role("dialog").wait_for()
    assert '"match": true' in page.get_by_role("dialog").inner_text()
    page.get_by_role("button", name="Close ×").click()
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path="docs/v2/control-room.png", full_page=False)


def test_weekend_coverage(page):
    page.get_by_role("button", name="Weekend").click()
    page.get_by_label("Stage", exact=True).select_option("post_practice")
    page.get_by_role("button", name="Inspect current cutoff").click()
    page.get_by_role("heading", name="Practice pace", exact=True).wait_for()
    assert page.get_by_role("heading", name="Input coverage").is_visible()


def test_championship(page):
    page.get_by_role("button", name="Championship", exact=False).first.click()
    page.get_by_role("button", name="Simulate championship").click()
    page.get_by_role("heading", name="Points outlook").wait_for()
    assert page.get_by_role("heading", name="Drivers’ championship").is_visible()


def test_replay(page):
    page.get_by_role("button", name="Live / replay").click()
    page.get_by_role("button", name="Load illustrative replay").click()
    page.get_by_role("button", name="Inspect at cutoff").wait_for()
    page.get_by_role("button", name="Inspect at cutoff").click()
    page.get_by_role("heading", name="Timing tower").wait_for()
    assert page.get_by_text("Track position unavailable", exact=True).is_visible()
    page.get_by_role("button", name="Forecast remaining race").click()
    page.get_by_role("button", name="Race", exact=False).first.click()
    page.get_by_role("heading", name="Whole field", exact=True).wait_for()


def test_evidence_and_failed_request_retains_forecast(page):
    generate(page)
    page.get_by_label("Information cutoff UTC").fill("2026-04-06T12:00")
    page.get_by_role("button", name="Generate forecast").click()
    page.get_by_role("alert").wait_for()
    assert page.get_by_role("heading", name="Whole field", exact=True).is_visible()
    assert page.get_by_text("Previous snapshot", exact=False).is_visible()
    page.get_by_role("button", name="Evidence", exact=False).first.click()
    assert page.get_by_role("heading", name="Append-only forecast ledger").is_visible()


def test_calendar_selection_and_keyboard_evidence(page):
    page.get_by_role("button", name="Select Illustrative Grand Prix 4", exact=True).click()
    assert page.get_by_label("Event", exact=True).input_value() == "2026-04"
    assert (
        page.get_by_role("button", name="Select Illustrative Grand Prix 4", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    trigger = page.get_by_role("button", name="Calendar evidence")
    trigger.click()
    dialog = page.get_by_role("dialog")
    assert '"event_id": "2026-04"' in dialog.inner_text()
    page.keyboard.press("Tab")
    assert page.get_by_role("button", name="Close ×").evaluate("el => el === document.activeElement")
    page.keyboard.press("Escape")
    assert dialog.count() == 0
    assert trigger.evaluate("el => el === document.activeElement")
    page.get_by_label("Event", exact=True).select_option("2026-03")
    assert (
        page.get_by_role("button", name="Select Illustrative Grand Prix 3", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    page.get_by_label("Information cutoff UTC").fill("2026-04-06T12:00")
    assert page.get_by_text("Scheduled start has passed", exact=True).is_visible()
    assert page.get_by_role("timer").count() == 0


def test_mobile_offline_fonts_and_workspaces(page):
    page.set_viewport_size({"width": 390, "height": 844})
    page.emulate_media(reduced_motion="reduce")
    external = []

    def local_only(route):
        if not route.request.url.startswith(os.environ["F1ENGINE_E2E_URL"]):
            external.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    page.route("**/*", local_only)
    page.reload()
    page.get_by_role("button", name="Generate forecast").wait_for()
    page.evaluate("document.fonts.ready")
    assert page.evaluate("document.fonts.check('800 64px \"Playfair Display Variable\"')")
    generate(page)
    page.locator(".js-plotly-plot").first.wait_for()
    for workspace in ["Race", "Weekend", "Championship", "Live / replay", "Evidence"]:
        page.get_by_role("navigation", name="Workspaces").get_by_role(
            "button", name=workspace, exact=True
        ).click()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), workspace
    page.get_by_role("navigation", name="Workspaces").get_by_role("button", name="Race", exact=True).click()
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path="docs/v2/pit-wall-mobile.png", full_page=True)
    assert not external, external
