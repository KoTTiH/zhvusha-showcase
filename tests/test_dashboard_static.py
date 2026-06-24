"""Static checks for dashboard refresh wiring."""

from __future__ import annotations

from pathlib import Path


def _dashboard_html() -> str:
    return Path("src/mcp_server/static/dashboard.html").read_text(encoding="utf-8")


def test_status_panels_refresh_periodically() -> None:
    html = _dashboard_html()

    assert "const STATUS_REFRESH_MS = 30000;" in html
    assert "refreshStatusPanels();" in html
    assert "setInterval(refreshStatusPanels, STATUS_REFRESH_MS);" in html
    assert "setInterval(updateStatusBar" not in html


def test_status_fetch_bypasses_browser_cache() -> None:
    html = _dashboard_html()

    assert "fetch(path, { cache: 'no-store' })" in html


def test_overview_refresh_does_not_replace_entry_properties() -> None:
    html = _dashboard_html()

    assert "rightPanelMode: 'overview'" in html
    assert "state.rightPanelMode = 'entry';" in html
    assert "if (state.rightPanelMode === 'overview')" in html
