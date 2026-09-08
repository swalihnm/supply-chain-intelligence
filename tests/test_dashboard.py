"""Dashboard smoke tests using Streamlit's AppTest.

A 200 from /_stcore/health only proves the server started - Streamlit does not
execute the script until a websocket connects, so a plain HTTP check would pass
on a dashboard that raises on every render. AppTest runs the script for real.
"""

from __future__ import annotations

import pytest

from sci.config import REPO_ROOT, settings

# AppTest resolves relative paths against the CALLING file, so this must be
# absolute or it looks for tests/app/dashboard.py.
DASHBOARD = REPO_ROOT / "app" / "dashboard.py"

pytestmark = pytest.mark.skipif(
    not (settings.artifacts_dir / "serving_bundle.npz").exists(),
    reason="serving bundle not built (run: scripts/build_serving_bundle.py)",
)

VIEWS = ["Executive", "Forecast explorer", "Policy explorer", "Model health"]


def _run(view: str | None = None, timeout: int = 180):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD), default_timeout=timeout)
    at.run()
    if view is not None:
        at.radio[0].set_value(view).run()
    return at


def test_dashboard_loads_without_exception():
    at = _run()
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("view", VIEWS)
def test_each_view_renders(view):
    at = _run(view)
    assert not at.exception, f"{view} raised: {[str(e) for e in at.exception]}"


def test_executive_view_shows_the_headline_metrics():
    at = _run("Executive")
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert any("cost" in label.lower() for label in labels), labels
    assert any("fill rate" in label.lower() for label in labels), labels


def test_model_health_view_lists_models():
    at = _run("Model health")
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(
        str(s.value) for s in at.subheader
    )
    assert "ranking" in text.lower() or "calibration" in text.lower()
