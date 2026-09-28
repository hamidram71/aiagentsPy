"""The web app, driven like a user would: sign in, ask, approve, read.

Streamlit's AppTest runs app.py in this process, without a browser, and lets
the test click buttons and read what is on the page. The model is the fake
provider, so the whole run is scripted and needs no API key.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import runner  # noqa: E402

QUESTION = "Which region drives revenue?"


@pytest.fixture
def app(monkeypatch, override_settings):
    import streamlit as st

    from src.llm import get_llm

    override_settings(provider="fake", model="", require_approval=True, checkpoint_db=":memory:")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    st.cache_resource.clear()  # a fresh job list and checkpointer per test
    get_llm.cache_clear()
    yield lambda: AppTest.from_file("../app.py", default_timeout=30)
    st.cache_resource.clear()


def _sign_in(at: AppTest, name: str = "Hamid", password: str | None = None) -> AppTest:
    at.run()
    at.text_input[0].input(name)
    if password is not None:
        at.text_input[1].input(password)
    at.button[0].click()
    return at.run()


def _button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


def _wait_until(at: AppTest, done, seconds: float = 60) -> AppTest:
    deadline = time.time() + seconds
    while True:
        at.run()
        if done(at):
            return at
        assert time.time() < deadline, [m.value for m in at.markdown][:5]
        time.sleep(0.3)


def _text(at: AppTest) -> str:
    return "\n".join(str(m.value) for m in [*at.markdown, *at.caption, *at.title, *at.subheader])


def test_sign_in_ask_approve_and_read_the_report(app) -> None:
    at = _sign_in(app())
    assert not at.exception
    assert "New analysis" in [t.value for t in at.title]

    at.radio[0].set_value("Sample data (sales.csv)")
    at.text_area[0].input(QUESTION)
    _button(at.run(), "Start analysis").click()
    at.run()

    # The plan comes back for approval.
    at = _wait_until(at, lambda a: "Plan: approve or change it" in _text(a))
    assert "Total revenue per region" in _text(at)
    _button(at, "Approve and run").click()
    at.run()

    at = _wait_until(at, lambda a: any(b.label == "Everything (.zip)" for b in a.get("download_button")))
    assert not at.exception
    text = _text(at)
    assert "North leads revenue" in text  # the scripted report's heading
    # The chart named in the report is on the page as a picture, not a path.
    images = at.get("image")
    assert images, "the chart should be displayed"
    assert images[0].captions == ["revenue_by_region.png"]

    runs = runner.list_runs(user="Hamid")
    assert len(runs) == 1 and runs[0]["status"] == "done"
    assert runs[0]["charts"] == ["revenue_by_region.png"]


def test_a_password_keeps_people_out(app, monkeypatch) -> None:
    monkeypatch.setenv("APP_PASSWORD", "open sesame")
    at = _sign_in(app(), password="wrong")
    assert any("Wrong password" in e.value for e in at.error)
    assert "New analysis" not in [t.value for t in at.title]

    at = _sign_in(app(), password="open sesame")
    assert "New analysis" in [t.value for t in at.title]


def test_a_run_id_from_the_address_bar_cannot_leave_the_runs_folder(app) -> None:
    at = app()
    at.query_params["run"] = "../../secrets"
    at = _sign_in(at)
    assert not at.exception
    assert "New analysis" in [t.value for t in at.title]


def test_the_preview_reads_a_german_excel_file(app) -> None:
    from app import read_preview

    german = "Datum;Region;Umsatz\n13.02.2025;Süd;1234,56\n".encode("cp1252")
    frame = read_preview(german)
    assert list(frame.columns) == ["Datum", "Region", "Umsatz"]
    assert frame["Umsatz"].iloc[0] == 1234.56
    assert "only one column" in read_preview(b"just one\n1\n2\n")
    assert "no rows" in read_preview(b"a,b\n")
    assert not isinstance(read_preview(b"a,b\n1,2\n"), str)


def test_chart_references_in_the_report_become_pictures(tmp_path) -> None:
    """`![title](file.png)` is replaced by the picture -- from the run folder
    only, whatever directory the text names."""
    import shutil

    chart = tmp_path / "revenue_by_region.png"
    shutil.copyfile(_any_png(tmp_path), chart)

    def page(folder: str) -> None:
        from pathlib import Path

        from app import render_report

        render_report(
            "# Title\n\n![Revenue by region](C:/elsewhere/revenue_by_region.png)\n\n"
            "Text after.\n\n![Missing](nope.png)\n",
            Path(folder),
        )

    at = AppTest.from_function(page, args=(str(tmp_path),)).run()
    assert not at.exception
    assert [i.captions for i in at.get("image")] == [["Revenue by region"]]
    assert any("chart not found: nope.png" in c.value for c in at.caption)
    assert "Text after." in "\n".join(m.value for m in at.markdown)


def _any_png(folder) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = folder / "source.png"
    plt.figure(figsize=(1, 1))
    plt.savefig(path)
    plt.close("all")
    return str(path)
