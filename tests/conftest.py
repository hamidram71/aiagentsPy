"""Shared test fixtures.

`Settings` is a frozen dataclass -- immutable config is the right default,
because it means no node can quietly reconfigure the run at half past three.
The price is that tests cannot just assign to it. `override_settings` pays
that price properly: it builds a *new* Settings with `dataclasses.replace`
and patches it into every module that did `from src.config import settings`.

That last part is the subtle bit. `from x import y` binds a name in the
importing module, so patching `src.config.settings` alone would leave every
node still pointing at the old object.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import src.config

# Modules imported up front so every consumer of `settings` is loaded before
# the fixtures look for them.
PRELOAD = ("src.graph", "src.budget", "src.llm", "src.tools.analysis", "src.tools.charts")


def settings_consumers() -> list:
    """Every loaded src module that holds its own reference to `settings`.

    This used to be a hand-written list of five modules. src.nodes.analyst,
    src.nodes.charter and (later) src.budget were not on it, so tests that
    set max_steps or max_tool_rounds changed the value everywhere EXCEPT in
    the code that enforced it -- and still passed, because the supervisor
    happened to enforce the same budget a step later. Found when a new test
    for the per-step turn cap failed for no visible reason. A list of "all
    the places" is always one place short; ask the interpreter instead.
    """
    import importlib
    import sys

    for name in PRELOAD:
        importlib.import_module(name)
    return [
        module
        for name, module in list(sys.modules.items())
        if (name == "src" or name.startswith("src."))
        and isinstance(getattr(module, "settings", None), src.config.Settings)
    ]


# Every environment variable Settings reads. The suite must not depend on
# any of them.
ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT",
    "REQUIRE_APPROVAL",
    "REQUIRE_CHART",
    "MAX_FAILURES",
    "MAX_STEPS",
    "MAX_TOOL_ROUNDS",
    "CHECKPOINT_DB",
    "CODE_TIMEOUT",
    "MAX_ROWS_TO_MODEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Run every test against default settings, never the developer's .env.

    This fixture exists because of a real bug. The suite passed on a machine
    with no `.env`, then failed on a user's machine with three errors like

        assert 'llama3.1' == 'claude-sonnet-5'

    because their `.env` set `LLM_MODEL=llama3.1` and the tests read it. A
    test suite whose result depends on local configuration is not a test
    suite: it is green for the author, red for everyone else, and it cannot
    tell a real regression from a config difference.

    So: clear the variables, rebuild `Settings` from the clean environment,
    and patch that object into every module holding a reference. Tests that
    need a specific value ask for it explicitly via `override_settings`.
    """

    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    pristine = src.config.Settings()
    for module in settings_consumers():
        monkeypatch.setattr(module, "settings", pristine)
    yield pristine


@pytest.fixture
def override_settings(monkeypatch):
    """override_settings(max_steps=1, require_approval=False)"""

    def _override(**changes: Any) -> src.config.Settings:

        new = dataclasses.replace(src.config.settings, **changes)
        for module in settings_consumers():
            monkeypatch.setattr(module, "settings", new)
        return new

    return _override


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch, request):
    """Keep the suite from writing reports and charts into the real outputs/.

    A test suite that litters the project directory is a test suite people
    stop running. Opted out of by tests that patch OUTPUT_DIR themselves.
    """
    if "no_output_isolation" in request.keywords:
        yield
        return

    import importlib

    sandbox = tmp_path / "outputs"
    sandbox.mkdir(exist_ok=True)
    for module_name in ("src.nodes.reporter", "src.tools.charts", "src.dataset"):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "OUTPUT_DIR", sandbox)
    # Subprocesses (main.py run for real) inherit this and write here too.
    monkeypatch.setenv("OUTPUT_DIR", str(sandbox))
    runner = importlib.import_module("src.runner")
    monkeypatch.setattr(runner, "RUNS_DIR", sandbox / "runs")
    monkeypatch.setattr(runner, "USAGE_FILE", sandbox / "usage.csv")
    yield sandbox
