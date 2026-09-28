"""Settings and command-line behaviour, including the Windows traps.

Every test here reproduces a failure found in the full review, most of them
in a real subprocess: the bugs lived in how the program meets its
environment -- a .env saved by PowerShell, a pipe, no keyboard, a parent
folder -- and a function call in the test process would not meet it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import src.config as config

ROOT = Path(__file__).resolve().parent.parent


def cli(*args: str, env: dict | None = None, stdin=subprocess.DEVNULL, cwd: Path = ROOT):
    """Run main.py for real and return (exit code, stdout, stderr)."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "REQUIRE_", "MAX_"))}
    run = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=cwd, env={**base, **(env or {})}, stdin=stdin,
        capture_output=True, text=True, timeout=120,
    )
    return run.returncode, run.stdout, run.stderr


# --- .env as Windows saves it ---------------------------------------------------


def test_a_utf16_env_file_is_read(tmp_path) -> None:
    """PowerShell 5.1's `echo LLM_PROVIDER=fake > .env` writes UTF-16; the
    loader crashed on import with UnicodeDecodeError."""
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=fake\n", encoding="utf-16")
    assert "LLM_PROVIDER=fake" in config._read_env_text(env)


def test_a_bom_does_not_hide_the_first_key(tmp_path) -> None:
    """Notepad's UTF-8-with-BOM made the first key '\\ufeffLLM_PROVIDER'."""
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=fake\n", encoding="utf-8-sig")
    assert config._read_env_text(env).startswith("LLM_PROVIDER=fake")


def test_a_non_utf8_env_file_gets_a_clear_message(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_bytes("LLM_MODEL=gpté\n".encode("latin-1"))
    with pytest.raises(config.ConfigError, match="UTF-8"):
        config._read_env_text(env)


def test_a_comment_after_a_blank_value_leaves_it_blank() -> None:
    """`LLM_TEMPERATURE=   # blank = no temperature` used to set the value to
    the comment, and the import crashed on float('# blank ...')."""
    text = "LLM_TEMPERATURE=          # blank = send no temperature\nMAX_STEPS=30   # thirty\n"
    cleaned = config._clean_env_text(text)
    assert "LLM_TEMPERATURE=\n" in cleaned
    assert "MAX_STEPS=30   # thirty" in cleaned, "a comment after a real value is dotenv's job"


def test_only_the_projects_env_is_loaded_never_a_parent_folders(tmp_path) -> None:
    """With no .env in the project, a bare load_dotenv() walked up and loaded
    a PARENT folder's .env -- while check_llm.py reported ".env MISSING"."""
    project = tmp_path / "parent" / "proj"
    shutil.copytree(ROOT / "src", project / "src")
    (tmp_path / "parent" / ".env").write_text("LLM_PROVIDER=anthropic\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
    out = subprocess.run(
        [sys.executable, "-c", "import src.config as c; print(c.settings.provider)"],
        cwd=project, env=env, capture_output=True, text=True, timeout=60,
    )
    assert out.stdout.strip() == "openai", out.stdout + out.stderr


# --- values ---------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [("0,2", 0.2), ("0.7", 0.7), ("", None)])
def test_temperature_accepts_a_decimal_comma(raw, expected) -> None:
    assert config._as_temperature(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [("ja", True), ("Nein", False), ("yes", True), ("0", False)])
def test_yes_no_values_including_german(raw, expected) -> None:
    assert config._as_bool(raw, default=not expected, name="X") is expected


def test_an_unknown_yes_no_value_is_an_error_not_false() -> None:
    """REQUIRE_APPROVAL=ja used to switch approval OFF, silently."""
    with pytest.raises(config.ConfigError, match="REQUIRE_APPROVAL"):
        config._as_bool("vielleicht", True, "REQUIRE_APPROVAL")


def test_a_bad_number_names_the_setting(monkeypatch) -> None:
    monkeypatch.setenv("MAX_STEPS", "30#x")
    with pytest.raises(config.ConfigError, match="MAX_STEPS"):
        config.Settings()


def test_the_provider_name_is_not_case_sensitive(monkeypatch) -> None:
    """'Anthropic' passed the key check untested and showed an OpenAI model."""
    monkeypatch.setenv("LLM_PROVIDER", "Anthropic")
    settings = config.Settings()
    assert settings.provider == "anthropic"
    assert settings.resolved_model() == config.Settings.DEFAULT_MODELS["anthropic"]


def test_a_settings_error_is_one_readable_line_not_a_traceback() -> None:
    code, _out, err = cli("data/sales.csv", "q", "--provider", "fake", env={"MAX_STEPS": "abc"})
    assert code == 2
    assert "MAX_STEPS='abc' is not a whole number" in err
    assert "Traceback" not in err


# --- the command line -----------------------------------------------------------


def test_provider_without_model_uses_that_providers_default() -> None:
    """--provider anthropic kept LLM_MODEL=gpt-5.6-terra from .env and sent an
    OpenAI model name to Anthropic."""
    code, out, err = cli(
        "data/sales.csv", "q", "--provider", "anthropic", "--no-approval",
        env={"LLM_MODEL": "gpt-5.6-terra", "ANTHROPIC_API_KEY": "sk-ant-dummy",
             "CHECKPOINT_DB": ":memory:", "LLM_TIMEOUT": "1"},
    )
    shown = next(line for line in out.splitlines() if line.startswith("provider:"))
    assert "gpt-5.6-terra" not in shown
    assert config.Settings.DEFAULT_MODELS["anthropic"] in shown


def test_a_new_question_on_an_existing_thread_is_refused(tmp_path) -> None:
    """It used to come back with the previous run's report as the answer."""
    env = {"CHECKPOINT_DB": str(tmp_path / "ck.sqlite")}
    first = cli("data/sales.csv", "Which region?", "--provider", "fake", "--no-approval",
                "--thread-id", "t1", env=env)
    assert first[0] == 0, first[1][-300:] + first[2][-300:]

    code, _out, err = cli("data/sales.csv", "A different question", "--provider", "fake",
                          "--no-approval", "--thread-id", "t1", env=env)
    assert code == 2
    assert "already holds a run" in err and "--resume --thread-id t1" in err


def test_approval_without_a_keyboard_stops_cleanly_and_can_be_resumed(tmp_path) -> None:
    """No stdin: an EOFError traceback. A pipe: an invisible, endless wait."""
    env = {"CHECKPOINT_DB": str(tmp_path / "ck.sqlite")}
    code, out, err = cli("data/sales.csv", "q", "--provider", "fake", "--thread-id", "ap", env=env)
    assert code == 4, out[-300:] + err[-300:]
    assert "PLAN APPROVAL NEEDED" in out
    assert "--resume --thread-id ap" in err
    assert "Traceback" not in err


def test_the_csv_path_is_stored_absolute_so_resume_works_from_anywhere(tmp_path) -> None:
    env = {"CHECKPOINT_DB": str(tmp_path / "ck.sqlite")}
    cli("data/sales.csv", "q", "--provider", "fake", "--thread-id", "abs", env=env)
    probe = (
        "import sys; sys.path.insert(0, r'%s'); import src.graph as g\n"
        "cp, conn = g.make_checkpointer()\n"
        "print(g.build_graph(checkpointer=cp).get_state({'configurable': {'thread_id': 'abs'}}).values['csv_path'])"
    ) % ROOT
    out = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True,
                         text=True, env={**os.environ, **env}, timeout=60)
    stored = out.stdout.strip().splitlines()[-1]
    assert Path(stored).is_absolute(), stored
    assert Path(stored).exists()


def test_a_run_without_a_report_exits_non_zero(tmp_path) -> None:
    """It printed "report: not written" and exited 0."""
    driver = tmp_path / "d.py"
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import tests.fake_llm as f\n"
        "f.default_script = lambda: [TimeoutError('t')] * 6\n"
        "sys.argv = ['main.py', 'data/sales.csv', 'q', '--provider', 'fake', '--no-approval']\n"
        "import main\n"
        "raise SystemExit(main.main())\n",
        encoding="utf-8",
    )
    run = subprocess.run([sys.executable, str(driver)], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "CHECKPOINT_DB": ":memory:"}, timeout=120)
    assert "not written" in run.stdout
    assert run.returncode == 1


# --- explain_fatal ---------------------------------------------------------------


def _fatal(body, param=None):
    pytest.importorskip("langchain_openai")
    import httpx
    from langchain_openai.chat_models.base import OpenAIInvalidRequestError

    from src.errors import FatalModelError

    if isinstance(body, dict) and param:
        body = {**body, "param": param}
    cause = OpenAIInvalidRequestError(
        "Error code: 400", response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
        body=body,
    )
    return FatalModelError("analyst", cause)


@pytest.mark.parametrize("body", [{"message": ["a", "b"]}, {"error": "plain text"}, "just a string", None])
def test_explain_fatal_survives_any_error_shape(body) -> None:
    import main

    text = main.explain_fatal(_fatal(body), "t1")
    assert "stopped at [analyst]" in text


def test_a_malformed_transcript_is_called_a_bug_not_a_setting() -> None:
    import main

    text = main.explain_fatal(_fatal({"message": "role 'tool' must follow ..."}, "messages.[3].role"), "t1")
    assert "bug in this program" in text
    assert "Try another LLM_MODEL" not in text


def test_no_resume_promise_without_a_checkpoint(override_settings) -> None:
    import main

    override_settings(checkpoint_db=":memory:")
    text = main.explain_fatal(_fatal({"message": "x"}), "t1")
    assert "--resume" not in text
    assert "no checkpoint" in text


# --- check_llm tests the path the graph uses -------------------------------------


@pytest.mark.parametrize(("kind", "label"), [
    ("plain", "plain chat"),
    ("structured", "structured output"),
    ("tools", "tool calling"),
])
def test_check_llm_fails_for_a_model_the_graph_could_not_stream(kind, label, monkeypatch, capsys) -> None:
    """The graph streams every model call. check_llm used plain .invoke(), so
    a provider that refuses streaming got three OKs here and a failure at
    the first graph call.

    One case per check, each refusing streaming for ONLY that kind of
    request: with a server that refused all streaming, reverting one check
    to .invoke() was hidden by the other two failing anyway.
    """
    pytest.importorskip("langchain_openai")
    import json

    import httpx

    import check_llm
    import src.llm as llm_module
    from tests.fake_openai import FakeOpenAI

    def kind_of(body: dict) -> str:
        if body.get("tools"):
            return "tools"
        if body.get("response_format"):
            return "structured"
        return "plain"

    class RefusesStreaming(FakeOpenAI):
        def handler(self, request):
            body = json.loads(request.content or b"{}")
            if body.get("stream") and kind_of(body) == kind:
                return httpx.Response(400, json={"error": {
                    "message": "Your organization must be verified to stream this model.",
                    "type": "invalid_request_error", "param": "stream", "code": None}})
            return super().handler(request)

    server = RefusesStreaming(structured_reply={"next": "reporter", "instruction": "x",
                                                "reason": "y", "step_index": 1})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    client = llm_module._openai_class()(model="m", temperature=0.0,
                                        http_client=server.client(), max_retries=0)
    monkeypatch.setattr(llm_module, "get_llm", lambda *a, **k: client)
    monkeypatch.setattr("sys.argv", ["check_llm.py", "--provider", "openai"])

    assert check_llm.main() == 1
    failed = [line for line in capsys.readouterr().out.splitlines() if "FAIL" in line]
    assert any(label in line for line in failed), failed


def test_both_model_clients_get_a_request_timeout(override_settings, monkeypatch) -> None:
    """Without one, the SDK default applied: ten minutes per attempt."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    override_settings(llm_timeout=42)
    from src.llm import get_llm

    get_llm.cache_clear()
    try:
        if pytest.importorskip("langchain_openai"):
            assert get_llm("openai").request_timeout == 42
        if pytest.importorskip("langchain_anthropic"):
            assert get_llm("anthropic").default_request_timeout == 42
    finally:
        get_llm.cache_clear()


# --- the progress line fits the terminal -----------------------------------------


def test_the_progress_line_never_wraps_in_a_narrow_terminal(monkeypatch, capsys) -> None:
    """`\\r` returns to the start of the current screen row only. A progress
    line longer than the terminal wrapped, and the erase left
    "[reporter] working......" behind. Your VS Code panel is narrow."""
    import main

    class FakeGraph:
        def stream(self, payload, config, stream_mode):
            meta = {"langgraph_node": "a_rather_long_node_name"}
            for _ in range(400):  # a long streamed answer: 50 dots at 1 per 8 chunks
                yield "messages", (None, meta)
            yield "updates", {"a_rather_long_node_name": {}}

        def get_state(self, config):
            class Snapshot:
                values = {}
            return Snapshot()

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("shutil.get_terminal_size", lambda fallback=None: os.terminal_size((30, 20)))
    main.run(FakeGraph(), {}, {})

    out = capsys.readouterr().out
    for segment in out.split("\r"):
        assert len(segment) <= 29, f"{len(segment)} chars on a 30-column terminal: {segment!r}"


class _GraphThatAsksForApproval:
    """Streams one approval interrupt, like the real graph after the planner."""

    def stream(self, payload, config, stream_mode):
        class Interrupt:
            value = {"question": "q", "plan": ["Region", "Write the report"]}

        yield "updates", {"__interrupt__": (Interrupt(),)}

    def get_state(self, config):
        class Snapshot:
            values = {}
        return Snapshot()


def test_approval_is_not_asked_when_the_output_is_piped(monkeypatch) -> None:
    """Layer 1 -- stdout is not a terminal (`| Select-String`): the question
    would go into the pipe, invisible, and input() would wait forever. So it
    must not be asked at all, even though a keyboard is attached."""
    import main

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("input() must not be called"))
    with pytest.raises(main.ApprovalNeeded):
        main.run(_GraphThatAsksForApproval(), {}, {})


def test_approval_survives_input_ending_early(monkeypatch) -> None:
    """Layer 2 -- a terminal on both ends, but the input stream closes
    (Ctrl+Z on Windows, a closed pipe): EOFError, which used to crash with a
    traceback. Each layer is tested alone, because the other hides it."""
    import main

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    def eof(*_a):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(main.ApprovalNeeded):
        main.run(_GraphThatAsksForApproval(), {}, {})
