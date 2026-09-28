"""CLI preflight checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from main import credential_problem
from src.config import Settings


@pytest.fixture
def cfg():
    return Settings()


def test_no_complaint_when_the_key_is_present(cfg, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    assert credential_problem(dataclasses.replace(cfg, provider="openai")) is None


def test_a_keyless_provider_is_never_blocked(cfg, monkeypatch) -> None:
    """The scripted test model must never need an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert credential_problem(dataclasses.replace(cfg, provider="fake")) is None


@pytest.mark.parametrize(
    ("provider", "key"),
    [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
)
def test_a_missing_key_is_reported_before_the_graph_runs(
    cfg, monkeypatch, provider: str, key: str
) -> None:
    monkeypatch.delenv(key, raising=False)
    message = credential_problem(dataclasses.replace(cfg, provider=provider))

    assert message is not None
    assert key in message
    assert provider in message


def test_the_message_names_both_the_provider_and_the_model(cfg, monkeypatch) -> None:
    """The mistake this exists for: `--model X` without `--provider`.

    The old failure was "OpenAIError: Missing credentials" halfway through a
    run, which points at the key rather than at the real cause. The message
    must name the provider AND the model, and show a command that works.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mismatched = dataclasses.replace(cfg, provider="openai", model="some-other-model")

    message = credential_problem(mismatched)

    assert "some-other-model" in message
    assert "openai" in message
    assert "--provider anthropic" in message


def test_a_fatal_stop_explains_itself_and_says_how_to_continue() -> None:
    """The message a user acts on: the provider's sentence (not a dict repr),
    the parameter it blames, and the exact command to resume."""
    pytest.importorskip("langchain_openai")
    import httpx
    from langchain_openai.chat_models.base import OpenAIInvalidRequestError

    import main
    from src.errors import FatalModelError

    cause = OpenAIInvalidRequestError(
        "Error code: 400 - {'error': {...}}",
        response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
        body={"message": "Function tools with reasoning_effort are not supported.",
              "param": "reasoning_effort", "type": "invalid_request_error"},
    )
    text = main.explain_fatal(FatalModelError("analyst", cause), "run-42")

    assert "stopped at [analyst]" in text
    assert "Function tools with reasoning_effort are not supported." in text
    assert "{'error'" not in text, "the dict repr is noise; show the sentence"
    assert "parameter: reasoning_effort" in text
    assert "python main.py --resume --thread-id run-42" in text


def test_a_piped_run_survives_characters_the_code_page_lacks(tmp_path) -> None:
    """The real crash: `python main.py ... | Select-String ...` on Windows.

    Piped, Python encodes stdout with the legacy code page; cp1252 has no
    U+2212, which a model wrote in "(1 − discount_pct/100)", and `print`
    raised UnicodeEncodeError in main.py -- outside every node's safety net
    -- and the run died. Reproduced here in a subprocess with
    PYTHONIOENCODING=cp1252 and a pipe. Remove make_output_safe() from
    main() and this fails with that exact traceback.
    """
    import os
    import subprocess
    import sys

    driver = tmp_path / "drive.py"
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path.cwd())!r})\n"
        "from langchain_core.messages import AIMessage\n"
        "import tests.fake_llm as fake\n"
        "script = fake.default_script()\n"
        "script[3] = AIMessage(content='FINDING: revenue = units * (1 \\u2212 discount)\\n"
        "EVIDENCE: formula check')\n"
        "fake.default_script = lambda: list(script)\n"
        "sys.argv = ['main.py', 'data/sales.csv', 'q', '--provider', 'fake', '--no-approval']\n"
        "import main\n"
        "raise SystemExit(main.main())\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",
        "CHECKPOINT_DB": ":memory:",
        "REQUIRE_CHART": "false",
    }
    run = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,  # a pipe, exactly like `| Select-String`
        env=env,
        cwd=Path.cwd(),
        timeout=120,
    )

    out = run.stdout.decode("cp1252", errors="replace")
    err = run.stderr.decode("cp1252", errors="replace")
    assert "UnicodeEncodeError" not in err, err[-600:]
    assert run.returncode == 0, err[-600:]
    assert "finding: revenue = units * (1 ? discount)" in out


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGINT") or __import__("os").name == "nt",
                    reason="sends a POSIX SIGINT to a child process")
def test_ctrl_c_during_a_slow_model_call_exits_at_once_and_can_resume(tmp_path) -> None:
    """Ctrl+C used to wait for the model call in flight: 17.7 s on a 20 s
    request. Now it leaves immediately -- and the checkpoint written before
    the interrupted call still lets --resume finish the run."""
    import os
    import signal
    import subprocess
    import sys
    import time

    db = tmp_path / "ck.sqlite"
    slow = tmp_path / "slow.py"
    slow.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(Path.cwd())!r})\n"
        "import tests.fake_llm as fake\n"
        "orig = fake.ScriptedChatModel._generate\n"
        "def slow(self, *a, **k):\n"
        "    if not getattr(slow, 'once', False):\n"
        "        slow.once = True\n"
        "        time.sleep(30)\n"
        "    return orig(self, *a, **k)\n"
        "fake.ScriptedChatModel._generate = slow if sys.argv[1] == 'slow' else orig\n"
        "if sys.argv[1] == 'fast':\n"
        "    # resume continues at the interrupted analyst call: script from there\n"
        "    rest = fake.default_script()[2:]\n"
        "    fake.default_script = lambda: list(rest)\n"
        "sys.argv = ['main.py', *sys.argv[2:]]\n"
        "import main\n"
        "raise SystemExit(main.main())\n",
        encoding="utf-8",
    )
    env = {**os.environ, "CHECKPOINT_DB": str(db), "REQUIRE_CHART": "false"}
    first = subprocess.Popen(
        [sys.executable, str(slow), "slow", "data/sales.csv", "q", "--provider", "fake",
         "--no-approval", "--thread-id", "ctrlc"],
        cwd=Path.cwd(), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    time.sleep(4)  # the planner's structured call is not slowed; the first _generate is
    sent = time.monotonic()
    first.send_signal(signal.SIGINT)
    out, _ = first.communicate(timeout=30)
    took = time.monotonic() - sent

    assert first.returncode == 130, out.decode()[-400:]
    assert took < 3, f"Ctrl+C took {took:.1f}s"
    assert b"--resume --thread-id ctrlc" in out

    resumed = subprocess.run(
        [sys.executable, str(slow), "fast", "--resume", "--thread-id", "ctrlc", "--provider", "fake"],
        cwd=Path.cwd(), env=env, capture_output=True, timeout=60,
    )
    assert resumed.returncode == 0, resumed.stdout.decode()[-400:] + resumed.stderr.decode()[-400:]
    out2 = resumed.stdout.decode()
    assert "findings: 1" in out2, out2[-600:]
    assert "not written" not in out2
    assert "script" not in out2.lower(), "the resumed run must follow the script exactly"
