"""Central configuration, read once from the environment.

Everything that you might want to tweak without touching code lives here.
"""

from __future__ import annotations

import io
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigError(ValueError):
    """A setting in .env (or the environment) cannot be used as written.

    Raised with a sentence a person can act on. The alternative was a raw
    traceback from deep inside Settings -- "could not convert string to
    float: '0,2'" -- which names neither the setting nor the file.
    """


def _read_env_text(path: Path) -> str:
    """The .env file as text, whatever Windows saved it as.

    Two real traps on Windows:
    * PowerShell 5.1's `echo LLM_PROVIDER=openai > .env` writes UTF-16. The
      old loader crashed on import with a UnicodeDecodeError.
    * Notepad's "UTF-8 with BOM" puts an invisible \ufeff before the first
      key, so `LLM_PROVIDER=...` on the first line was read as a key named
      "\ufeffLLM_PROVIDER" -- and silently ignored.
    """
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")  # strips a BOM if there is one
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{path} is not UTF-8 text. Open it in VS Code and save it with "
            "the encoding 'UTF-8' (bottom-right corner of the window)."
        ) from exc


# `KEY=   # comment`: python-dotenv reads the comment AS the value when the
# value itself is blank. .env.example documents LLM_TEMPERATURE= (blank) as a
# legal setting, so a comment after it crashed the import.
_BLANK_WITH_COMMENT = re.compile(r"(?m)^(\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=)[ \t]+#.*$")


def _clean_env_text(text: str) -> str:
    return _BLANK_WITH_COMMENT.sub(r"\1", text)


def _warn_on_duplicate_env_keys(path: Path) -> list[str]:
    """Warn about keys assigned twice in .env, and return their names.

    python-dotenv keeps the *last* assignment and says nothing about it. So
    an .env where you carefully set MAX_STEPS=6 near the top, while an older
    MAX_STEPS=12 still sits further down, runs with 12 -- and nothing in the
    output hints at why your setting "did not work". Config that is silently
    ignored is one of the most expensive kinds of bug to chase, so it is
    worth six lines to make it loud.
    """
    if not path.exists():
        return []

    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for raw in _read_env_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        if key in seen:
            if key not in duplicates:
                duplicates.append(key)
        seen[key] = 1

    for key in duplicates:
        print(
            f"warning: {key!r} is set more than once in {path.name}; "
            "only the LAST value is used",
            file=sys.stderr,
        )
    return duplicates


def _load_env_file(path: Path) -> None:
    """Load exactly the project's .env -- never one found somewhere else.

    A bare `load_dotenv()` searches upward from the calling file, and from
    the current directory when a debugger or coverage tool is active. With
    no .env in the project, a .env in a PARENT folder was loaded, while
    check_llm.py reported ".env MISSING" -- the two disagreed about which
    file was in effect.
    """
    if path.exists():
        load_dotenv(stream=io.StringIO(_clean_env_text(_read_env_text(path))), override=False)


DUPLICATE_ENV_KEYS = _warn_on_duplicate_env_keys(ENV_FILE)
_load_env_file(ENV_FILE)

# Where runs write their folders, reports and charts. Settable (OUTPUT_DIR in
# .env) so the web app's runs can live on a drive the team backs up -- and so
# the test-suite's subprocesses stop writing into the real outputs/ folder.
if os.environ.get("OUTPUT_DIR", "").strip():
    OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"].strip()).expanduser().resolve()


_TRUE = {"1", "true", "yes", "on", "ja", "an"}
_FALSE = {"0", "false", "no", "off", "nein", "aus"}


def _as_bool(value: str | None, default: bool = False, name: str = "setting") -> bool:
    """true/false, yes/no, on/off, 1/0 (and ja/nein) -- anything else is an error.

    It used to be "anything not in the true-list is False", so a German
    REQUIRE_APPROVAL=ja switched plan approval OFF, silently.
    """
    if value is None or not value.strip():
        return default
    text = value.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{name}={value!r} is not a yes/no value. Use true or false.")


def _as_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        number = int(raw.strip())
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a whole number (for example {name}={default}).") from None
    if number < minimum:
        raise ConfigError(f"{name}={number} is too small; it must be at least {minimum}.")
    return number


def _as_temperature(value: str | None) -> float | None:
    """Parse LLM_TEMPERATURE, where "nothing" is a legal answer.

    `None` means *do not send the parameter at all* -- which is different
    from sending zero. Some models accept only their own default and reject
    an explicit `temperature=0` with a 400, so "unset" has to be expressible.
    `LLM_TEMPERATURE=` (blank) in .env says it. A decimal comma (0,2 -- the
    German way to write it) is accepted.
    """
    if value is None:
        return 0.0
    text = value.strip().lower()
    if text in {"", "default", "none", "unset"}:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        raise ConfigError(
            f"LLM_TEMPERATURE={value!r} is not a number. Use for example 0, 0.2, "
            "or leave it blank to send no temperature."
        ) from None


@dataclass(frozen=True)
class Settings:
    # --- LLM ---------------------------------------------------------------
    # "openai" | "anthropic" | "fake"
    # Lower-cased: "Anthropic" used to pass the key check (which compared
    # exact strings) and show an OpenAI model name, while get_llm quietly
    # lower-cased it and built the Anthropic client anyway.
    provider: str = field(
        default_factory=lambda: (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    )
    model: str | None = field(
        default_factory=lambda: (os.getenv("LLM_MODEL") or "").strip() or None
    )
    # Seconds to wait for one model request. Without it the SDK default
    # applied -- ten minutes per attempt -- and Ctrl+C had to wait it out.
    llm_timeout: int = field(default_factory=lambda: _as_int("LLM_TIMEOUT", 120, minimum=1))
    # None = send no temperature at all, and let the model use its default.
    temperature: float | None = field(
        default_factory=lambda: _as_temperature(os.getenv("LLM_TEMPERATURE"))
    )
    # --- Graph behaviour ---------------------------------------------------
    # Ask a human to approve the analysis plan before any code runs.
    require_approval: bool = field(
        default_factory=lambda: _as_bool(os.getenv("REQUIRE_APPROVAL"), True, "REQUIRE_APPROVAL")
    )
    # Every report with findings gets at least one chart. The supervisor
    # prompt asks for charts; on a real run gpt-5.6-terra went straight to
    # the reporter anyway, twice. A prompt asks, code guarantees.
    require_chart: bool = field(
        default_factory=lambda: _as_bool(os.getenv("REQUIRE_CHART"), True, "REQUIRE_CHART")
    )
    # Total node failures tolerated in one run before we stop and write a
    # partial report. A whole-run budget, not a per-node retry count.
    max_failures: int = field(
        default_factory=lambda: _as_int("MAX_FAILURES", 3)
    )
    # Upper bound on model turns in one run (supervisor + workers), a safety
    # net against loops. It was 12, sized for the scripted test model; a real
    # model takes three or four turns per plan step, and 12 ran out before
    # the question was answered. See src/budget.py for how it is divided.
    max_steps: int = field(default_factory=lambda: _as_int("MAX_STEPS", 30, minimum=1))
    # Turns a worker gets for ONE dispatched plan step before it must wrap
    # up with what it has. Without it, one careful step ate the whole budget.
    max_tool_rounds: int = field(
        default_factory=lambda: _as_int("MAX_TOOL_ROUNDS", 4, minimum=1)
    )
    # SQLite file used by the checkpointer. ":memory:" disables persistence.
    checkpoint_db: str = field(
        default_factory=lambda: os.getenv("CHECKPOINT_DB", "outputs/checkpoints.sqlite")
    )

    # --- Sandbox -----------------------------------------------------------
    # Wall-clock limit for one generated pandas snippet, in seconds.
    code_timeout: int = field(default_factory=lambda: _as_int("CODE_TIMEOUT", 15, minimum=1))
    # Rows a tool may return to the model. Keeps the context window sane.
    max_rows_to_model: int = field(
        default_factory=lambda: _as_int("MAX_ROWS_TO_MODEL", 20, minimum=1)
    )

    # Default model name per provider, used when LLM_MODEL is unset.
    #
    # These rot. `gpt-4o-mini` was the right default when this project was
    # written and was retired from OpenAI's line-up within months, so a fresh
    # checkout would fail with "model not found" for a reason that has nothing
    # to do with the code. Treat every hard-coded model name as a value with
    # an expiry date: keep it in one place, let LLM_MODEL override it, and
    # make the failure legible when it happens.
    DEFAULT_MODELS = {
        "openai": "gpt-6-luna",
        "anthropic": "claude-sonnet-5",
        "fake": "fake",
    }

    def resolved_model(self, provider: str | None = None) -> str:
        """Default model name for `provider` (the configured one when omitted).

        The `provider` argument matters: without it, asking for a model while
        overriding the provider -- `get_llm("anthropic")` -- silently returned
        the *configured* provider's default, so you got ChatAnthropic holding
        an OpenAI model name and a confusing 404 from the API. Any "resolve a
        default for X" helper must take X as an argument rather than reading
        it from global state.
        """
        if self.model:
            return self.model
        return self.DEFAULT_MODELS.get(provider or self.provider, "gpt-6-luna")


settings = Settings()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
