"""Config loading, including the traps."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import _warn_on_duplicate_env_keys


def test_duplicate_keys_are_detected(tmp_path: Path, capsys) -> None:
    """A silently-ignored setting is the bug this guards against.

    python-dotenv keeps the last assignment and never mentions it, so an
    .env with MAX_STEPS twice runs with a value you did not choose.
    """
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "MAX_STEPS=6",
                "# a comment = not a key",
                "LLM_PROVIDER=anthropic",
                "MAX_STEPS=12",  # the one that actually wins
                "export LLM_MODEL=claude-sonnet-5",
                "LLM_MODEL=claude-haiku-4-5",
            ]
        ),
        encoding="utf-8",
    )

    duplicates = _warn_on_duplicate_env_keys(env)

    assert set(duplicates) == {"MAX_STEPS", "LLM_MODEL"}
    assert "MAX_STEPS" in capsys.readouterr().err


def test_a_clean_env_warns_about_nothing(tmp_path: Path, capsys) -> None:
    env = tmp_path / ".env"
    env.write_text("MAX_STEPS=6\nLLM_PROVIDER=anthropic\n", encoding="utf-8")

    assert _warn_on_duplicate_env_keys(env) == []
    assert capsys.readouterr().err == ""


def test_a_missing_env_is_fine(tmp_path: Path) -> None:
    assert _warn_on_duplicate_env_keys(tmp_path / "nope.env") == []


def test_the_shipped_example_has_no_duplicates(capsys) -> None:
    """The .env.example people copy must not itself contain the trap."""
    from src.config import PROJECT_ROOT

    assert _warn_on_duplicate_env_keys(PROJECT_ROOT / ".env.example") == []


def test_the_suite_is_insulated_from_a_local_env_file(clean_env) -> None:
    """Guard on the guard.

    If someone later removes the `clean_env` fixture, this fails and says
    why, instead of the suite quietly becoming config-dependent again.
    """
    assert clean_env.provider == "openai", "a local .env leaked into the tests"
    assert clean_env.model is None
    # the defaults the rest of the suite assumes
    assert clean_env.max_steps == 30
    assert clean_env.max_failures == 3
    assert clean_env.resolved_model("anthropic") == "claude-sonnet-5"
    assert clean_env.resolved_model("openai") == "gpt-6-luna"


# --- schema bounds ---------------------------------------------------------


def test_every_free_text_field_the_model_fills_is_length_bounded() -> None:
    """Structured output constrains shape, not length.

    An unbounded `str` in a routing schema lets a model write an essay inside
    a decision. Observed on a local qwen2.5: ~650 tokens of prose for a
    routing turn that needs ~50, which on CPU is over a minute of waiting.
    If a new free-text field is added without a bound, this test fails.
    """
    from src.schemas import Plan, Route

    unbounded: list[str] = []
    for model in (Route, Plan):
        schema = model.model_json_schema()
        for name, prop in schema["properties"].items():
            if prop.get("type") == "string" and "enum" not in prop:
                if "maxLength" not in prop:
                    unbounded.append(f"{model.__name__}.{name}")

    assert not unbounded, f"free-text fields with no maxLength: {unbounded}"


def test_the_bound_is_actually_enforced() -> None:
    from pydantic import ValidationError

    from src.schemas import Route

    with pytest.raises(ValidationError, match="at most 200 characters"):
        Route(next="analyst", instruction="x" * 250, reason="fine")
