"""Provider wiring. No network: the clients are constructed, never called."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.llm import get_llm


@pytest.fixture(autouse=True)
def clear_cache():
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


@pytest.fixture(autouse=True)
def dummy_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")


def test_resolved_model_respects_the_provider_argument(clean_env) -> None:
    """The regression: it used to read the *configured* provider instead.

    `get_llm("anthropic")` then returned a ChatAnthropic client holding
    "gpt-4o-mini", and the failure only showed up as a 404 from the API.

    Note the `clean_env` argument rather than a module-level `settings`
    import. `from src.config import settings` binds the object built at
    import time -- from the developer's real `.env` -- and no fixture can
    reach that binding. Take config from a fixture, and the test states the
    configuration it needs instead of inheriting whatever is on the machine.
    """
    assert clean_env.resolved_model("anthropic") == Settings.DEFAULT_MODELS["anthropic"]
    assert clean_env.resolved_model("openai") == Settings.DEFAULT_MODELS["openai"]
    assert clean_env.resolved_model("anthropic") != clean_env.resolved_model("openai")


@pytest.mark.parametrize(
    ("provider", "expected_class", "package"),
    [
        ("openai", "ChatOpenAI", "langchain_openai"),
        ("anthropic", "ChatAnthropic", "langchain_anthropic"),
    ],
)
def test_each_provider_builds_its_own_client_with_its_own_model(
    provider: str, expected_class: str, package: str, clean_env
) -> None:
    """One provider's package being absent must not fail the suite.

    `importorskip` is the tool for that. Without it, commenting a provider
    out of requirements.txt turns a deliberate "this one is optional" into a
    red test suite on everyone else's machine -- which is exactly what
    happened here, and it is the kind of break that makes people stop
    trusting `pytest`.
    """
    module = pytest.importorskip(package, reason=f"{package} is an optional provider")

    client = get_llm(provider)
    # `isinstance`, not `type(...).__name__ ==`. The OpenAI client is a
    # subclass now (see test_temperature_rejection_*), and an equality check
    # on the class name would call that a regression when it is the fix.
    assert isinstance(client, getattr(module, expected_class))

    name = getattr(client, "model_name", None) or getattr(client, "model", "")
    assert name == Settings.DEFAULT_MODELS[provider], (
        f"{provider} client was built with model {name!r}"
    )


def test_an_unknown_provider_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm("gpt5-turbo-ultra")


def test_the_client_is_cached_not_rebuilt_per_node() -> None:
    assert get_llm("openai") is get_llm("openai")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 0.0), ("0", 0.0), ("0.7", 0.7), ("", None), ("default", None)],
)
def test_temperature_can_be_configured_as_unset(raw, expected, monkeypatch) -> None:
    """"Send nothing" has to be expressible, and it is not the same as zero."""
    import src.config

    if raw is None:
        monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("LLM_TEMPERATURE", raw)

    assert src.config.Settings().temperature == expected
