"""One place that knows how to build a chat model.

Every node asks for a model through `get_llm()`, so switching provider is a
single environment variable and never a code change. This is the pattern you
want in any real agent: nodes depend on the *interface* (BaseChatModel), not
on a concrete vendor SDK.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import settings

# Request parameters we know how to repair when a model refuses them. Keyed by
# *parameter name* -- a stable part of the API -- never by model name.
REPAIRABLE_PARAMETERS = ("temperature", "reasoning_effort")


def rejected_parameter(exc: Exception) -> str | None:
    """Name the request parameter a provider's 400 blames, or None.

    Two real runs of this project died on a parameter, not on the analysis:

      gpt-6-luna     'temperature' does not support 0.0 with this model.
                     Only the default (1) value is supported.

      gpt-5.6-terra  Function tools with reasoning_effort are not supported
                     ... set reasoning_effort to 'none'.

    The second one is worth staring at: this project never *sends*
    reasoning_effort. The model applied its own default and then refused the
    combination with tools. No amount of care in our request would have
    predicted that.

    The tempting fix is a list of model names with their quirks. Resist it.
    `gpt-4o-mini` vanished from OpenAI's line-up within months, and
    langchain-openai itself keeps exactly such a list -- `gpt-5.6-terra` is
    not on it, which is why it was sent to the wrong endpoint. A list of
    model quirks rots silently: nothing fails until someone picks a model
    released after the list was written.

    The error is the better source. It is structured, it names the offending
    parameter (`exc.param`, or `body.error.param`), and it comes from the one
    party that actually knows. Read the 400 instead of predicting it.
    """
    param = getattr(exc, "param", None)
    if isinstance(param, str) and param:
        return param

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        param = error.get("param")
        if isinstance(param, str) and param:
            return param

    # Fallback for clients that expose neither: recognise only parameters we
    # can repair, and only in a message that is about something unsupported.
    text = str(exc).lower()
    if "unsupported" in text or "not support" in text:
        for name in REPAIRABLE_PARAMETERS:
            if name in text:
                return name
    return None


@lru_cache(maxsize=None)
def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Return a chat model for the configured provider.

    Cached, so all nodes share one client (and one connection pool) -- and,
    for OpenAI, so a repair learned on the first call holds for the run.
    """
    provider = (provider or settings.provider).lower()
    model = model or settings.resolved_model(provider)

    if provider == "openai":
        return _build_openai(model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=settings.temperature,
            timeout=settings.llm_timeout,
            max_retries=2,
        )

    if provider == "fake":
        # Only used by the test-suite; keeps the graph runnable with no API key.
        from tests.fake_llm import scripted_model

        return scripted_model()

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: openai, anthropic, fake."
    )


def _note(message: str) -> None:
    print(f"note: {message}", file=sys.stderr)


_OPENAI_CLASS: type | None = None

# One repair per known parameter; a bound so a confused server can never put
# a request into a retry loop.
_MAX_REPAIRS = len(REPAIRABLE_PARAMETERS)


def _openai_class() -> type:
    """Build (once) a ChatOpenAI that repairs a refused parameter and retries.

    Why a subclass rather than a try/except at the call site: the graph
    reaches the model through four different doors -- `invoke`,
    `with_structured_output(...)`, `bind_tools(...)`, and the token stream
    that draws the progress line. A fix at one call site leaves the other
    three broken, and you find that out one node at a time. These four
    methods are the floor every door ends up on, so a repair here covers all
    of them.

    Repairs are *scoped*. The gpt-5.6-terra refusal is about function tools
    specifically: its structured-output calls (planner, supervisor) worked
    with reasoning on. So `reasoning_effort="none"` is applied only to calls
    that carry tools, and the nodes that were fine keep their reasoning.
    """
    global _OPENAI_CLASS
    if _OPENAI_CLASS is not None:
        return _OPENAI_CLASS

    from langchain_openai import ChatOpenAI
    from pydantic import PrivateAttr

    class ChatOpenAIAdaptive(ChatOpenAI):
        # scope -> {parameter: value}; scope is "tools" or "all".
        _learned: dict = PrivateAttr(default_factory=dict)

        # --- what we learned, and how to apply it ----------------------------

        def _repair(self, exc: Exception, kwargs: dict) -> bool:
            """Change the request so the refused parameter is acceptable.

            Returns False when there is nothing (left) to change, so the
            caller re-raises instead of sending the same request again.
            """
            param = rejected_parameter(exc)
            name = getattr(self, "model_name", None) or "this model"

            if param == "temperature":
                if self.temperature is None:
                    return False
                _note(
                    f"{name} does not accept temperature={self.temperature}; "
                    "using the model's default instead. Set LLM_TEMPERATURE= "
                    "(blank) in .env to skip this on every run."
                )
                self.temperature = None
                return True

            if param == "reasoning_effort":
                scope = "tools" if kwargs.get("tools") else "all"
                learned = self._learned.setdefault(scope, {})
                if learned.get("reasoning_effort") == "none":
                    return False
                learned["reasoning_effort"] = "none"
                where = "calls that use tools" if scope == "tools" else "every call"
                _note(
                    f"{name} refuses its default reasoning for {where}; "
                    f"sending reasoning_effort=none on {where} from now on."
                )
                return True

            return False

        def _prepared(self, kwargs: dict) -> dict:
            extra = dict(self._learned.get("all", {}))
            if kwargs.get("tools"):
                extra.update(self._learned.get("tools", {}))
            return {**kwargs, **extra} if extra else kwargs

        # --- the four floors ---------------------------------------------------

        def _generate(self, *args: Any, **kwargs: Any):
            repairs = 0
            while True:
                try:
                    return super()._generate(*args, **self._prepared(kwargs))
                except Exception as exc:
                    if repairs >= _MAX_REPAIRS or not self._repair(exc, kwargs):
                        raise
                    repairs += 1

        async def _agenerate(self, *args: Any, **kwargs: Any):
            repairs = 0
            while True:
                try:
                    return await super()._agenerate(*args, **self._prepared(kwargs))
                except Exception as exc:
                    if repairs >= _MAX_REPAIRS or not self._repair(exc, kwargs):
                        raise
                    repairs += 1

        def _stream(self, *args: Any, **kwargs: Any):
            repairs = 0
            while True:
                started = False
                try:
                    for chunk in super()._stream(*args, **self._prepared(kwargs)):
                        started = True
                        yield chunk
                    return
                except Exception as exc:
                    # Retry only while the caller has seen nothing. Replaying
                    # a stream that already produced output would hand the
                    # reader its first half twice.
                    if (
                        started
                        or repairs >= _MAX_REPAIRS
                        or not self._repair(exc, kwargs)
                    ):
                        raise
                    repairs += 1

        async def _astream(self, *args: Any, **kwargs: Any):
            repairs = 0
            while True:
                started = False
                try:
                    async for chunk in super()._astream(*args, **self._prepared(kwargs)):
                        started = True
                        yield chunk
                    return
                except Exception as exc:
                    if (
                        started
                        or repairs >= _MAX_REPAIRS
                        or not self._repair(exc, kwargs)
                    ):
                        raise
                    repairs += 1

    _OPENAI_CLASS = ChatOpenAIAdaptive
    return _OPENAI_CLASS


def _build_openai(model: str, **extra: Any) -> BaseChatModel:
    """`extra` exists for tests, which pass a fake `http_client`."""
    # An explicit timeout: without one the SDK waited up to ten minutes per
    # attempt, and a hung request held the run -- and Ctrl+C -- for that long.
    options = {"timeout": settings.llm_timeout, "max_retries": 2, **extra}
    return _openai_class()(model=model, temperature=settings.temperature, **options)
