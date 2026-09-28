"""Node-level error handling.

An agent that crashes on the first bad tool call is useless in production.
The pattern here: every node is wrapped so that an exception becomes *data*
in the state instead of a traceback that kills the run. The supervisor then
sees the error on its next turn and can route around it -- retry a different
way, skip the step, or go straight to writing a partial report.

This is the agent equivalent of "return an error, don't throw".
"""

from __future__ import annotations

import functools
import logging
import traceback
from typing import Any, Callable

from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
)
from langgraph.errors import GraphBubbleUp

logger = logging.getLogger("csv_agent")


# Errors where the *next* request would be identical and fail identically.
#
# "Turn the exception into data and let the supervisor route around it" is
# right for a 429, a timeout, a 500, or a pandas snippet that raised: the
# world changes, or the model writes different code, and the retry can work.
#
# It is exactly wrong for these. A bad key, a model name that does not
# exist, a parameter the model refuses: the supervisor cannot change any of
# them, because none of them is in the conversation. A real run showed what
# happens when you try anyway. gpt-5.6-terra refused `reasoning_effort` four
# times in a row, and the supervisor, reading the error text, started
# writing "using pandas with reasoning_effort disabled" into its
# instructions -- an LLM trying to fix an HTTP parameter by writing prose.
# The whole failure budget went on it, and the run ended with nothing.
#
# These are LangChain's provider-neutral classes: langchain-openai and
# langchain-anthropic both raise subclasses of them, so this tuple needs no
# vendor SDK and covers both providers.
UNFIXABLE_BY_RETRY = (
    ModelAuthenticationError,   # 401: the key
    ModelPermissionDeniedError, # 403: the key's access
    ModelNotFoundError,         # 404: the model name
    ModelInvalidRequestError,   # 400: a request the model will never accept
)


class FatalModelError(RuntimeError):
    """The model API refused the request, and retrying cannot help.

    Raised *out of* the graph on purpose. The run stops, but the checkpoint
    from just before the failing node is kept -- so after fixing .env,
    `--resume` continues from that node without having spent any budget.
    """

    def __init__(self, node: str, cause: Exception) -> None:
        self.node = node
        self.cause = cause
        super().__init__(f"{node}: {type(cause).__name__}: {cause}")


def safe_node(name: str) -> Callable:
    """Wrap a node so exceptions land in `state["errors"]`.

    Except the ones in UNFIXABLE_BY_RETRY: those stop the run as a
    FatalModelError, because feeding them to the supervisor only spends the
    failure budget on an error nobody inside the graph can fix.

    On failure the wrapped node returns:
        errors:       the formatted error (accumulated by the reducer)
        failure_count: +1
        failed_node:  which node blew up

    On success it returns `failed_node: ""`.

    That last line is not cosmetic, and getting it wrong is the subtlest bug
    in this whole project. `failed_node` is a *sticky* flag: the routing
    functions read it to decide "did the node I just ran fail?". If a node
    never clears it on success, then one transient 429 on the supervisor
    leaves the flag set for the rest of the run, and every routing decision
    afterwards is made on stale information -- the graph either exits early
    with nothing or loops between two nodes until the recursion limit.
    Clear-on-success turns it back into what the name promises.
    """

    def decorator(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(func)
        def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                result = func(state, *args, **kwargs) or {}
                result.setdefault("failed_node", "")
                return result
            except GraphBubbleUp:
                # `interrupt()` and `Command` work by raising GraphBubbleUp
                # subclasses that the runtime is meant to catch. A bare
                # `except Exception` would swallow them and turn a
                # human-in-the-loop pause into a logged error -- one of the
                # nastier bugs you can write in LangGraph, because the graph
                # just quietly carries on without ever asking anyone.
                raise
            except UNFIXABLE_BY_RETRY as exc:
                # debug, not warning: main.py prints this one properly, and
                # the same error twice in a row reads like two failures.
                logger.debug("node %s: model API refused the request: %s", name, exc)
                raise FatalModelError(name, exc) from exc
            except Exception as exc:  # noqa: BLE001 - deliberate catch-all
                logger.warning("node %s failed: %s", name, exc)
                logger.debug("%s", traceback.format_exc())
                return {
                    "errors": [f"{name}: {type(exc).__name__}: {exc}"],
                    "failure_count": 1,
                    "failed_node": name,
                }

        return wrapper

    return decorator


def errors_block(state: dict, limit: int = 3) -> str:
    """Render recent errors for inclusion in a prompt."""
    errors = state.get("errors") or []
    if not errors:
        return ""
    recent = errors[-limit:]
    body = "\n".join(f"- {e}" for e in recent)
    return f"\nRecent failures you must work around:\n{body}\n"
