"""Message-window helpers.

Long agent runs blow the context window. The naive fix -- "keep the last N
messages" -- corrupts the conversation, because an `AIMessage` carrying
`tool_calls` and the `ToolMessage` answering it are a matched pair. Cut
between them and the provider rejects the request with an error like
"tool_use ids must have matching tool_result blocks".

`window()` keeps the pairing intact in *both* directions:

  * a `ToolMessage` whose `AIMessage` fell outside the window is dropped;
  * an `AIMessage` whose `ToolMessage` is missing has its `tool_calls`
    stripped and replaced by a plain note.

The second case is the one people forget. It happens whenever the graph
leaves a tool loop while a tool call is still pending -- a step budget kicked
in, a node failed, a human cancelled. The transcript then carries an
`AIMessage` asking for a tool that was never run, and the next provider call
fails on it even though the window itself is small.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def _answered_ids(messages: list[BaseMessage]) -> set[str]:
    return {
        msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)
    }


def _strip_unanswered_calls(
    messages: list[BaseMessage], answered: set[str]
) -> list[BaseMessage]:
    """Replace AIMessages holding unanswered tool_calls with a plain note."""
    result: list[BaseMessage] = []
    for msg in messages:
        calls = getattr(msg, "tool_calls", None) or []
        pending = [c for c in calls if c["id"] not in answered]
        if not pending:
            result.append(msg)
            continue

        names = ", ".join(c["name"] for c in pending)
        text = str(msg.content or "").strip()
        note = f"[requested {names}; not executed]"
        result.append(AIMessage(content=f"{text}\n{note}".strip(), id=msg.id))
    return result


def window(messages: list[BaseMessage], keep: int = 14) -> list[BaseMessage]:
    """Return a context-safe slice of the conversation."""
    if len(messages) <= keep:
        return _strip_unanswered_calls(messages, _answered_ids(messages))

    seed, tail = messages[0], messages[-keep:]

    # Which tool_call ids are *asked for* by an AIMessage inside the window?
    asked_ids = {
        call["id"]
        for msg in tail
        if isinstance(msg, AIMessage)
        for call in (msg.tool_calls or [])
    }

    # Drop ToolMessages answering a call we can no longer see.
    cleaned = [
        msg
        for msg in tail
        if not (isinstance(msg, ToolMessage) and msg.tool_call_id not in asked_ids)
    ]

    # ...and neutralise calls that were never answered.
    cleaned = _strip_unanswered_calls(cleaned, _answered_ids(cleaned))

    return [seed, *cleaned]


def has_tool_calls(message: BaseMessage) -> bool:
    return bool(getattr(message, "tool_calls", None))


def parse_tagged(text: str, tag: str) -> str:
    """Pull the text after `TAG:` out of a model reply. '' when absent."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(f"{tag.upper()}:"):
            return stripped[len(tag) + 1 :].strip()
    return ""
