"""A fake OpenAI *server*, for testing the real client offline.

`tests/fake_llm.py` replaces the whole chat model, which is right for
testing the graph and useless for testing the client: the bug lives in the
HTTP request the real ChatOpenAI builds, and a fake model never builds one.

This goes one layer lower. The real `langchain_openai.ChatOpenAI` and the
real `openai` SDK run unmodified; only the network is swapped for an
`httpx.MockTransport`. So a test sees exactly the JSON body that would have
gone to api.openai.com, and gets back the same 400 the real API sends.

The quirks are the ones observed on real runs of this project:

    reject_temperature        gpt-6-luna: "'temperature' does not support 0.0
                              with this model. Only the default (1) value is
                              supported."
    reject_tools_with_reasoning
                              gpt-5.6-terra: "Function tools with
                              reasoning_effort are not supported ... set
                              reasoning_effort to 'none'."

Both are sent even when the client never set the parameter in question,
which is the point: the API applies its own default and then refuses it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx


def _error(message: str, param: str, code: str | None) -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
    )


@dataclass
class FakeOpenAI:
    reject_temperature: bool = False
    reject_tools_with_reasoning: bool = False
    # What a successful structured-output call should return, as JSON text.
    structured_reply: dict[str, Any] = field(default_factory=dict)
    # Every request body the client sent, in order. The thing tests assert on.
    requests: list[dict[str, Any]] = field(default_factory=list)

    # --- the server --------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append(body)

        if self.reject_temperature and body.get("temperature", 1) != 1:
            return _error(
                f"Unsupported value: 'temperature' does not support "
                f"{body['temperature']} with this model. Only the default (1) "
                "value is supported.",
                param="temperature",
                code="unsupported_value",
            )

        if (
            self.reject_tools_with_reasoning
            and body.get("tools")
            and body.get("reasoning_effort") != "none"
        ):
            return _error(
                f"Function tools with reasoning_effort are not supported for "
                f"{body.get('model')} in /v1/chat/completions. To use function "
                "tools, use /v1/responses or set reasoning_effort to 'none'.",
                param="reasoning_effort",
                code=None,
            )

        message = self._reply_for(body)
        if body.get("stream"):
            return self._as_stream(body, message)
        return httpx.Response(200, json=self._completion(body, message))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    # --- replies -------------------------------------------------------------

    def _reply_for(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_fake_1",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                ],
            }
        if body.get("response_format"):
            return {"role": "assistant", "content": json.dumps(self.structured_reply)}
        return {"role": "assistant", "content": "pong"}

    @staticmethod
    def _completion(body: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @staticmethod
    def _as_stream(body: dict[str, Any], message: dict[str, Any]) -> httpx.Response:
        delta: dict[str, Any] = {"role": "assistant"}
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {"index": 0, **call} for call in message["tool_calls"]
            ]
        else:
            delta["content"] = message.get("content") or ""

        def chunk(d: dict[str, Any], finish: str | None) -> str:
            payload = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body.get("model", "fake"),
                "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        finish = "tool_calls" if message.get("tool_calls") else "stop"
        text = chunk(delta, None) + chunk({}, finish) + "data: [DONE]\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=text.encode()
        )
