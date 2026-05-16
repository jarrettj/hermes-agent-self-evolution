"""OpenAI-compatible HTTP shim that routes chat completions through the
Claude Code CLI's OAuth subscription (via the official ``claude-agent-sdk``).

This lets tools that speak OpenAI (litellm, dspy, etc.) use a Claude Code
subscription instead of an Anthropic Console API key. Useful when:

  * you already pay for Claude Code and don't want a second metered API key
  * you want token usage to count against your subscription, not a separate
    billing account

USAGE
-----
Start the shim (defaults to 127.0.0.1:8765)::

    python scripts/claude_oai_shim.py

Point an OpenAI-compatible client at it::

    export OPENAI_BASE_URL=http://127.0.0.1:8765/v1
    export OPENAI_API_KEY=anything   # ignored, but most clients require it
    python -m evolution.skills.evolve_skill \\
        --skill github-code-review \\
        --optimizer-model openai/claude-opus-4-7 \\
        --eval-model openai/claude-haiku-4-5

MODEL ROUTING
-------------
Use OpenAI-style model names with a ``claude-*`` suffix. The shim maps
families to Claude Code's three tiers; specific minor versions resolve to
the latest of that family.

    claude-opus-*    -> opus
    claude-sonnet-*  -> sonnet
    claude-haiku-*   -> haiku

ENDPOINTS
---------
GET  /health                  -> {"ok": true}
POST /v1/chat/completions     -> OpenAI Chat Completions response shape

CAVEATS
-------
- Streaming is not yet supported (``stream: true`` is ignored, full response
  is buffered then returned).
- Token usage counts are best-effort and may be 0 if the SDK doesn't expose
  them for the current model.
- This routes through your Claude Code subscription. Review your plan's
  acceptable-use terms before automating against it.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import uvicorn
from claude_agent_sdk import ClaudeAgentOptions, query
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODEL_MAP = {
    "opus": "claude-opus-4-1-20250805",
    "sonnet": "claude-sonnet-4-5",
    "haiku": "claude-haiku-4-5",
}


def _resolve_model(name: str) -> str:
    """Map an OpenAI-style model name to a Claude tier."""
    n = name.lower()
    if "opus" in n:
        return MODEL_MAP["opus"]
    if "haiku" in n:
        return MODEL_MAP["haiku"]
    if "sonnet" in n or "claude" in n:
        return MODEL_MAP["sonnet"]
    # default: sonnet
    return MODEL_MAP["sonnet"]


def _messages_to_prompt(messages: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Flatten OpenAI messages into a single prompt + optional system."""
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # OpenAI content blocks -> concatenate text parts
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"Assistant: {content}")
        else:
            convo_parts.append(f"User: {content}")
    system = "\n\n".join(system_parts) if system_parts else None
    prompt = "\n\n".join(convo_parts).strip()
    return prompt, system


app = FastAPI(title="claude-oai-shim", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "claude-sonnet-4-5")

    resolved = _resolve_model(model)
    prompt, system = _messages_to_prompt(messages)

    options = ClaudeAgentOptions(model=resolved, system_prompt=system) if system else ClaudeAgentOptions(model=resolved)

    text_chunks: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    async for message in query(prompt=prompt, options=options):
        # AssistantMessage with TextBlock children carries the response
        content = getattr(message, "content", None)
        if content:
            for block in content:
                t = getattr(block, "text", None)
                if t:
                    text_chunks.append(t)
        # ResultMessage carries usage at the end
        u = getattr(message, "usage", None)
        if u:
            usage["input_tokens"] = u.get("input_tokens", 0) or 0
            usage["output_tokens"] = u.get("output_tokens", 0) or 0

    response_text = "".join(text_chunks).strip()

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            },
        }
    )


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
