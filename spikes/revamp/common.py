"""Throwaway spike helpers for the conductor revamp (never merged).

Plain httpx against llama-swap's OpenAI endpoint, one call at a time, with the
model's own sampling. Results are appended to JSONL so a run can be resumed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

BASE = os.environ.get("LLAMA_SWAP", "http://127.0.0.1:8200")
OUT = Path(__file__).with_name("results")
OUT.mkdir(exist_ok=True)

# Per-model published sampling (agent-standard/model-profile.md; chess profiles).
SAMPLING: dict[str, dict[str, float]] = {
    "gemma-4-12b": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "gemma-4-26b-a4b": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "qwen38-27b": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}

_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))


@dataclass
class Reply:
    content: str
    reasoning: str
    seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish: str | None


def chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    schema: dict[str, Any] | None = None,
    thinking: bool = False,
    temperature: float | None = None,
    max_tokens: int = 800,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[Reply, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **SAMPLING[model],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "plan", "strict": True, "schema": schema},
        }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    started = time.monotonic()
    response = _client.post(f"{BASE}/v1/chat/completions", json=payload)
    seconds = time.monotonic() - started
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    message = choice["message"]
    usage = body.get("usage") or {}
    return (
        Reply(
            content=message.get("content") or "",
            reasoning=message.get("reasoning_content") or "",
            seconds=seconds,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish=choice.get("finish_reason"),
        ),
        message,
    )


def wilson(successes: int, n: int, z: float = 1.6449) -> tuple[float, float]:
    """One-sided 95% Wilson interval (the fleet gate's z)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]
