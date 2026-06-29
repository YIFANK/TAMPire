"""Cerebras client wrapper (OpenAI-compatible chat completions).

Provides:
  - chat()        text/vision call -> (text, CallMetrics)
  - chat_json()   same but robustly parses a JSON object out of the reply
  - stream()      token generator for the live demo
  - vision helpers (encode an image into a content part)
  - METRICS       global aggregator powering the on-screen latency counter
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

from .config import CONFIG


# ----------------------------------------------------------------------------
# Metrics (the "speed is the architecture" story lives here)
# ----------------------------------------------------------------------------
@dataclass
class CallMetrics:
    label: str = ""
    total_s: float = 0.0          # wall-clock incl. network
    model_s: float = 0.0          # Cerebras-reported completion compute time
    queue_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_tokens: int = 0


@dataclass
class MetricsAggregator:
    calls: List[CallMetrics] = field(default_factory=list)

    def add(self, m: CallMetrics) -> None:
        self.calls.append(m)

    def reset(self) -> None:
        self.calls.clear()

    @property
    def n(self) -> int:
        return len(self.calls)

    @property
    def total_wall_s(self) -> float:
        return sum(c.total_s for c in self.calls)

    @property
    def total_model_s(self) -> float:
        return sum(c.model_s for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    @property
    def tokens_per_s(self) -> float:
        comp = sum(c.completion_tokens for c in self.calls)
        return comp / self.total_model_s if self.total_model_s > 0 else 0.0


METRICS = MetricsAggregator()


# ----------------------------------------------------------------------------
# Message construction helpers
# ----------------------------------------------------------------------------
def encode_image(path: str) -> str:
    """Read an image file -> data URL string."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def user_with_image(text: str, image_path: str) -> Dict[str, Any]:
    """Build a multimodal user message (text + one image)."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
        ],
    }


def sys(text: str) -> Dict[str, Any]:
    return {"role": "system", "content": text}


def user(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


# ----------------------------------------------------------------------------
# Core calls
# ----------------------------------------------------------------------------
class CerebrasError(RuntimeError):
    pass


def _post(payload: Dict[str, Any], stream: bool = False, *, max_retries: int = 7) -> requests.Response:
    CONFIG.require_key()
    url = f"{CONFIG.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CONFIG.api_key}",
        "Content-Type": "application/json",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=CONFIG.timeout_s, stream=stream
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e  # transient network blip — back off and retry
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = CerebrasError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            else:
                raise CerebrasError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        # exponential backoff capped at 12s (survives short API outages): 0.5,1,2,4,8,12,12
        time.sleep(min(12.0, 0.5 * (2 ** attempt)))
    raise CerebrasError(f"request failed after {max_retries} retries: {last_exc}")


def chat(
    messages: List[Dict[str, Any]],
    *,
    label: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, CallMetrics]:
    payload = {
        "model": CONFIG.model,
        "messages": messages,
        "temperature": CONFIG.temperature if temperature is None else temperature,
        "max_tokens": CONFIG.max_tokens if max_tokens is None else max_tokens,
    }
    t0 = time.time()
    resp = _post(payload)
    data = resp.json()
    total_s = time.time() - t0

    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {}) or {}
    tinfo = data.get("time_info", {}) or {}
    m = CallMetrics(
        label=label,
        total_s=total_s,
        model_s=float(tinfo.get("completion_time", 0.0)),
        queue_s=float(tinfo.get("queue_time", 0.0)),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        image_tokens=int(usage.get("image_tokens", 0)),
    )
    METRICS.add(m)
    return text, m


def stream(
    messages: List[Dict[str, Any]],
    *,
    label: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Iterator[str]:
    """Yield content deltas. Used by the demo to stream debate live."""
    payload = {
        "model": CONFIG.model,
        "messages": messages,
        "temperature": CONFIG.temperature if temperature is None else temperature,
        "max_tokens": CONFIG.max_tokens if max_tokens is None else max_tokens,
        "stream": True,
    }
    t0 = time.time()
    resp = _post(payload, stream=True)
    completion_text = []
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        line = line[len("data:"):].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        delta = chunk["choices"][0].get("delta", {}).get("content")
        if delta:
            completion_text.append(delta)
            yield delta
    # streaming gives no usage block; record a coarse metric
    METRICS.add(CallMetrics(label=label, total_s=time.time() - t0))


# ----------------------------------------------------------------------------
# JSON parsing
# ----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply, tolerating fences
    and surrounding prose."""
    text = text.strip()
    # 1) fenced block
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            text = candidate  # fall through to brace scan on the fenced content
    # 2) direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 3) brace/bracket scan for the first balanced structure
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No JSON found in model reply:\n{text[:400]}")


def chat_json(
    messages: List[Dict[str, Any]],
    *,
    label: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retries: int = 1,
) -> Tuple[Any, CallMetrics]:
    """Call chat() and parse a JSON value. Retries once with a stern reminder."""
    msgs = list(messages)
    last_err: Optional[Exception] = None
    last_metrics = CallMetrics(label=label)
    for attempt in range(retries + 1):
        text, m = chat(msgs, label=label, temperature=temperature, max_tokens=max_tokens)
        last_metrics = m
        try:
            return extract_json(text), m
        except ValueError as e:
            last_err = e
            msgs = list(messages) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Return ONLY valid JSON, no prose, no markdown fences."},
            ]
    raise CerebrasError(f"Could not parse JSON after {retries + 1} tries: {last_err}")
