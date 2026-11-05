"""Minimal HTTP clients for querying chat models (stdlib only).

- "openai": any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM,
  Ollama, LM Studio, Together, OpenRouter, ...). Key from OPENAI_API_KEY.
- "anthropic": Anthropic Messages API. Key from ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class BackendError(RuntimeError):
    pass


def _post(url: str, headers: dict, body: dict, retries: int = 4, timeout: float = 120) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise BackendError(f"HTTP {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise BackendError(f"Could not reach {url}: {e.reason}") from e
    raise BackendError("unreachable")


class Backend:
    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatible(Backend):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 400, temperature: float | None = 0.0):
        self.model = model
        self.url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_tokens, self.temperature = max_tokens, temperature

    def complete(self, prompt: str) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"authorization": f"Bearer {self.key}"} if self.key else {}
        out = _post(self.url, headers, body)
        return out["choices"][0]["message"]["content"] or ""


class Anthropic(Backend):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 400, temperature: float | None = 0.0):
        self.model = model
        self.url = (base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.key:
            raise BackendError("Set ANTHROPIC_API_KEY to use the anthropic backend.")
        self.max_tokens, self.temperature = max_tokens, temperature

    def complete(self, prompt: str) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        out = _post(self.url, {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, body)
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")


def get_backend(name: str, **kwargs) -> Backend:
    if name == "openai":
        return OpenAICompatible(**kwargs)
    if name == "anthropic":
        return Anthropic(**kwargs)
    raise ValueError(f"unknown backend {name!r} (choose 'openai' or 'anthropic')")
