"""Interchangeable model backends. Each one maps (system, user) -> raw text.

`files` needs no network: an agent (or a person) reads chunks/*.prompt.md and
writes chunks/*.response.json. The HTTP backends use only the standard library
and read API keys from the environment; they have not been run against live
services from this repository yet.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Model APIs answer directly; a redirect could carry an API key elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, f"redirect to {newurl} refused", headers, fp
        )


_OPENER = urllib.request.build_opener(_NoRedirect)


def _check_secure(url: str, headers: dict) -> None:
    """Never send credentials in clear text, except to this machine."""
    parsed = urllib.parse.urlparse(url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    secret = any(k.lower() in {"authorization", "x-api-key"} for k in headers)
    if secret and parsed.scheme != "https" and not local:
        raise ValueError(f"refusing to send an API key over {parsed.scheme}: {url}")


def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    _check_secure(url, headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with _OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


RETRY_TEMPERATURE = 0.4


class Ollama:
    name = "ollama"

    def __init__(
        self, model: str, host: str = "http://127.0.0.1:11434", num_ctx: int = 16384
    ):
        self.model, self.host, self.num_ctx = model, host.rstrip("/"), num_ctx

    def complete(self, system: str, user: str, attempt: int = 0) -> str:
        # Greedy decoding is deterministic: a model that loops on a chunk (the
        # same identifier until num_predict) loops again on every retry. Retries
        # after an invalid answer therefore sample, with a fixed seed per attempt.
        sampling = (
            {"temperature": 0}
            if not attempt
            else {"temperature": RETRY_TEMPERATURE, "seed": attempt}
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "think": False,
            # num_predict stops a model that loops emitting whitespace in JSON mode
            "options": {**sampling, "num_ctx": self.num_ctx, "num_predict": 4096},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        content = _post(f"{self.host}/api/chat", payload, {}, 900)["message"]["content"]
        if not content.strip():
            # Some reasoning models return nothing under format=json/think=false:
            # ask once more with the model's own defaults.
            del payload["format"], payload["think"]
            content = _post(f"{self.host}/api/chat", payload, {}, 900)["message"][
                "content"
            ]
        if not content.strip():
            raise RuntimeError("the model returned an empty answer")
        return content


class OpenAICompatible:
    """OpenAI, or any server with /chat/completions (vLLM, LM Studio, Ollama /v1)."""

    name = "openai"

    def __init__(self, model: str, host: str = "https://api.openai.com/v1"):
        self.model, self.host = model, host.rstrip("/")

    def complete(self, system: str, user: str, attempt: int = 0) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        data = _post(
            f"{self.host}/chat/completions",
            {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            {"Authorization": f"Bearer {key}"} if key else {},
            600,
        )
        return data["choices"][0]["message"]["content"]


class Anthropic:
    name = "anthropic"

    def __init__(self, model: str, host: str = "https://api.anthropic.com"):
        self.model, self.host = model, host.rstrip("/")

    def complete(self, system: str, user: str, attempt: int = 0) -> str:
        data = _post(
            f"{self.host}/v1/messages",
            {
                "model": self.model,
                "max_tokens": 8000,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            {
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            600,
        )
        return "".join(block.get("text", "") for block in data["content"])


def make(name: str, model: str | None, host: str | None, num_ctx: int = 16384):
    if name == "files":
        return None
    if not model:
        raise ValueError(f"--model is required for backend {name}")
    kinds = {"ollama": Ollama, "openai": OpenAICompatible, "anthropic": Anthropic}
    backend = kinds[name](model, host) if host else kinds[name](model)
    if name == "ollama":
        backend.num_ctx = num_ctx
    return backend
