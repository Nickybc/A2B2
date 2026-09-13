"""Model providers — the pluggable seam that routes the prompt.

A provider's only job is ``generate(prompt) -> dict``, returning the raw JSON
candidate a model produced. The orchestrator validates that JSON; a provider
that raises is treated as a failed model (graceful degradation) and the run
continues with the remaining models.

``MockProvider`` and ``FailingProvider`` make the pipeline deterministic and
testable with no network and no credentials. ``OpenAIChatProvider`` shows how
a real model wires in (stdlib ``urllib``, explicit timeout); it is optional and
not required to run the demo or the eval.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a candidate."""


class ModelProvider(Protocol):
    @property
    def name(self) -> str:
        """Human-readable identifier of the model/agent (read-only)."""
        ...

    def generate(self, prompt: str) -> dict[str, Any]:
        """Return the raw candidate JSON as a dict-shaped object."""
        ...


class MockProvider:
    """Deterministic provider backed by a prompt -> candidate mapping."""

    def __init__(
        self,
        name: str,
        candidates: dict[str, dict[str, Any]] | None = None,
        *,
        default: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._candidates = candidates or {}
        self._default = default

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> dict[str, Any]:
        if prompt in self._candidates:
            return self._candidates[prompt]
        if self._default is not None:
            return self._default
        raise ProviderError(f"{self._name}: no candidate for this prompt")


class FailingProvider:
    """Always raises — used to assert graceful degradation in the eval."""

    def __init__(self, name: str, message: str = "simulated outage") -> None:
        self._name = name
        self._message = message

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> dict[str, Any]:
        raise ProviderError(f"{self._name}: {self._message}")


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse a model's reply into a JSON object, tolerating markdown fences.

    A model that returns malformed JSON raises ``ProviderError``; the
    orchestrator catches it and degrades gracefully (marks the model failed).
    """
    text = content.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:  # pi-lens-ignore: no-boolean-in-except
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise ProviderError("response was not JSON") from None
    if not isinstance(obj, dict):
        raise ProviderError("response JSON was not an object")
    return obj


class OpenAIChatProvider:
    """Optional adapter for an OpenAI-compatible chat-completions endpoint.

    Instructs the model to answer with JSON matching the candidate schema, then
    parses that JSON. An explicit timeout and typed failure path keep a stuck or
    malformed model from taking down the run — the orchestrator catches the
    resulting ``ProviderError`` and degrades gracefully.
    """

    def __init__(
        self,
        name: str,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self._name = name
        self._model = model
        self._base_url = base_url.rstrip("/")
        parsed = urllib.parse.urlparse(self._base_url)
        if parsed.scheme not in ("http", "https"):
            raise ProviderError(
                f"{self._name}: base_url scheme must be http(s), got {parsed.scheme!r}"
            )
        self._timeout = timeout
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> dict[str, Any]:
        if not self._api_key:
            raise ProviderError(f"{self._name}: OPENAI_API_KEY is not set")
        system = (
            "You answer with JSON only. Reply with a JSON object of the form "
            '{"model": "<your-name>", "claims": [{"text": "...", "citations": '
            '[{"url": "https://...", "title": "...", "quote": "..."}]}]}. '
            "One fact per claim; every factual claim must carry a citation."
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(  # pi-lens-ignore: S310 - scheme whitelist asserted in __init__
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self._timeout
        ) as response:  # pi-lens-ignore: S310 - scheme whitelist asserted in __init__
            raw_body = response.read().decode("utf-8")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self._name}: endpoint returned invalid JSON: {exc}"
            ) from exc
        content = payload["choices"][0]["message"]["content"]
        candidate = _parse_json_object(content)
        # The model's self-reported "model" field is untrusted; assert ours.
        candidate["model"] = self._name
        return candidate
