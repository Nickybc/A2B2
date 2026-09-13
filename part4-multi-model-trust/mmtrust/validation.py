"""Structured-output validation with graceful degradation.

A model's candidate JSON is validated here *before* it enters the trust
pipeline. The envelope is strict (unknown fields and wrong types are rejected),
but individual claims are tolerated: a malformed claim is dropped and reported
while the model's remaining valid claims survive. A model that ends up with
zero valid claims is treated as failed by the orchestrator.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, BaseModel

from .models import Citation, Claim, ModelCandidate


class _Envelope(BaseModel):
    """Strict shape check for the candidate wrapper (per-claim tolerance below)."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1)
    claims: list[Any] = Field(default_factory=list)


_claim_adapter = TypeAdapter(Claim)


def candidate_json_schema() -> dict[str, Any]:
    """The public JSON Schema a model is expected to satisfy."""
    return ModelCandidate.model_json_schema()


def _is_verifiable_url(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


def validate_candidate(payload: Any) -> tuple[ModelCandidate | None, list[str]]:
    """Validate one model's candidate output.

    Returns ``(validated_candidate_or_None, list_of_notes)``. A ``None``
    candidate means the envelope itself is malformed (wrong type, missing
    ``model``, unknown top-level fields, or ``claims`` is not a list).
    """
    if not isinstance(payload, dict):
        return None, ["candidate is not a JSON object"]

    try:
        envelope = _Envelope.model_validate(payload)
    except ValidationError as exc:
        notes = [f"envelope invalid: {e['loc']} {e['msg']}" for e in exc.errors()[:3]]
        return None, notes

    notes: list[str] = []
    claims: list[Claim] = []
    for idx, item in enumerate(envelope.claims):
        try:
            claim = _claim_adapter.validate_python(item)
        except ValidationError as exc:
            notes.append(f"claim[{idx}] dropped: {_summarize(exc)}")
            continue

        kept: list[Citation] = []
        for citation in claim.citations:
            if not _is_verifiable_url(citation.url):
                notes.append(f"claim[{idx}] dropped unverifiable citation ({citation.url!r})")
                continue
            kept.append(citation)
        if len(kept) != len(claim.citations):
            claim = claim.model_copy(update={"citations": kept})

        claims.append(claim)

    return ModelCandidate(model=envelope.model, claims=claims), notes


def _summarize(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:2])