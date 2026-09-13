"""Top-level flow: route one prompt to N models and return a validated report."""

from __future__ import annotations

from typing import Sequence

from . import engine, validation
from .models import ConsensusReport
from .providers import ModelProvider


def route(
    prompt: str,
    providers: Sequence[ModelProvider],
    *,
    sim_threshold: float = 0.6,
) -> ConsensusReport:
    """Route ``prompt`` to every provider and synthesize a trust report.

    Pipeline: call each model -> validate its structured output -> drop
    malformed claims -> run cross-model agreement detection -> reassemble and
    re-validate the final report (so the returned object is itself guaranteed
    to satisfy the report schema).

    Provider errors and validation failures degrade gracefully: the model is
    recorded in ``failed_models`` and the run continues.
    """
    models: list[str] = []
    failed: list[str] = []
    notes: list[str] = []
    entries: list[tuple[str, object]] = []

    for provider in providers:
        if provider is None:
            continue
        models.append(provider.name)

        try:
            raw = provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001 — a dead model must not kill the run
            failed.append(provider.name)
            notes.append(f"{provider.name}: provider error — {exc}")
            continue

        candidate, warnings = validation.validate_candidate(raw)
        if candidate is None:
            failed.append(provider.name)
            joined = "; ".join(warnings[:2]) or "no details"
            notes.append(f"{provider.name}: invalid output — {joined}")
            continue

        for warning in warnings:
            notes.append(f"{provider.name}: {warning}")

        if not candidate.claims:
            failed.append(provider.name)
            notes.append(f"{provider.name}: no valid claims after validation")
            continue

        for claim in candidate.claims:
            entries.append((provider.name, claim))

    assessments = engine.compute_assessments(entries, sim_threshold=sim_threshold)
    report = ConsensusReport(
        prompt=prompt,
        models=models,
        failed_models=failed,
        notes=notes,
        assessments=assessments,
    )

    # "Validate the structured output before returning it": round-trip the
    # report through its own schema so callers never receive a malformed one.
    return ConsensusReport.model_validate(report.model_dump())