"""Data models — the contract every model's output must satisfy.

``ModelCandidate`` is the structured JSON shape a single model is forced to
produce before its claims enter the trust pipeline. ``ConsensusReport`` is the
validated shape returned to the caller. Making these Pydantic models means the
schema is machine-checkable end to end.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A verifiable source backing one or more claims."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Verifiable http(s) URL of the source document")
    title: str = Field(..., min_length=1, description="Human-readable source title")
    quote: str | None = Field(default=None, description="Supporting excerpt from the source")


class Claim(BaseModel):
    """A single, atomic factual assertion made by one model."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="The assertion, one fact per claim")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Sources that back this claim; empty means 'uncited'",
    )


class ModelCandidate(BaseModel):
    """Structured output a single model produces for one prompt."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1, description="Identifier of the model/agent")
    claims: list[Claim] = Field(default_factory=list, description="Assertions made")


class AgreementVerdict(str, Enum):
    """How one model's claim relates to the other models' claims."""

    CONSENSUS = "consensus"           # >=2 models assert the same fact
    DISAGREEMENT = "disagreement"     # >=2 models assert the same fact with different values
    SINGLE_SOURCE = "single_source"   # only one model asserted this fact


class ClaimAssessment(BaseModel):
    """A claim plus the trust verdict computed across the model set."""

    claim: Claim
    source_model: str = Field(..., description="Which model made this claim")
    verdict: AgreementVerdict
    agreeing_models: list[str] = Field(default_factory=list)
    agreeing_claim_texts: list[str] = Field(
        default_factory=list, description="Exact texts of the agreeing claims (paired)"
    )
    conflicting_models: list[str] = Field(default_factory=list)
    conflicting_claim_texts: list[str] = Field(
        default_factory=list, description="Exact texts of the conflicting claims (paired)"
    )
    conflict_detail: str | None = Field(
        default=None, description="Human-readable diff of the conflicting values"
    )
    uncited: bool = Field(default=False, description="True when no verifiable citation was attached")


class ConsensusReport(BaseModel):
    """The validated, structured result returned to the caller."""

    prompt: str
    models: list[str] = Field(default_factory=list)
    failed_models: list[str] = Field(
        default_factory=list, description="Providers that errored or failed validation"
    )
    notes: list[str] = Field(default_factory=list, description="Validation and degradation notes")
    assessments: list[ClaimAssessment] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out = {v.value: 0 for v in AgreementVerdict}
        out["total"] = len(self.assessments)
        out["uncited"] = sum(1 for a in self.assessments if a.uncited)
        for a in self.assessments:
            out[a.verdict.value] += 1
        return out

    def to_markdown(self) -> str:
        """Render the report as a human-readable trace (used by the demo)."""
        c = self.counts
        lines: list[str] = [
            "# Consensus report",
            "",
            f"Prompt: _{self.prompt}_",
            f"Models: {', '.join(self.models) or '(none)'}",
            f"Failed models: {', '.join(self.failed_models) or '(none)'}",
            "",
            (
                f"{c['consensus']} consensus · {c['disagreement']} disagreement · "
                f"{c['single_source']} single-source · {c['uncited']} uncited · "
                f"{len(self.notes)} validation note(s)"
            ),
            "",
        ]

        by_verdict = {
            AgreementVerdict.DISAGREEMENT: "## Disagreement",
            AgreementVerdict.CONSENSUS: "## Consensus",
            AgreementVerdict.SINGLE_SOURCE: "## Single source",
        }
        order = [
            AgreementVerdict.DISAGREEMENT,
            AgreementVerdict.CONSENSUS,
            AgreementVerdict.SINGLE_SOURCE,
        ]
        for verdict in order:
            group = [a for a in self.assessments if a.verdict is verdict]
            if not group:
                continue
            lines.append(by_verdict[verdict])
            lines.append("")
            for a in group:
                status = "⚠️" if verdict is AgreementVerdict.DISAGREEMENT else "✅"
                if a.uncited:
                    status += " 🚫(uncited)"
                lines.append(f"- {status} {a.claim.text}  _(by {a.source_model})_")
                if a.conflict_detail:
                    lines.append(f"  - conflict: {a.conflict_detail}")
                if a.agreeing_models:
                    lines.append(f"  - agrees with: {', '.join(a.agreeing_models)}")
                for citation in a.claim.citations:
                    quote = f" — “{citation.quote}”" if citation.quote else ""
                    lines.append(f"  - source: [{citation.title}]({citation.url}){quote}")
            lines.append("")

        if self.notes:
            lines.append("## Validation notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines)