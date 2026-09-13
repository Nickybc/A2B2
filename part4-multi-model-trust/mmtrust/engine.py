"""Deterministic claim comparison — the engine that finds agreement/disagreement.

Design
------
The trust question ("do these models agree?") is reduced to something we can
measure deterministically and reproduce in an eval:

1. Normalize text (lowercase, expand magnitude abbreviations like ``1.2bn``,
   split letter/digit boundaries so ``FY2024`` becomes ``FY 2024``).
2. Extract *values* (currency, percentages, scaled/plain numbers) and replace
   each span with a ``NUM`` placeholder.
3. The *subject* of a claim is the multiset of remaining tokens (minus
   stopwords). Two claims are "about the same fact" when their subjects have
   Jaccard similarity ``>= threshold``. The default threshold (0.6) is set so
   that same-fact-different-value claims (Jaccard ~1.0) match while
   adjacent-fact claims that share context tokens (e.g. "revenue" vs
   "net income", ~0.57) do not.
4. If they are about the same fact, compare their value multisets:
   equal values -> consensus, different values -> disagreement.
5. A claim with no cross-model match is single-source.

This is deliberately offline and deterministic so the eval is reproducible.
``analyze`` is the single seam where a production deployment could swap in
embedding-based similarity without touching the orchestration.

Known limitation (documented, not hidden): the engine compares *scalar values*.
It does not do open-ended entailment; direction words (``grew`` vs ``fell``)
are treated as part of the subject, not the value. Claims should be atomic
(one fact each).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .models import AgreementVerdict, Claim, ClaimAssessment

_ABBREV_SCALE = {
    "thousand": "thousand",
    "k": "thousand",
    "million": "million",
    "m": "million",
    "billion": "billion",
    "bn": "billion",
    "b": "billion",
    "trillion": "trillion",
    "t": "trillion",
}
_ABBREV_RE = re.compile(
    r"(?<=\d)\s*(bn|k|m|b|t|thousand|million|billion|trillion)\b", re.IGNORECASE
)

# Separate glued letter/digit boundaries: "FY2024" -> "FY 2024", "q3" -> "q 3".
_LD_SPLIT_RE = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")

_PERCENT_RE = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)\s*(?P<sign>%|percent|pct)", re.IGNORECASE
)

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_VALUE_RE = re.compile(
    rf"(?P<currency>\$|€|£|USD|usd)?\s*(?P<number>{_NUM})\s*(?P<scale>thousand|million|billion|trillion)?\b",
    re.IGNORECASE,
)

_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "at",
        "on",
        "for",
        "to",
        "by",
        "with",
        "and",
        "or",
        "its",
        "it",
        "s",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "as",
        "per",
        "from",
        "their",
        "over",
        "under",
        "reported",
    }
)


@dataclass(frozen=True)
class Value:
    """A single extracted scalar. ``text`` is the original matched span (for display)."""

    unit: str  # "money" | "percent" | "plain" | "scaled"
    magnitude: float
    text: str


def _to_float(raw: str) -> float:
    """Parse a regex-matched digit string; float() cannot fail here."""
    return float(raw.replace(",", ""))  # pi-lens-ignore: unchecked-throwing-call-python


def _value_key(value: Value) -> tuple[str, float]:
    """Equality key for a value (ignores display text, rounds float dust)."""
    return (value.unit, round(value.magnitude, 6))


def analyze(text: str) -> tuple[Counter[str], tuple[Value, ...]]:
    """Return (subject multiset, extracted values) for one claim's text."""
    work = text.lower()
    work = _ABBREV_RE.sub(lambda m: " " + _ABBREV_SCALE[m.group(1).lower()] + " ", work)
    work = _LD_SPLIT_RE.sub(" ", work)

    values: list[Value] = []

    def pct_repl(match: re.Match) -> str:
        values.append(
            Value("percent", _to_float(match.group("number")), match.group(0).strip())
        )
        return " NUM "

    work = _PERCENT_RE.sub(pct_repl, work)

    def val_repl(match: re.Match) -> str:
        currency = match.group("currency")
        scale = (match.group("scale") or "").lower()
        raw_number = match.group("number")
        number = _to_float(raw_number)
        # Bare 4-digit integers are years ("fiscal 2025", "ended 2024"): context,
        # not a fact value. Mask them without recording a value.
        if not currency and not scale and re.fullmatch(r"\d{4}", raw_number):
            return " "
        unit = "money" if currency else ("scaled" if scale else "plain")
        magnitude = number * _SCALES[scale] if scale else number
        values.append(Value(unit, magnitude, match.group(0).strip()))
        return " NUM "

    work = _VALUE_RE.sub(val_repl, work)

    tokens = re.findall(r"[a-z0-9]+", work)
    subject = Counter(t for t in tokens if t not in _STOPWORDS)
    return subject, tuple(values)


def extract_values(text: str) -> list[Value]:
    """Public convenience: the scalar values parsed out of a claim's text."""
    return list(analyze(text)[1])


def subject(text: str) -> Counter[str]:
    """Public convenience: the subject token multiset of a claim's text."""
    return analyze(text)[0]


def jaccard(a: Counter[str], b: Counter[str]) -> float:
    """Multiset Jaccard similarity in [0, 1]."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union


def values_compatible(a: Sequence[Value], b: Sequence[Value]) -> bool:
    """Two claims agree when their value multisets are equal OR one is a subset
    of the other. A model that merely *adds* a qualifying figure (e.g.
    "up 4% YoY") to the same core fact must not be treated as a conflict, so
    consensus is containment-based rather than exact-equality-based.
    """
    ca = Counter(_value_key(v) for v in a)
    cb = Counter(_value_key(v) for v in b)
    return ca == cb or ca <= cb or cb <= ca


def _format_values(values: Sequence[Value]) -> str:
    return ", ".join(v.text for v in values) or "no values"


def compute_assessments(
    entries: Sequence[tuple[str, Claim]], sim_threshold: float = 0.6
) -> list[ClaimAssessment]:
    """Compute a trust verdict for every claim across the model set.

    ``entries`` is a list of ``(model_name, claim)``. Consensus/disagreement is
    only ever computed *between different models*; claims from a single model
    are never compared against each other.
    """
    prepared = [(model, claim, *analyze(claim.text)) for model, claim in entries]
    assessments: list[ClaimAssessment] = []
    for i, (model, claim, subject, values) in enumerate(prepared):
        agreeing: list[str] = []
        conflicting: list[str] = []
        agreeing_texts: list[str] = []
        conflicting_texts: list[str] = []
        details: list[str] = []
        for j, (other_model, other_claim, other_subject, other_values) in enumerate(
            prepared
        ):
            if i == j or model == other_model:
                continue
            if jaccard(subject, other_subject) < sim_threshold:
                continue
            if values_compatible(values, other_values):
                agreeing.append(other_model)
                agreeing_texts.append(other_claim.text)
            else:
                conflicting.append(other_model)
                conflicting_texts.append(other_claim.text)
                details.append(
                    f"{other_model} says {_format_values(other_values)} "
                    f"(“{other_claim.text}”)"
                )

        if conflicting:
            verdict = AgreementVerdict.DISAGREEMENT
        elif agreeing:
            verdict = AgreementVerdict.CONSENSUS
        else:
            verdict = AgreementVerdict.SINGLE_SOURCE

        assessments.append(
            ClaimAssessment(
                claim=claim,
                source_model=model,
                verdict=verdict,
                agreeing_models=sorted(set(agreeing)),
                agreeing_claim_texts=agreeing_texts,
                conflicting_models=sorted(set(conflicting)),
                conflicting_claim_texts=conflicting_texts,
                conflict_detail=" ; ".join(details) or None,
                uncited=not claim.citations,
            )
        )
    return assessments
