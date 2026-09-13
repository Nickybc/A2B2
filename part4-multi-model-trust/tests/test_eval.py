"""Eval harness for multi-model trust orchestration.

What this measures (and what it will catch if someone regresses it)
---------------------------------------------------------------------
1. **Disagreement detection** — precision / recall / F1 over a labelled set of
   cross-model claim pairs. If the value parser breaks, the similarity
   threshold drifts, the stopword list changes, or the "only compare across
   models" guard is removed, F1 collapses and the headline test fails.
2. **Agreement (consensus) detection** — same metric over labelled agreeing
   pairs.
3. **Citation correctness** — every claim must surface exactly the source from
   the oracle, and an input with no citation must be flagged ``uncited``.
4. **Deterministic unit guards** — value parsing scales, strict-schema
   validation, graceful degradation on a failing provider, and mutual
   self-reference (same-model claims must never be paired).

The fixtures are deterministic (no network, no credentials), so the eval is
reproducible and fast. It does not merely assert that the code "runs": every
test pins a behaviour whose breakage would change a trust decision.
"""

from __future__ import annotations

from mmtrust import route
from mmtrust.engine import compute_assessments, extract_values, jaccard
from mmtrust.models import AgreementVerdict, Claim, ConsensusReport
from mmtrust.providers import FailingProvider, MockProvider
from mmtrust.validation import candidate_json_schema, validate_candidate

# ---------------------------------------------------------------------------
# Deterministic scenario + oracle
# ---------------------------------------------------------------------------

PROMPT = "Summarize Acme Corp's third-quarter financial results and its leadership."

REV_A = "Acme's third-quarter revenue was $1.2 billion."
REV_B = "Acme reported third-quarter revenue of $1.4 billion."
NI_A = "Acme's third-quarter net income was $104 million."
NI_B = "For the third quarter, Acme's net income was $104 million."
CEO_A = "Jane Doe is Acme's chief executive officer."
CEO_B = "Jane Doe is the chief executive of Acme."
CFO_B = "John Smith is Acme's chief financial officer."

BASE = {
    "model": "base-model",
    "claims": [
        {
            "text": REV_A,
            "citations": [
                {
                    "url": "https://investor.acme.example.com/filings/10q-q3.pdf",
                    "title": "Acme 10-Q",
                }
            ],
        },
        {
            "text": NI_A,
            "citations": [
                {
                    "url": "https://investor.acme.example.com/news/q3-results",
                    "title": "Acme press release",
                }
            ],
        },
        {
            "text": CEO_A,
            "citations": [
                {"url": "https://news.example.com/acme-jane-doe-ceo", "title": "NYT"}
            ],
        },
    ],
}
AUDIT = {
    "model": "audit-model",
    "claims": [
        {
            "text": REV_B,
            "citations": [
                {
                    "url": "https://investor.acme.example.com/events/q3-earnings-call",
                    "title": "Acme call transcript",
                }
            ],
        },
        {
            "text": NI_B,
            "citations": [
                {
                    "url": "https://investor.acme.example.com/news/q3-results",
                    "title": "Acme press release",
                }
            ],
        },
        {
            "text": CEO_B,
            "citations": [
                {"url": "https://news.example.com/acme-jane-doe-ceo", "title": "NYT"}
            ],
        },
        {"text": CFO_B, "citations": []},
    ],
}

ORACLE_DISAGREE: set[frozenset[str]] = {frozenset((REV_A, REV_B))}
ORACLE_AGREE: set[frozenset[str]] = {frozenset((NI_A, NI_B)), frozenset((CEO_A, CEO_B))}
ORACLE_CITATIONS = {
    REV_A: "https://investor.acme.example.com/filings/10q-q3.pdf",
    REV_B: "https://investor.acme.example.com/events/q3-earnings-call",
    NI_A: "https://investor.acme.example.com/news/q3-results",
    NI_B: "https://investor.acme.example.com/news/q3-results",
    CEO_A: "https://news.example.com/acme-jane-doe-ceo",
    CEO_B: "https://news.example.com/acme-jane-doe-ceo",
}


def _pairs(report: ConsensusReport, attr: str) -> set[frozenset[str]]:
    """Reconstruct the exact cross-model claim pairs the engine linked.

    Uses the report's stored *claim texts* (not model names), so a claim that
    merely shares a model with another is never falsely paired.
    """
    return {
        frozenset((a.claim.text, other))
        for a in report.assessments
        for other in getattr(a, attr)
    }


def _f1(true: set[frozenset[str]], reported: set[frozenset[str]]) -> dict:
    tp = len(true & reported)
    precision = tp / len(reported) if reported else 1.0
    recall = tp / len(true) if true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": len(reported) - tp,
        "fn": len(true) - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_scenario() -> tuple[ConsensusReport, dict]:
    providers = [
        MockProvider("base-model", candidates={PROMPT: BASE}),
        MockProvider("audit-model", candidates={PROMPT: AUDIT}),
    ]
    report = route(PROMPT, providers)

    by_text = {a.claim.text: a for a in report.assessments}
    disagree_metrics = _f1(ORACLE_DISAGREE, _pairs(report, "conflicting_claim_texts"))
    agree_metrics = _f1(ORACLE_AGREE, _pairs(report, "agreeing_claim_texts"))

    cited = 0
    missing = 0
    for text, expected_url in ORACLE_CITATIONS.items():
        assessment = by_text[text]
        urls = [c.url for c in assessment.claim.citations]
        if expected_url in urls:
            cited += 1
        else:
            missing += 1
    citation_accuracy = cited / (cited + missing) if (cited + missing) else 1.0

    cfo = by_text[CFO_B]
    metrics = {
        "disagreement": disagree_metrics,
        "agreement": agree_metrics,
        "citation_accuracy": citation_accuracy,
        "cfo_flagged_uncited": cfo.uncited,
        "cfo_verdict": cfo.verdict.value,
        "counts": report.counts,
    }
    return report, metrics


# ---------------------------------------------------------------------------
# The headline regression test
# ---------------------------------------------------------------------------


def test_disagreement_and_citation_metrics() -> None:
    report, m = run_scenario()

    # A real regression here means the trust decision itself changed:
    assert m["disagreement"]["f1"] == 1.0, m["disagreement"]
    assert m["disagreement"]["precision"] == 1.0, "false-positive disagreement"
    assert m["agreement"]["f1"] == 1.0, m["agreement"]
    assert m["agreement"]["recall"] == 1.0, "missed consensus"

    assert m["citation_accuracy"] == 1.0, "a claim lost or mismatched its source"
    assert m["cfo_flagged_uncited"] is True, "uncited claim was not flagged"
    assert m["cfo_verdict"] == "single_source"

    # Exactly one model should own each disagreement, consensus is 2v2, etc.
    assert m["counts"]["disagreement"] == 2  # both sides of the one disputed fact
    assert m["counts"]["consensus"] == 4  # 2 facts x 2 models
    assert m["counts"]["single_source"] == 1  # the uncited CFO claim
    assert m["counts"]["total"] == 7
    assert report.failed_models == []


# ---------------------------------------------------------------------------
# Deterministic unit guards (each pins a behaviour whose breakage = regression)
# ---------------------------------------------------------------------------


def test_value_parser_scales_and_units() -> None:
    def first(text: str):
        return extract_values(text)[0]

    assert (first("$1.2 billion").unit, first("$1.2 billion").magnitude) == (
        "money",
        1.2e9,
    )
    assert (first("$104 million").unit, first("$104 million").magnitude) == (
        "money",
        104e6,
    )
    assert (first("$1.2bn").unit, first("$1.2bn").magnitude) == (
        "money",
        1.2e9,
    )  # abbrev
    assert (first("12%").unit, first("12%").magnitude) == ("percent", 12.0)
    assert (first("1,200").unit, first("1,200").magnitude) == ("plain", 1200.0)
    assert first("$1.2 billion").magnitude != first("$1.4 billion").magnitude


def test_year_context_is_not_a_value() -> None:
    # 4-digit years are context, not fact values — "fiscal 2025" must not
    # introduce a phantom value or a spurious disagreement.
    assert extract_values("fiscal 2025") == []
    assert [v.text for v in extract_values("ended December 28, 2024")] == ["28"]

    a = Claim(text="Apple reported total revenue of $124.3 billion for fiscal 2025.")
    b = Claim(text="Apple's total revenue was $124.3 billion.")
    assessments = compute_assessments([("m1", a), ("m2", b)])
    assert all(x.verdict is AgreementVerdict.CONSENSUS for x in assessments)


def test_qualifying_figure_does_not_cause_disagreement() -> None:
    # A model that adds "up 4%" to the same core figure must agree, not conflict.
    a = Claim(text="Apple's quarterly revenue was $124.3 billion, up 4 percent.")
    b = Claim(text="Apple's quarterly revenue was $124.3 billion.")
    assessments = compute_assessments([("m1", a), ("m2", b)])
    assert all(x.verdict is AgreementVerdict.CONSENSUS for x in assessments)


def test_same_model_claims_are_never_paired() -> None:
    claims = [
        ("m1", Claim(text="Revenue was $1 billion.")),
        ("m1", Claim(text="Revenue was $2 billion.")),
    ]
    assessments = compute_assessments(claims)
    assert all(a.verdict is AgreementVerdict.SINGLE_SOURCE for a in assessments)
    assert all(a.conflicting_models == [] for a in assessments)


def test_validation_is_strict_and_claim_tolerant() -> None:
    # Envelope: unknown field rejected.
    candidate, _ = validate_candidate({"model": "m", "claims": [], "bogus": 1})
    assert candidate is None

    # Per-claim tolerance: one malformed claim dropped, the rest survive.
    candidate, notes = validate_candidate(
        {
            "model": "m",
            "claims": [
                {
                    "text": "Fine claim",
                    "citations": [{"url": "https://x.example.com/a", "title": "t"}],
                },
                {"text": "", "citations": []},  # empty text -> dropped
                {
                    "text": "Extra",
                    "citations": [],
                    "confidence": 0.9,
                },  # unknown field -> dropped
            ],
        }
    )
    assert candidate is not None
    assert len(candidate.claims) == 1
    assert any("claim[1] dropped" in n for n in notes)
    assert any("claim[2] dropped" in n for n in notes)

    # Unverifiable (non-http) citation is stripped, leaving the claim uncited.
    candidate, _ = validate_candidate(
        {
            "model": "m",
            "claims": [
                {"text": "A claim", "citations": [{"url": "ftp://file", "title": "t"}]}
            ],
        }
    )
    assert candidate is not None
    assert candidate.claims[0].citations == []
    assert candidate.claims[0].text == "A claim"


def test_failing_provider_degrades_gracefully() -> None:
    report = route(
        PROMPT,
        [MockProvider("good", candidates={PROMPT: BASE}), FailingProvider("down")],
    )
    assert report.failed_models == ["down"]
    assert "good" in report.models
    assert any("down: provider error" in n for n in report.notes)
    assert len(report.assessments) == 3  # good model's claims all survive


def test_report_self_validates_and_schema_emits() -> None:
    report, _ = run_scenario()
    # The final object round-trips its own schema ("validate before returning").
    assert ConsensusReport.model_validate(report.model_dump()) == report
    schema = candidate_json_schema()
    assert schema["properties"].keys() >= {"model", "claims"}
    assert schema.get("additionalProperties") is False


def test_jaccard_is_symmetric_and_bounded() -> None:
    from mmtrust.engine import analyze

    a = analyze(REV_A)[0]
    b = analyze(REV_B)[0]
    c = analyze(NI_B)[0]
    assert jaccard(a, b) == jaccard(b, a)
    assert 0.0 <= jaccard(a, b) <= 1.0
    # Same-fact (revenue) beats adjacent-fact (net income) under the default gate.
    assert jaccard(a, b) >= 0.6 > jaccard(a, c)
