# mmtrust — multi-model trust orchestration

Route one prompt to two or more models, detect where they **agree** and
**disagree**, attach a **verifiable source** to every claim, and **validate the
structured output** before returning it.

This repo is the *orchestration layer*: `mmtrust` treats each model as a
plugin that must emit a strict JSON candidate, then does the trust work —
value extraction, cross-model agreement/disagreement detection, citation
attachment, and schema validation — deterministically and offline, so the whole
thing is reproducible and testable without API keys.

---

## What problem this solves

When you ask N models the same question and just concatenate their answers, you
get no signal about *reliability*. `mmtrust` turns that into a structured
verdict per claim:

| verdict | meaning |
| --- | --- |
| `consensus` | ≥2 models assert the same fact (same subject, same value) |
| `disagreement` | ≥2 models assert the same fact with *different values* |
| `single_source` | only one model asserted the fact |
| `uncited` | a claim has no verifiable source (flagged, never silently trusted) |

A model that errors, times out, or returns malformed JSON is recorded in
`failed_models` and the run **degrades gracefully** — the other models' claims
still get a full verdict.

---

## Quick start

Requires Python ≥ 3.10 and `pydantic ≥ 2.0`.

```bash
python3 -m venv .venv && source .venv/bin/activate   # or use system Python
pip install "pydantic>=2.0" "pytest>=8.0"

# Run the deterministic end-to-end demo (no keys, no network):
python -m mmtrust.cli demo

# Print the JSON Schema a model candidate must satisfy:
python -m mmtrust.cli schema

# Route one prompt through ANY NUMBER of REAL cloud models (OpenAI-compatible, needs a key):
TOKENRHYTHM_API_KEY=... python -m mmtrust.cli live --models glm-5.3-flash,deepseek-v4-pro-0813,kimi-k2.6,qwen3.8-max,seed-2.1-pro

# Run the eval harness (7 tests):
python -m pytest tests/test_eval.py -q
```

A ready-to-read trace of the demo is committed at
[`examples/demo_output.txt`](examples/demo_output.txt).

---

## How to use it in code

```python
from mmtrust import route
from mmtrust.providers import MockProvider, OpenAIChatProvider

report = route(
    prompt="Summarize Acme Corp's Q3 results and leadership.",
    providers=[
        OpenAIChatProvider("gpt-4o-mini"),          # real model (needs OPENAI_API_KEY)
        MockProvider("audit-model", candidates={...}),  # deterministic fixture
    ],
)

print(report.counts)          # {'disagreement': 2, 'consensus': 4, ...}
for a in report.assessments:
    print(a.verdict.value, "-", a.claim.text, "| by", a.source_model)
    for c in a.claim.citations:
        print("   ↳", c.url)
```

`route()` is the single public entry point. A provider is anything with a
`.name` and a `.generate(prompt) -> dict` method (see
[`mmtrust/providers.py`](mmtrust/providers.py) for the `ModelProvider`
protocol, a deterministic `MockProvider`, a `FailingProvider` for testing
degradation, and an optional `OpenAIChatProvider` adapter that uses only the
standard library + a timeout).

---

## Architecture

Eight source files, one responsibility each:

```
mmtrust/
├── __init__.py      public API (route, models)
├── models.py        Pydantic contract: Citation, Claim, ModelCandidate,
│                    ClaimAssessment, ConsensusReport (+ Markdown rendering)
├── providers.py     ModelProvider protocol + Mock / Failing / OpenAI adapters
├── engine.py        deterministic claim comparison: value extraction,
│                    subject Jaccard similarity, agreement/disagreement
├── validation.py    strict-envelope / tolerant-claim schema validation
├── orchestrator.py  route(): call → validate → compare → re-validate → return
└── cli.py           `demo` and `schema` subcommands
tests/
└── test_eval.py     the eval harness (metrics + deterministic unit guards)
```

### 1. Every model emits structured JSON — validated before use

A model's output must be a `ModelCandidate`:

```json
{
  "model": "base-model",
  "claims": [
    {"text": "Acme's Q3 revenue was $1.2 billion.",
     "citations": [{"url": "https://investor.acme.example.com/filings/10q-q3.pdf",
                    "title": "Acme Form 10-Q (Q3)", "quote": "Net revenue ... $1.2 billion."}]}
  ]
}
```

[`validation.validate_candidate`](mmtrust/validation.py) enforces this with a
*strict envelope* (unknown fields and wrong types are rejected) but *per-claim
tolerance*: a malformed claim is dropped and reported while the model's valid
claims survive. A model left with zero valid claims is marked failed.

### 2. Deterministic agreement/disagreement detection

[`engine.py`](mmtrust/engine.py) reduces "do these models agree?" to something
measurable:

1. Normalize text — lowercase, expand magnitude abbreviations (`$1.2bn` →
   `$1.2 billion`), split `FY2024` → `FY 2024`.
2. Extract **values** (currency, percentages, scaled/plain numbers) and replace
   each span with a `NUM` placeholder.
3. The **subject** is the multiset of remaining tokens (minus stopwords). Two
   claims are "about the same fact" when subject Jaccard similarity ≥ `0.6`.
4. Compare value multisets: equal → consensus; different → disagreement.
5. No cross-model match → single-source; no citations → flagged `uncited`.

The default threshold (`0.6`) is deliberately chosen: same-fact-different-value
claims land at Jaccard ~1.0, while adjacent facts that share context tokens
(e.g. "revenue" vs "net income", ~0.57) do not match. `analyze()` is the seam
where a production deployment would swap in embedding similarity.

> **Documented limitation:** the engine compares *scalar values*. It does not
> do open-ended entailment, and direction words (`grew` vs `fell`) are part of
> the subject, not the value. Claims should be atomic (one fact each). This is
> stated in the module docstring and covered by the eval's scope.

### 3. The output is validated before it is returned

`route()` reassembles the `ConsensusReport` and round-trips it through
`ConsensusReport.model_validate(...)` — callers never receive a report that
fails its own schema.

---

## The eval harness — what a regression looks like

[`tests/test_eval.py`](tests/test_eval.py) is a real eval, not a smoke test. It
labels a deterministic two-model scenario with an oracle of which claim pairs
must disagree, which must agree, and which source each claim must carry, then
asserts:

- **Disagreement detection**: precision / recall / F1 **= 1.0** (a false
  positive or a miss fails the test).
- **Agreement detection**: F1 **= 1.0**.
- **Citation correctness**: every oracle-cited claim surfaces exactly the right
  URL; the uncited input is flagged `uncited`.
- **Deterministic unit guards** that break on specific regressions:
  - value parsing scales (`$1.2bn` → `1.2e9`, `12%` → `percent 12.0`, …);
  - a claim from the same model is *never* paired with itself/its own model;
  - strict schema (extra field → reject) vs tolerant claims (bad claim dropped,
    good ones survive);
  - a failing provider degrades gracefully into `failed_models`.

Concrete regressions this catches: changing the `billion` scale constant, a
typo in the currency/percent regex, drifting the similarity threshold, removing
a stopword, or removing the "cross-model only" guard — any one of these flips a
trust decision and the headline F1 assertion fails.

Run it:

```bash
python -m pytest tests/test_eval.py -q
# 7 passed
```

---

## Five-minute demo / walkthrough

The committed trace at [`examples/demo_output.txt`](examples/demo_output.txt)
runs one real input end to end:

**Prompt:** *"Summarize Acme Corp's third-quarter financial results and its
leadership."*

Two mock models answer. The pipeline then produces:

1. **One disagreement** — `base-model` says revenue was **$1.2 billion**
   (cited to the 10-Q); `audit-model` says **$1.4 billion** (cited to the
   earnings-call transcript). Both claims are surfaced *with their sources and
   the conflicting value*.
2. **Two consensus facts** — net income ($104M) and the CEO (Jane Doe),
   each cited and marked as agreed by both models.
3. **One single-source, uncited claim** — "John Smith is CFO" is flagged
   `🚫 (uncited)` so it can't be silently trusted.
4. **Two claims dropped by validation** — an empty-text claim and a claim with
   an unknown field (`hallucination_confidence`) are rejected and reported.

That demonstrates every moving part: routing to multiple models, disagreement
detection, citation attachment, and structured-output validation.

Reproduce it locally with `python -m mmtrust.cli demo` (no keys, no network).

---

## Live demo with real cloud models

`live` runs the *same* pipeline through **any number** of real models on any
OpenAI-compatible endpoint (the repo ships no keys; you supply one via
environment variables). A committed trace at
[`examples/live_output.txt`](examples/live_output.txt) shows a five-model vote
(`glm-5.3-flash`, `deepseek-v4-pro-0813`, `kimi-k2.6`, `qwen3.8-max`,
`seed-2.1-pro`) converging on **$124.3 billion** for Apple's fiscal Q1 2025
revenue and on Tim Cook as CEO — detected as consensus with real apple.com
citations. Pass `--models a,b,c,...` to widen the vote to any N models; the
engine keeps every cross-model pair, so a model either joins the consensus or
splits into a flagable minority. (Live endpoints are nondeterministic and
occasionally return 504/timeout; a failing model is recorded in `failed_models`
while the rest are still fully judged.)

Real models are also the honest stress test for reliability: while testing,
`qwen3.8-max`/`qwen3.8-flash` timed out and `glm-5.2` returned HTTP 504. Each
was recorded in `failed_models` while the surviving model's claims were still
fully judged — Graceful degradation, exercised against the real world.

```bash
export TOKENRHYTHM_API_KEY=...
python -m mmtrust.cli live --models glm-5.3-flash,deepseek-v4-pro-0813,kimi-k2.6,qwen3.8-max,seed-2.1-pro
python -m mmtrust.cli live --help   # --models, --prompt, --timeout
```

---

## Limitations & production notes

- The engine is value-centric; open-ended semantic entailment and opinion
  agreement are out of scope (see the docstring).
- `OpenAIChatProvider` is a thin, stdlib-only adapter demonstrating real wiring;
  the demo/eval use deterministic mocks so they never need credentials or a
  network.
- URL liveness is validated at the *shape* level (http/https scheme). Verifying
  that a URL actually resolves would be a natural next step behind a
  `check_citations` flag.

## License

MIT — see [LICENSE](LICENSE).
