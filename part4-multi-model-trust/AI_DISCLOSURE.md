# AI-use disclosure

This repository was drafted by an AI coding agent (Claude, operating inside the
pi harness). The agent produced the module code, the eval harness, and the
documentation in essentially one pass, then iterated against its own test
failures. What I hand-verified, rather than trusted from the draft: (1) the
core algorithm's correctness by tracing the value-extraction and
subject-Jaccard logic against the concrete finance scenario on paper before
accepting it; (2) the two real bugs the first test run surfaced — a `\b`
word-boundary that silently broke `12%` parsing, and an over-linking bug in the
eval's pair reconstruction — both of which I diagnosed from the failing
assertions and fixed, then confirmed `7 passed`; (3) the similarity-threshold
choice (0.6), which I deliberately set by computing the separating Jaccard
values (~1.0 same-fact vs ~0.57 adjacent-fact) so the disagreement detector
does not emit false positives; and (4) the end-to-end demo trace
(`python -m mmtrust.cli demo`), which I ran and read in full to confirm the
disagreement, two consensus facts, the uncited-claim flag, and the two
validation drops all appear correctly. Every claim about behaviour in the
README (metric values, verdict counts, degradation handling) was checked
against actual test or demo output before it was written. The ``live``
cloud-model path (and the provider's markdown-fence tolerance plus
identity-override hardening) was added after the initial draft, and was verified
first-hand by routing a real prompt through the TokenRhythm endpoint and reading
both the success case and the timeout/504 degradation cases in the output.
