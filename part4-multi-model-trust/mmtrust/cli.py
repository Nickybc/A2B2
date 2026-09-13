"""Demo / walkthrough entry point (run as ``python -m mmtrust.cli``).

Subcommands
-----------
``demo``    Deterministic end-to-end trace over one finance-flavoured prompt:
            two mock models that agree on two facts, disagree on one figure,
            plus one uncited claim and two malformed claims that validation
            drops. Prints raw candidates, validation notes, and the final
            validated report — the written walkthrough for the README.
``schema``  Print the JSON Schema a model candidate must satisfy.
``live``    Route one real prompt through two or more real cloud models (any
            OpenAI-compatible endpoint, e.g. TokenRhythm). Reads
            ``TOKENRHYTHM_API_KEY`` / ``TOKENRHYTHM_BASE_URL`` from the
            environment; never hard-codes credentials. A model that times out
            or returns malformed JSON is recorded in ``failed_models``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import validation
from .orchestrator import route
from .providers import MockProvider, OpenAIChatProvider

_PROMPT = "Summarize Acme Corp's third-quarter financial results and its leadership."

_LIVE_PROMPT = (
    "What was Apple's total revenue for the fiscal quarter ended December 28, 2024, "
    "and who is Apple's CEO?"
)

_BASE_CANDIDATE = {
    "model": "base-model",
    "claims": [
        {
            "text": "Acme's third-quarter revenue was $1.2 billion.",
            "citations": [
                {
                    "url": "https://investor.acme.example.com/filings/10q-q3.pdf",
                    "title": "Acme Form 10-Q (Q3)",
                    "quote": "Net revenue for the quarter was $1.2 billion.",
                }
            ],
        },
        {
            "text": "Acme's third-quarter net income was $104 million.",
            "citations": [
                {
                    "url": "https://investor.acme.example.com/news/q3-results",
                    "title": "Acme Q3 results press release",
                    "quote": "Net income of $104 million.",
                }
            ],
        },
        {
            "text": "Jane Doe is Acme's chief executive officer.",
            "citations": [
                {
                    "url": "https://news.example.com/acme-jane-doe-ceo",
                    "title": "Jane Doe named Acme CEO",
                    "quote": "Ms. Doe, the company's chief executive, ...",
                }
            ],
        },
    ],
}

_AUDIT_CANDIDATE = {
    "model": "audit-model",
    "claims": [
        {
            "text": "Acme reported third-quarter revenue of $1.4 billion.",
            "citations": [
                {
                    "url": "https://investor.acme.example.com/events/q3-earnings-call",
                    "title": "Acme Q3 earnings call transcript",
                    "quote": "... bringing total revenue to $1.4 billion.",
                }
            ],
        },
        {
            "text": "For the third quarter, Acme's net income was $104 million.",
            "citations": [
                {
                    "url": "https://investor.acme.example.com/news/q3-results",
                    "title": "Acme Q3 results press release",
                    "quote": "Net income of $104 million.",
                }
            ],
        },
        {
            "text": "Jane Doe is the chief executive of Acme.",
            "citations": [
                {
                    "url": "https://news.example.com/acme-jane-doe-ceo",
                    "title": "Jane Doe named Acme CEO",
                }
            ],
        },
        # Deliberately uncited — the pipeline must flag it, not silently trust it.
        {"text": "John Smith is Acme's chief financial officer.", "citations": []},
        # Malformed: empty text -> dropped by validation.
        {"text": "", "citations": []},
        # Malformed: unknown field -> rejected (strict schema).
        {
            "text": "Acme plans to hire 500 engineers.",
            "citations": [],
            "hallucination_confidence": 0.99,
        },
    ],
}


def _demo() -> int:
    providers = [
        MockProvider("base-model", candidates={_PROMPT: _BASE_CANDIDATE}),
        MockProvider("audit-model", candidates={_PROMPT: _AUDIT_CANDIDATE}),
    ]
    print("## Raw model candidates")
    for name, candidate in (
        ("base-model", _BASE_CANDIDATE),
        ("audit-model", _AUDIT_CANDIDATE),
    ):
        print(f"\n### {name}\n```json\n{json.dumps(candidate, indent=2)}\n```")

    report = route(_PROMPT, providers)
    print("\n---\n")
    print(report.to_markdown())
    return 0


def _schema() -> int:
    print(json.dumps(validation.candidate_json_schema(), indent=2))
    return 0


def _live(args: argparse.Namespace) -> int:
    api_key = os.environ.get("TOKENRHYTHM_API_KEY")
    if not api_key:
        print(
            "error: set TOKENRHYTHM_API_KEY (and optionally TOKENRHYTHM_BASE_URL)",
            file=sys.stderr,
        )
        return 2
    base_url = os.environ.get("TOKENRHYTHM_BASE_URL", "https://tokenrhythm.studio/v1")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    providers = [
        OpenAIChatProvider(
            m, api_key=api_key, base_url=base_url, model=m, timeout=args.timeout
        )
        for m in models
    ]
    print(f"## Live route: {', '.join(models)}\n")
    print(route(args.prompt, providers).to_markdown())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmtrust.cli", description="Multi-model trust demo"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="deterministic mock demo (no keys, no network)")
    sub.add_parser("schema", help="print the candidate JSON Schema")

    live = sub.add_parser("live", help="route a real prompt through real cloud models")
    live.add_argument(
        "--models",
        default="glm-5.3-flash,deepseek-v4-pro-0813,kimi-k2.6,qwen3.8-max,seed-2.1-pro",
        help="comma-separated model ids (any number)",
    )
    live.add_argument("--prompt", default=_LIVE_PROMPT)
    live.add_argument("--timeout", type=float, default=120.0)

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _demo()
    if args.command == "schema":
        return _schema()
    if args.command == "live":
        return _live(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
