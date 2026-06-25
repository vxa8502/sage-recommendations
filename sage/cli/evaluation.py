from __future__ import annotations

import argparse
from datetime import datetime

from sage.config import RESULTS_DIR
from sage.services.corpus_alignment import (
    CorpusAlignmentError,
    assert_corpus_alignment,
)

from .evaluation_support import boundary as evaluation_boundary
from .evaluation_support import readiness as evaluation_readiness
from .stage_experiments.handoff import ensure_calibration_handoff_ready
from .shared import (
    PROJECT_ROOT,
    Step,
    capture_output,
    cli_display_command,
    ensure_env,
    python_command,
    run_command,
    run_steps,
)
from .script_command import script_command
from .eval_status import build_eval_status


def _sample_limit_label(limit: int | None) -> str:
    return "all" if limit is None else str(limit)


def _raise_eval_incomplete(eval_status: dict[str, object]) -> None:
    raw_reasons = eval_status.get("execution_reasons")
    reasons = raw_reasons if isinstance(raw_reasons, list) else []
    detail_lines = [
        f"- {reason}" for reason in reasons[:5] if isinstance(reason, str) and reason
    ]
    detail = "\n".join(detail_lines) if detail_lines else "- no details provided"
    raise SystemExit(
        "ERROR: Evaluation did not finish with a complete current-cycle artifact set.\n"
        f"{detail}\n"
        "The runtime cannot be treated as execution-complete until all required "
        "latest artifacts are refreshed by this run."
    )


def _print_eval_status_surfaces(eval_status: dict[str, object]) -> None:
    print("Evaluation status surfaces:")
    print(f"  execution_complete: {eval_status['execution_complete']}")
    print(f"  safety_green:      {eval_status['safety_green']}")
    print(f"  reportable_green:  {eval_status['reportable_green']}")

    reportable_reasons = eval_status.get("reportable_reasons") or []
    if not eval_status.get("reportable_green") and isinstance(
        reportable_reasons, list
    ):
        for reason in reportable_reasons[:3]:
            if isinstance(reason, str) and reason:
                print(f"  reportable_note:  {reason}")


def _run_full_eval_workflow(
    samples: int | None,
    ragas_samples: int | None,
    url: str,
    requests: int,
    *,
    enforce_gate: bool,
) -> None:
    run_started_at = datetime.now().astimezone()
    build_dataset = python_command("scripts/build_natural_eval_dataset.py")
    explanation_basic = python_command("scripts/explanation.py", "--section", "basic")
    explanation_gate = python_command("scripts/explanation.py", "--section", "gate")
    explanation_verify = python_command("scripts/explanation.py", "--section", "verify")
    explanation_cold = python_command("scripts/explanation.py", "--section", "cold")

    (PROJECT_ROOT / "assets").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    steps = [
        Step(
            title="[1/8] EDA (production data)",
            commands=(python_command("scripts/eda.py"),),
        ),
        Step(
            title="[2/8] Retrieval metrics + ablations",
            commands=(
                build_dataset,
                python_command(
                    "scripts/evaluation.py",
                    "--dataset",
                    "eval_natural_queries.json",
                    "--section",
                    "all",
                ),
            ),
        ),
        Step(
            title="[3/8] Baseline comparison",
            commands=(
                python_command(
                    "scripts/evaluation.py",
                    "--dataset",
                    "eval_natural_queries.json",
                    "--section",
                    "primary",
                    "--baselines",
                ),
            ),
        ),
        Step(
            title="[4/8] Explanation tests",
            commands=(
                explanation_basic,
                explanation_gate,
                explanation_verify,
                explanation_cold,
            ),
        ),
        Step(
            title="[5/8] Faithfulness + refusal-aware metrics",
            commands=(
                python_command(
                    "scripts/faithfulness.py",
                    "--samples",
                    _sample_limit_label(samples),
                    "--ragas",
                    "--ragas-samples",
                    _sample_limit_label(ragas_samples),
                ),
            ),
        ),
        Step(
            title="[6/8] Boundary behavior",
            commands=(
                python_command(
                    "scripts/evaluate_boundary_behavior.py",
                    "--artifact-scope",
                    "canonical",
                ),
            ),
        ),
        Step(
            title="[7/8] All sanity checks",
            commands=(
                python_command(
                    "scripts/sanity_checks.py",
                    "--section",
                    "all",
                ),
            ),
        ),
        Step(
            title="[8/8] Load test",
            commands=(
                python_command(
                    "scripts/load_test.py",
                    "--url",
                    url,
                    "--requests",
                    str(requests),
                    "--save",
                ),
            ),
        ),
    ]

    run_steps("FULL REPRODUCIBLE EVALUATION", steps, [])
    status = build_eval_status(
        results_dir=RESULTS_DIR,
        run_started_at=run_started_at,
    )
    if not status["execution_complete"]:
        _raise_eval_incomplete(status)
    evaluation_boundary.ensure_boundary_guardrail_passed()
    print()
    print(capture_output(python_command("scripts/summary.py")))
    print()
    _print_eval_status_surfaces(status)
    print()

    if enforce_gate:
        # Exit non-zero if the run is sampled, below metric thresholds, or
        # missing artifacts — prevents a dev-lane result from silently
        # becoming the canonical baseline.
        run_command(python_command("scripts/eval_gate.py"))
        print("=== CANONICAL EVALUATION COMPLETE ===")
    else:
        print(
            "DEV LANE: results are sampled and NOT a canonical baseline."
        )
        print(
            f"  faithfulness samples: {_sample_limit_label(samples)}"
        )
        print(f"  RAGAS samples:        {_sample_limit_label(ragas_samples)}")
        print(
            "  Run 'sage eval run' for the reportable canonical evaluation."
        )
        print("=== DEV LANE EVALUATION COMPLETE ===")

    print()
    print("Results saved to: data/eval_results/")
    print("  - eval_natural_queries_latest.json  (NDCG, Hit@K, MRR)")
    print("  - faithfulness_latest.json          (HHEM, RAGAS)")
    print("  - adjusted_faithfulness_latest.json (refusal-aware pass rate)")
    print(
        "  - boundary_behavior_latest.json"
        "     (refusal/clarify guardrail benchmark)"
    )
    print(
        "  - load_test_latest.json"
        "             (steady-state latency headline + drill-down)"
    )
    if enforce_gate:
        print()
        print("NEXT STEPS:")
        print(f"  1. {cli_display_command('eval', 'summary')}")


def command_eval_boundary(args: argparse.Namespace) -> None:
    evaluation_readiness.ensure_boundary_eval_query_bank_ready(
        subset_tag=args.subset_tag,
        path=args.query_bank_path,
    )
    ensure_env()

    command = (
        script_command("scripts/evaluate_boundary_behavior.py")
        .option("--query-bank-path", args.query_bank_path)
        .option("--subset-tag", args.subset_tag)
        .option("--top-k", args.top_k)
        .option("--aggregation", args.aggregation)
        .option("--max-evidence", args.max_evidence)
        .option(
            "--artifact-scope", "dev" if args.query_limit is not None else "canonical"
        )
        .optional("--query-limit", args.query_limit)
        .optional("--min-rating", args.min_rating)
        .to_list()
    )

    run_command(command)


def _run_eval(
    *,
    samples: int | None,
    ragas_samples: int | None,
    url: str,
    requests: int,
    enforce_gate: bool,
) -> None:
    evaluation_readiness.ensure_eval_query_bank_ready()
    evaluation_readiness.ensure_faithfulness_cases_ready()
    ensure_calibration_handoff_ready()
    ensure_env()
    try:
        assert_corpus_alignment()
    except CorpusAlignmentError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    _run_full_eval_workflow(
        samples,
        ragas_samples,
        url,
        requests,
        enforce_gate=enforce_gate,
    )


def command_eval(args: argparse.Namespace) -> None:
    _run_eval(
        samples=getattr(args, "samples", None),
        ragas_samples=getattr(args, "ragas_samples", None),
        url=args.url,
        requests=args.requests,
        enforce_gate=True,
    )


def command_eval_dev(args: argparse.Namespace) -> None:
    """Run the sampled evaluation dev lane with CLI-managed defaults."""
    _run_eval(
        samples=args.samples,
        ragas_samples=args.ragas_samples,
        url=args.url,
        requests=args.requests,
        enforce_gate=False,
    )


def command_eval_summary(_args: argparse.Namespace) -> None:
    run_command(python_command("scripts/summary.py"))
