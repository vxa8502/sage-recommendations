"""
Evaluation gate.

Reads the latest eval artifacts, prints a structured verdict, and exits 1
if the run is not reportable-green. Called as the final step of sage eval run
so that a sampled, boundary-failing, or metric-missing run cannot silently
pass as a canonical baseline.

Usage:
    python scripts/eval_gate.py
"""

import sys

from sage.cli.eval_status import build_eval_status
from sage.config import RESULTS_DIR

WIDTH = 60
SEP = "=" * WIDTH


def main() -> None:
    status = build_eval_status(results_dir=RESULTS_DIR)

    print(f"\n{SEP}")
    print("EVALUATION GATE")
    print(SEP)

    print(f"Execution:   {status['execution_status']}")
    print(f"Safety:      {status['safety_status']}")
    print(f"Reportable:  {status['reportable_status']}")

    metrics = status["metrics"]
    ndcg = metrics.get("retrieval_ndcg_at_10")
    faith_name = metrics.get("faithfulness_metric_name")
    faith_val = metrics.get("faithfulness_metric_value")
    faith_target = metrics.get("faithfulness_target")

    if ndcg is not None:
        print(f"NDCG@10:     {ndcg:.3f}")
    if faith_val is not None and faith_name is not None:
        target_str = (
            f"  (target {faith_target:.2f})" if faith_target is not None else ""
        )
        print(f"Faithfulness ({faith_name}): {faith_val:.3f}{target_str}")

    reasons = status.get("reportable_reasons") or []
    if reasons:
        print("\nReasons this run is WITHHELD:")
        for reason in reasons:
            print(f"  - {reason}")

    print(SEP)

    if not status["reportable_green"]:
        print(
            "GATE FAILED: run is not reportable-green. "
            "Fix the issues above before treating this as a canonical baseline.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("GATE PASSED: run is reportable-green.")


if __name__ == "__main__":
    main()
