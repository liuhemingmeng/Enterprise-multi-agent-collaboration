"""Stage 6 evaluation script.

Generates the deterministic dataset (eval/tasks_sample.json), runs the
single-agent vs multi-agent comparison, and writes eval/report.json.

Usage:
    python -m scripts.run_eval
"""
from __future__ import annotations

import json
from pathlib import Path

from p2_agent.eval.dataset import build_evaluation_set, save_dataset
from p2_agent.eval.runner import run_comparison

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset_path = save_dataset(ROOT / "eval" / "tasks_sample.json")
    tasks = build_evaluation_set()
    print(f"dataset: {dataset_path} ({len(tasks)} tasks)")

    report = run_comparison(tasks=tasks, workdir=ROOT / "eval" / "runs")

    report_path = ROOT / "eval" / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report:  {report_path}")

    ma = report["multi_agent"]
    sa = report["single_agent"]
    print("\n=== headline metrics (100-task evaluation set) ===")
    print(f"auto_completion_rate : multi={ma['auto_completion_rate']}  single={sa['auto_completion_rate']}")
    print(f"safe_termination_rate: multi={ma['safe_termination_rate']}  single={sa['safe_termination_rate']}")
    print(f"mean_citation_coverage: multi={ma['mean_citation_coverage']}  single={sa['mean_citation_coverage']}  ({report['relative']['mean_citation_coverage']})")
    print(f"mean_cost           : multi={ma['mean_cost']}  single={sa['mean_cost']}  ({report['relative']['mean_cost']})")
    print(f"mean_duration_ms    : multi={ma['mean_duration_ms']}  single={sa['mean_duration_ms']}  ({report['relative']['mean_duration_ms']})")
    print(f"mean_evidence_count : multi={ma['mean_evidence_count']}  single={sa['mean_evidence_count']}  ({report['relative']['mean_evidence_count']})")


if __name__ == "__main__":
    main()
