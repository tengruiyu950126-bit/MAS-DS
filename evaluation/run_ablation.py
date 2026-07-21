"""Ablation runner for validation and rollback safety mechanisms.

This script intentionally applies a mixed plan: one useful repair plus one
unsafe transformation. It shows why the validation expert and rollback branch
matter for multi-agent preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from evaluation.corrupt_dataset import inject_corruption
from evaluation.metrics import calculate_detection_metrics, calculate_repair_metrics
from evaluation.run_experiment import make_demo_dataset
from models.cleaning_plan import CleaningPlan, CleaningStep
from tools.cleaning import execute_plan
from tools.validation import apply_rollback_policy, validate_preprocessing


AblationMode = Literal["with_validation_rollback", "without_validation"]


@dataclass(frozen=True)
class AblationRecord:
    mode: str
    seed: int
    detection_precision: float
    detection_recall: float
    detection_f1: float
    repair_success_rate: float
    data_preservation_rate: float
    validation_valid: bool
    rolled_back: bool
    missing_before: int
    missing_after_candidate: int
    missing_after_committed: int
    latency_seconds: float
    plan_steps: int
    operations: str
    validation_issues: str


def build_unsafe_stress_plan() -> CleaningPlan:
    """Return a plan with one helpful and one harmful action.

    The first step repairs missing numeric values in ``age``. The second step
    is deliberately unsafe: converting the categorical ``city`` column to
    numeric values turns valid city names into missing values. A validation
    expert should detect this and recommend rollback.
    """
    return CleaningPlan(
        steps=[
            CleaningStep(
                column="age",
                operation="fill_mean",
                reason="Repair missing numeric age values.",
                confidence=0.95,
            ),
            CleaningStep(
                column="city",
                operation="convert_numeric",
                reason="Unsafe stress-test action for validation ablation.",
                confidence=0.20,
            ),
        ]
    )


def _total_missing(dataframe: pd.DataFrame) -> int:
    return int(dataframe.isna().sum().sum())


def evaluate_once(
    dataframe: pd.DataFrame,
    *,
    mode: AblationMode,
    seed: int,
    fraction: float,
) -> AblationRecord:
    case = inject_corruption(
        dataframe,
        "missing_value",
        fraction=fraction,
        seed=seed,
        columns=["age"],
    )
    plan = build_unsafe_stress_plan()

    started = time.perf_counter()
    candidate, _ = execute_plan(case.corrupted, plan)
    validation = validate_preprocessing(case.corrupted, candidate, plan)

    if mode == "with_validation_rollback":
        committed, rolled_back = apply_rollback_policy(
            case.corrupted,
            candidate,
            validation,
        )
    elif mode == "without_validation":
        committed = candidate
        rolled_back = False
    else:
        raise ValueError(f"Unknown ablation mode: {mode!r}.")

    latency = time.perf_counter() - started
    detection = calculate_detection_metrics(case.records, plan)
    repair = calculate_repair_metrics(case.clean, committed, case.records)

    return AblationRecord(
        mode=mode,
        seed=seed,
        detection_precision=detection.precision,
        detection_recall=detection.recall,
        detection_f1=detection.f1,
        repair_success_rate=repair.repair_success_rate,
        data_preservation_rate=repair.data_preservation_rate,
        validation_valid=validation.valid,
        rolled_back=rolled_back,
        missing_before=validation.missing_before,
        missing_after_candidate=validation.missing_after,
        missing_after_committed=_total_missing(committed),
        latency_seconds=latency,
        plan_steps=len(plan.steps),
        operations=";".join(
            f"{step.operation}:{step.column or '*'}" for step in plan.steps
        ),
        validation_issues=";".join(issue.code for issue in validation.issues),
    )


def write_results(records: list[AblationRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["with_validation_rollback", "without_validation"],
        default=["with_validation_rollback", "without_validation"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--output", default="outputs/ablation_validation_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = make_demo_dataset()
    output_path = Path(args.output)
    records: list[AblationRecord] = []
    total = len(args.modes) * len(args.seeds)

    for mode in args.modes:
        for seed in args.seeds:
            print(
                f"[{len(records) + 1}/{total}] mode={mode} seed={seed}",
                flush=True,
            )
            records.append(
                evaluate_once(
                    dataset,
                    mode=mode,
                    seed=seed,
                    fraction=args.fraction,
                )
            )
            write_results(records, output_path)

    frame = pd.DataFrame(asdict(record) for record in records)
    summary = frame.groupby(["mode"], as_index=False).mean(numeric_only=True)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote {len(records)} runs to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
