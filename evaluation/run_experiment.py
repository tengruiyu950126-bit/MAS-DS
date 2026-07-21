"""CLI experiment runner for rule, local-LLM, and hybrid planners."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from agents.cleaning_agent import RuleBasedCleaningAgent
from agents.hybrid_cleaning_agent import HybridCleaningAgent
from agents.local_llm_cleaning_agent import LocalLLMCleaningAgent
from evaluation.corrupt_dataset import CorruptionType, inject_corruption
from evaluation.datasets import (
    available_dataset_names,
    load_datasets,
    make_demo_dataset,
)
from evaluation.metrics import calculate_detection_metrics, calculate_repair_metrics
from models.policy import PreprocessingPolicy
from providers.ollama import OllamaClient
from tools.policy import load_policy_file
from workflow.graph import PreprocessingGraphOrchestrator


CORRUPTION_TYPES: list[CorruptionType] = [
    "missing_value",
    "duplicate_row",
    "numeric_type",
    "datetime_type",
]


@dataclass(frozen=True)
class ExperimentRecord:
    dataset: str
    method: str
    corruption: str
    seed: int
    detection_precision: float
    detection_recall: float
    detection_f1: float
    repair_success_rate: float
    data_preservation_rate: float
    invalid_plan: bool
    rolled_back: bool
    latency_seconds: float
    plan_steps: int
    operations: str
    validation_issues: str
    error_message: str


def build_planner(
    method: str,
    model: str,
    timeout_seconds: float = 180,
    policy: PreprocessingPolicy | None = None,
):
    if method == "rule":
        return RuleBasedCleaningAgent(policy=policy)
    llm = LocalLLMCleaningAgent(
        OllamaClient(model=model, timeout_seconds=timeout_seconds),
        policy=policy,
    )
    if method == "llm":
        return llm
    if method == "hybrid":
        return HybridCleaningAgent(llm, policy=policy)
    raise ValueError(f"Unknown method: {method!r}.")


def evaluate_once(
    dataframe: pd.DataFrame,
    *,
    method: str,
    corruption: CorruptionType,
    seed: int,
    fraction: float,
    model: str,
    dataset_name: str = "custom",
    timeout_seconds: float = 180,
    policy: PreprocessingPolicy | None = None,
) -> ExperimentRecord:
    case = inject_corruption(
        dataframe,
        corruption,
        fraction=fraction,
        seed=seed,
    )
    planner = build_planner(method, model, timeout_seconds, policy=policy)
    orchestrator = PreprocessingGraphOrchestrator(
        cleaning_agent=planner,
        policy=policy,
    )
    started = time.perf_counter()
    invalid_plan = False
    rolled_back = False
    validation_issues = ""
    error_message = ""
    try:
        proposal = orchestrator.propose(case.corrupted)
        outcome = orchestrator.execute_approved(case.corrupted, proposal.plan)
        repaired = outcome.dataframe
        rolled_back = outcome.rolled_back
        validation_issues = ";".join(
            issue.code for issue in outcome.validation.issues
        )
        plan = proposal.plan
    except Exception as exc:
        invalid_plan = True
        error_message = f"{type(exc).__name__}: {exc}"
        repaired = case.corrupted
        from models.cleaning_plan import CleaningPlan

        plan = CleaningPlan(steps=[])
    latency = time.perf_counter() - started
    detection = calculate_detection_metrics(case.records, plan)
    repair = calculate_repair_metrics(case.clean, repaired, case.records)
    return ExperimentRecord(
        dataset=dataset_name,
        method=method,
        corruption=corruption,
        seed=seed,
        detection_precision=detection.precision,
        detection_recall=detection.recall,
        detection_f1=detection.f1,
        repair_success_rate=repair.repair_success_rate,
        data_preservation_rate=repair.data_preservation_rate,
        invalid_plan=invalid_plan,
        rolled_back=rolled_back,
        latency_seconds=latency,
        plan_steps=len(plan.steps),
        operations=";".join(
            f"{step.operation}:{step.column or '*'}" for step in plan.steps
        ),
        validation_issues=validation_issues,
        error_message=error_message,
    )


def write_results(records: list[ExperimentRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["rule", "llm", "hybrid"],
        default=["rule"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=available_dataset_names(),
        default=["demo"],
    )
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional path to a PreprocessingPolicy JSON file.",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTION_TYPES,
        default=CORRUPTION_TYPES,
    )
    parser.add_argument("--output", default="outputs/evaluation_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = load_datasets(args.datasets, rows=args.rows)
    policy = load_policy_file(args.policy) if args.policy else None
    output_path = Path(args.output)
    records: list[ExperimentRecord] = []
    total = (
        len(datasets) * len(args.methods) * len(args.corruptions) * len(args.seeds)
    )
    for dataset in datasets:
        for method in args.methods:
            for corruption in args.corruptions:
                for seed in args.seeds:
                    print(
                        f"[{len(records) + 1}/{total}] "
                        f"dataset={dataset.name} method={method} "
                        f"corruption={corruption} seed={seed}",
                        flush=True,
                    )
                    records.append(
                        evaluate_once(
                            dataset.dataframe,
                            dataset_name=dataset.name,
                            method=method,
                            corruption=corruption,
                            seed=seed,
                            fraction=args.fraction,
                            model=args.model,
                            timeout_seconds=args.timeout,
                            policy=policy,
                        )
                    )
                    # Checkpoint after every run so an interrupted local-model
                    # experiment keeps all completed work.
                    write_results(records, output_path)
    frame = pd.DataFrame(asdict(record) for record in records)
    summary = frame.groupby(
        ["dataset", "method", "corruption"],
        as_index=False,
    ).mean(numeric_only=True)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {len(records)} runs to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
