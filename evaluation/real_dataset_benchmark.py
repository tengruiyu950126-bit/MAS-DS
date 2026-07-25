"""Benchmarks on real public datasets bundled with scikit-learn.

This runner avoids network access. It uses classic public datasets that ship
with scikit-learn when that package is available in the local environment.
The benchmark injects controlled corruptions, runs MAS-DS, and reports
detection/repair/preservation metrics.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from agents.cleaning_agent import RuleBasedCleaningAgent
from evaluation.corrupt_dataset import CorruptionType, inject_corruption
from evaluation.metrics import calculate_detection_metrics, calculate_repair_metrics
from models.cleaning_plan import CleaningPlan
from tools.chunked import execute_chunked_csv
from tools.validation import validate_preprocessing
from agents.orchestrator import PreprocessingOrchestrator


PublicDatasetName = Literal["iris", "wine", "breast_cancer", "diabetes"]
RunnerName = Literal["in_memory_rule", "chunked_rule"]

PUBLIC_DATASETS: list[PublicDatasetName] = [
    "iris",
    "wine",
    "breast_cancer",
    "diabetes",
]
RUNNERS: list[RunnerName] = ["in_memory_rule", "chunked_rule"]
CORRUPTIONS: list[CorruptionType] = [
    "missing_value",
    "duplicate_row",
    "numeric_type",
]


@dataclass(frozen=True)
class PublicDataset:
    name: str
    source: str
    dataframe: pd.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class RealDatasetBenchmarkRecord:
    dataset: str
    source: str
    runner: str
    corruption: str
    seed: int
    rows: int
    columns: int
    corruption_records: int
    detection_precision: float
    detection_recall: float
    detection_f1: float
    repair_success_rate: float
    data_preservation_rate: float
    validation_valid: bool
    rolled_back: bool
    latency_seconds: float
    plan_steps: int
    operations: str
    validation_issues: str
    error_message: str


def sklearn_available() -> bool:
    try:
        import sklearn.datasets  # noqa: F401
    except ImportError:
        return False
    return True


def load_public_dataset(name: PublicDatasetName, *, rows: int | None = None) -> PublicDataset:
    """Load a public dataset bundled with scikit-learn."""
    try:
        from sklearn import datasets
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is not installed, so bundled public datasets are unavailable."
        ) from exc

    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes,
    }
    if name not in loaders:
        raise ValueError(f"Unknown public dataset: {name!r}.")

    bunch = loaders[name](as_frame=True)
    frame = bunch.frame.copy(deep=True)
    feature_columns = [str(column) for column in bunch.data.columns]

    # For classification datasets, add readable class names as categorical
    # context while keeping the numeric target from sklearn's original frame.
    if hasattr(bunch, "target_names") and "target" in frame.columns:
        target_names = list(map(str, bunch.target_names))
        frame["target_name"] = [
            target_names[int(value)] if pd.notna(value) else None
            for value in frame["target"].tolist()
        ]

    # Stable, explicit row identifier for auditing. It is protected by policy
    # because the name ends in `_id`.
    frame.insert(0, "public_row_id", [f"{name}-{index:05d}" for index in range(len(frame))])

    if rows is not None:
        if rows <= 0:
            raise ValueError("rows must be positive.")
        frame = frame.head(rows).copy(deep=True)

    return PublicDataset(
        name=name,
        source="scikit-learn bundled public dataset",
        dataframe=frame,
        feature_columns=feature_columns,
    )


def run_real_dataset_benchmark(
    *,
    datasets: list[PublicDatasetName] | None = None,
    runners: list[RunnerName] | None = None,
    corruptions: list[CorruptionType] | None = None,
    seeds: list[int] | None = None,
    rows: int | None = None,
    fraction: float = 0.05,
    chunk_size: int = 100_000,
    planning_sample_rows: int = 100_000,
    output_dir: str | Path = "outputs/real_dataset_benchmark",
) -> list[RealDatasetBenchmarkRecord]:
    """Run benchmark cases and write detailed outputs to ``output_dir``."""
    selected_datasets = datasets or ["iris", "wine", "breast_cancer"]
    selected_runners = runners or ["in_memory_rule", "chunked_rule"]
    selected_corruptions = corruptions or CORRUPTIONS
    selected_seeds = seeds or [0, 1, 2]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[RealDatasetBenchmarkRecord] = []
    for dataset_name in selected_datasets:
        dataset = load_public_dataset(dataset_name, rows=rows)
        for runner in selected_runners:
            for corruption in selected_corruptions:
                for seed in selected_seeds:
                    records.append(
                        _evaluate_public_dataset_once(
                            dataset,
                            runner=runner,
                            corruption=corruption,
                            seed=seed,
                            fraction=fraction,
                            chunk_size=chunk_size,
                            planning_sample_rows=planning_sample_rows,
                            output_dir=output_path,
                        )
                    )
                    _write_records(
                        records,
                        output_path / "real_dataset_benchmark_results.csv",
                    )

    _write_summary(records, output_path / "real_dataset_benchmark_summary.csv")
    _write_markdown_report(records, output_path / "real_dataset_benchmark_report.md")
    return records


def _evaluate_public_dataset_once(
    dataset: PublicDataset,
    *,
    runner: RunnerName,
    corruption: CorruptionType,
    seed: int,
    fraction: float,
    chunk_size: int,
    planning_sample_rows: int,
    output_dir: Path,
) -> RealDatasetBenchmarkRecord:
    started = time.perf_counter()
    error_message = ""
    validation_valid = False
    rolled_back = False
    validation_issues = ""
    plan = CleaningPlan(steps=[])
    repaired = pd.DataFrame()

    try:
        case = inject_corruption(
            dataset.dataframe,
            corruption,
            fraction=fraction,
            seed=seed,
            columns=dataset.feature_columns if corruption != "duplicate_row" else None,
        )

        if runner == "in_memory_rule":
            planner = RuleBasedCleaningAgent()
            orchestrator = PreprocessingOrchestrator(cleaning_agent=planner)
            proposal = orchestrator.propose(case.corrupted)
            outcome = orchestrator.execute_approved(case.corrupted, proposal.plan)
            plan = proposal.plan
            repaired = outcome.dataframe
            validation_valid = outcome.validation.valid
            rolled_back = outcome.rolled_back
            validation_issues = ";".join(issue.code for issue in outcome.validation.issues)
        elif runner == "chunked_rule":
            case_dir = output_dir / "cases"
            case_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{dataset.name}_{corruption}_seed{seed}"
            dirty_csv = case_dir / f"{prefix}_dirty.csv"
            cleaned_csv = case_dir / f"{prefix}_chunked_cleaned.csv"
            summary_csv = case_dir / f"{prefix}_chunked_summary.csv"
            plan_json = case_dir / f"{prefix}_chunked_plan.json"
            case.corrupted.to_csv(dirty_csv, index=False)
            execute_chunked_csv(
                dirty_csv,
                cleaned_csv,
                chunk_size=chunk_size,
                planning_sample_rows=planning_sample_rows,
                summary_csv=summary_csv,
                plan_json=plan_json,
                read_csv_kwargs={"dtype": "object"},
            )
            repaired = pd.read_csv(cleaned_csv)
            plan = CleaningPlan.model_validate_json(plan_json.read_text(encoding="utf-8"))
            validation = validate_preprocessing(case.corrupted, repaired, plan)
            validation_valid = validation.valid
            rolled_back = validation.recommend_rollback
            validation_issues = ";".join(issue.code for issue in validation.issues)
        else:
            raise ValueError(f"Unknown runner: {runner!r}.")

        detection = calculate_detection_metrics(case.records, plan)
        repair = calculate_repair_metrics(case.clean, repaired, case.records)
        corruption_records = len(case.records)
    except Exception as exc:
        detection = None
        repair = None
        corruption_records = 0
        error_message = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started
    return RealDatasetBenchmarkRecord(
        dataset=dataset.name,
        source=dataset.source,
        runner=runner,
        corruption=corruption,
        seed=seed,
        rows=len(dataset.dataframe),
        columns=len(dataset.dataframe.columns),
        corruption_records=corruption_records,
        detection_precision=round(detection.precision, 6) if detection else 0.0,
        detection_recall=round(detection.recall, 6) if detection else 0.0,
        detection_f1=round(detection.f1, 6) if detection else 0.0,
        repair_success_rate=round(repair.repair_success_rate, 6) if repair else 0.0,
        data_preservation_rate=round(repair.data_preservation_rate, 6) if repair else 0.0,
        validation_valid=validation_valid,
        rolled_back=rolled_back,
        latency_seconds=round(latency, 4),
        plan_steps=len(plan.steps),
        operations=";".join(
            f"{step.operation}:{step.column or '*'}" for step in plan.steps
        ),
        validation_issues=validation_issues,
        error_message=error_message,
    )


def _write_records(records: list[RealDatasetBenchmarkRecord], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_summary(records: list[RealDatasetBenchmarkRecord], path: Path) -> None:
    if not records:
        return
    frame = pd.DataFrame(asdict(record) for record in records)
    summary = (
        frame.groupby(["dataset", "runner", "corruption"], as_index=False)
        .agg(
            runs=("seed", "count"),
            avg_detection_f1=("detection_f1", "mean"),
            avg_repair_success_rate=("repair_success_rate", "mean"),
            avg_data_preservation_rate=("data_preservation_rate", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
            rollback_rate=("rolled_back", "mean"),
            invalid_or_error_rate=("error_message", lambda values: (values != "").mean()),
        )
        .round(6)
    )
    summary.to_csv(path, index=False)


def _write_markdown_report(records: list[RealDatasetBenchmarkRecord], path: Path) -> None:
    if not records:
        return
    frame = pd.DataFrame(asdict(record) for record in records)
    summary = (
        frame.groupby(["dataset", "runner"], as_index=False)
        .agg(
            runs=("seed", "count"),
            avg_detection_f1=("detection_f1", "mean"),
            avg_repair_success_rate=("repair_success_rate", "mean"),
            avg_data_preservation_rate=("data_preservation_rate", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
            rollback_rate=("rolled_back", "mean"),
        )
        .round(4)
    )

    lines = [
        "# Real Public Dataset Benchmark",
        "",
        "This benchmark uses public datasets bundled with scikit-learn. It does not download data and does not call paid APIs.",
        "",
        "## Summary by dataset and runner",
        "",
        _markdown_table(summary),
        "",
        "## Notes",
        "",
        "- Corruptions are controlled and reproducible.",
        "- Public datasets are clean references; MAS-DS is evaluated after injecting missing-value, duplicate-row, and numeric-type corruptions.",
        "- `in_memory_rule` uses the standard MAS-DS graph.",
        "- `chunked_rule` uses the chunked CSV preprocessing path.",
        "- Duplicate-row detection F1 can look low because the plan contains one dataset-level `drop_duplicates` action, while the metric counts many row-level duplicate records.",
        "- Repair success and data preservation should be read together with detection F1.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[column]).replace("|", "\\|") for column in columns)
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAS-DS on public datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=PUBLIC_DATASETS,
        default=["iris", "wine", "breast_cancer"],
    )
    parser.add_argument(
        "--runners",
        nargs="+",
        choices=RUNNERS,
        default=RUNNERS,
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS,
        default=CORRUPTIONS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--planning-sample-rows", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        default="outputs/real_dataset_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = run_real_dataset_benchmark(
        datasets=args.datasets,
        runners=args.runners,
        corruptions=args.corruptions,
        seeds=args.seeds,
        rows=args.rows,
        fraction=args.fraction,
        chunk_size=args.chunk_size,
        planning_sample_rows=args.planning_sample_rows,
        output_dir=args.output_dir,
    )
    output_dir = Path(args.output_dir)
    summary = pd.read_csv(output_dir / "real_dataset_benchmark_summary.csv")
    print(summary.to_string(index=False))
    print(f"\nWrote {len(records)} runs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
