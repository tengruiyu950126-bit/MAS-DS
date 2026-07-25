"""Large synthetic dataset smoke/stress test for MAS-DS.

This script is intentionally local and free. It can:

- estimate memory for a requested shape without materializing the full dataset;
- run the core MAS-DS pipeline on large in-memory pandas DataFrames;
- optionally run the full audit/report path for moderate data sizes.

No paid API, cloud model, or network call is used.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.policy import PreprocessingPolicy
from tools.diff import diff_summary_frame, plan_diff_frame
from tools.reporting import build_cleaning_report
from agents.orchestrator import PreprocessingOrchestrator


CORE_COLUMNS = 8


@dataclass(frozen=True)
class CapacityEstimate:
    rows_requested: int
    columns_requested: int
    duplicate_fraction: float
    sample_rows: int
    sample_rows_after_duplicates: int
    sample_columns: int
    sample_memory_mb: float
    estimated_dataframe_memory_mb: float
    estimated_working_memory_mb: float
    memory_guard_mb: float
    full_run_recommended: bool
    recommendation: str


@dataclass(frozen=True)
class LargeDatasetSmokeResult:
    mode: str
    rows_input: int
    rows_after_duplicates: int
    rows_output: int
    columns: int
    memory_mb: float
    plan_steps: int
    operations: str
    validation_valid: bool
    rolled_back: bool
    validation_issues: str
    audit_rows: int
    generation_seconds: float
    propose_seconds: float
    execute_validate_seconds: float
    diff_seconds: float
    report_seconds: float
    total_seconds: float


def _currency_values(indexes: np.ndarray) -> np.ndarray:
    values = 50 + (indexes % 5000) * 1.37 + ((indexes * 17) % 100) / 100
    values = values.astype("float64")
    values[indexes % 997 == 0] = 100_000 + indexes[indexes % 997 == 0]
    text = np.char.add("$", np.char.mod("%.2f", values))
    return text.astype(object)


def make_large_dirty_dataset(
    *,
    rows: int,
    seed: int = 42,
    duplicate_fraction: float = 0.02,
    columns: int = CORE_COLUMNS,
) -> pd.DataFrame:
    """Create a reproducible dirty dataset with wide-table support."""
    if columns < CORE_COLUMNS:
        raise ValueError(f"--columns must be at least {CORE_COLUMNS}.")

    indexes = np.arange(rows, dtype=np.int64)

    cities = np.array(["Singapore", "Bangkok", "Jakarta", "Kuala Lumpur", "Manila"], dtype=object)
    city = cities[indexes % len(cities)].astype(object)
    city[indexes % 17 == 0] = [f" {value} " for value in city[indexes % 17 == 0]]
    city[indexes % 19 == 0] = [str(value).upper() for value in city[indexes % 19 == 0]]
    if rows > 123:
        city[123] = "Singaproe"
    city[indexes % 97 == 0] = None

    segments = np.array(["Retail", "Enterprise", "SMB", "Government"], dtype=object)
    segment = segments[indexes % len(segments)].astype(object)
    segment[indexes % 23 == 0] = [str(value).lower() for value in segment[indexes % 23 == 0]]
    segment[indexes % 29 == 0] = [f" {value} " for value in segment[indexes % 29 == 0]]
    segment[indexes % 211 == 0] = None

    amount_text = _currency_values(indexes)
    amount_text[indexes % 41 == 0] = None

    age = (18 + (indexes % 58)).astype("float32")
    age[indexes % 31 == 0] = np.nan

    score = (40 + (indexes % 61)).astype("float32")
    score[indexes % 37 == 0] = np.nan
    score[indexes % 997 == 0] = 999.0

    data: dict[str, object] = {
        "customer_id": np.char.add("C-", np.char.zfill(indexes.astype(str), 7)).astype(object),
        "age": age,
        "score": score,
        "city": city,
        "segment": segment,
        "amount_text": amount_text,
        "signup_date": np.char.add(
            np.char.add("2024-", np.char.zfill(((indexes % 12) + 1).astype(str), 2)),
            np.char.add("-", np.char.zfill(((indexes % 28) + 1).astype(str), 2)),
        ).astype(object),
        "channel": np.array(["Web", "Store", "Partner"], dtype=object)[indexes % 3],
    }

    # Extra columns are intentionally protected-like names ending in `_id`.
    # The planner will profile them, but will not waste time proposing cleanups.
    extra_columns = columns - CORE_COLUMNS
    base_feature = (indexes % 10_000).astype("int32")
    for extra_index in range(extra_columns):
        name = f"wide_feature_{extra_index + 1:03d}_id"
        kind = extra_index % 3
        if kind == 0:
            data[name] = (base_feature + extra_index).astype("int32")
        elif kind == 1:
            data[name] = ((indexes + extra_index) % 127).astype("int16")
        else:
            data[name] = ((indexes + extra_index) % 2 == 0)

    dataframe = pd.DataFrame(data)

    duplicate_count = max(1, int(rows * duplicate_fraction))
    rng = np.random.default_rng(seed)
    duplicate_indices = rng.integers(0, rows, size=duplicate_count)
    duplicates = dataframe.iloc[duplicate_indices].copy(deep=True)
    return pd.concat([dataframe, duplicates], ignore_index=True)


def estimate_capacity(
    *,
    rows: int,
    columns: int,
    seed: int,
    duplicate_fraction: float,
    sample_rows: int,
    memory_guard_mb: float,
) -> CapacityEstimate:
    """Estimate memory required for a requested shape from a smaller sample."""
    sample_rows = min(sample_rows, rows)
    sample = make_large_dirty_dataset(
        rows=sample_rows,
        seed=seed,
        duplicate_fraction=duplicate_fraction,
        columns=columns,
    )
    sample_memory_mb = float(sample.memory_usage(deep=True).sum()) / 1_000_000
    duplicate_multiplier = 1 + duplicate_fraction
    estimated_dataframe_memory_mb = sample_memory_mb * (rows / sample_rows)
    estimated_dataframe_memory_mb *= duplicate_multiplier / (len(sample) / sample_rows)
    # The current pandas pipeline creates multiple temporary copies during
    # execution and validation. This multiplier is intentionally conservative.
    estimated_working_memory_mb = estimated_dataframe_memory_mb * 4.5
    recommended = estimated_working_memory_mb <= memory_guard_mb
    return CapacityEstimate(
        rows_requested=rows,
        columns_requested=columns,
        duplicate_fraction=duplicate_fraction,
        sample_rows=sample_rows,
        sample_rows_after_duplicates=len(sample),
        sample_columns=len(sample.columns),
        sample_memory_mb=round(sample_memory_mb, 3),
        estimated_dataframe_memory_mb=round(estimated_dataframe_memory_mb, 3),
        estimated_working_memory_mb=round(estimated_working_memory_mb, 3),
        memory_guard_mb=memory_guard_mb,
        full_run_recommended=recommended,
        recommendation=(
            "Core in-memory run is within the configured guard."
            if recommended
            else (
                "Avoid full in-memory execution at this shape unless the machine "
                "has enough RAM. Use a smaller full run or implement chunked/out-of-core processing."
            )
        ),
    )


def run_large_dataset_smoke(
    *,
    rows: int,
    seed: int,
    output_dir: Path,
    columns: int = CORE_COLUMNS,
    duplicate_fraction: float = 0.02,
    mode: str = "full",
    max_memory_mb: float | None = None,
    audit_sample_rows: int = 20_000,
) -> LargeDatasetSmokeResult:
    """Run the MAS-DS pipeline on a large synthetic dataset."""
    started = time.perf_counter()

    generation_started = time.perf_counter()
    dataframe = make_large_dirty_dataset(
        rows=rows,
        seed=seed,
        duplicate_fraction=duplicate_fraction,
        columns=columns,
    )
    generation_seconds = time.perf_counter() - generation_started
    memory_mb = round(float(dataframe.memory_usage(deep=True).sum()) / 1_000_000, 3)
    if max_memory_mb is not None and memory_mb > max_memory_mb:
        raise SystemExit(
            f"Generated dataframe uses {memory_mb} MB, above guard {max_memory_mb} MB."
        )

    policy = PreprocessingPolicy()
    orchestrator = PreprocessingOrchestrator(policy=policy)

    propose_started = time.perf_counter()
    proposal = orchestrator.propose(dataframe)
    propose_seconds = time.perf_counter() - propose_started

    execute_started = time.perf_counter()
    outcome = orchestrator.execute_approved(dataframe, proposal.plan)
    execute_validate_seconds = time.perf_counter() - execute_started

    diff_seconds = 0.0
    report_seconds = 0.0
    audit_rows = 0
    diff = pd.DataFrame()
    diff_summary = pd.DataFrame(columns=["operation", "column", "change_type", "changes"])
    report = ""

    if mode == "full":
        diff_started = time.perf_counter()
        diff = plan_diff_frame(dataframe, proposal.plan)
        diff_summary = diff_summary_frame(diff)
        diff_seconds = time.perf_counter() - diff_started
        audit_rows = len(diff)

        report_started = time.perf_counter()
        report = build_cleaning_report(
            before=dataframe,
            after=outcome.dataframe,
            plan=proposal.plan,
            validation=outcome.validation,
            execution_records=outcome.execution_records,
            policy=policy,
            planner_name="Rule-based baseline large-data smoke test",
            source_label=f"synthetic_large:{rows}:columns={columns}:seed={seed}",
            rolled_back=outcome.rolled_back,
        )
        report_seconds = time.perf_counter() - report_started
    elif mode == "core":
        # For very large data, keep the core pipeline test realistic while
        # sampling audit/report generation so runtime does not explode.
        sample_before = dataframe.head(min(audit_sample_rows, len(dataframe)))
        sample_after = outcome.dataframe.head(min(audit_sample_rows, len(outcome.dataframe)))

        diff_started = time.perf_counter()
        diff = plan_diff_frame(sample_before, proposal.plan)
        diff_summary = diff_summary_frame(diff)
        diff_seconds = time.perf_counter() - diff_started
        audit_rows = len(diff)

        report_started = time.perf_counter()
        report = build_cleaning_report(
            before=sample_before,
            after=sample_after,
            plan=proposal.plan,
            validation=outcome.validation,
            execution_records=outcome.execution_records,
            policy=policy,
            planner_name="Rule-based baseline large-data core smoke test",
            source_label=(
                f"synthetic_large_core:{rows}:columns={columns}:"
                f"audit_sample={len(sample_before)}:seed={seed}"
            ),
            rolled_back=outcome.rolled_back,
        )
        report_seconds = time.perf_counter() - report_started
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    total_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)

    result = LargeDatasetSmokeResult(
        mode=mode,
        rows_input=rows,
        rows_after_duplicates=len(dataframe),
        rows_output=len(outcome.dataframe),
        columns=len(dataframe.columns),
        memory_mb=memory_mb,
        plan_steps=len(proposal.plan.steps),
        operations=";".join(
            f"{step.operation}:{step.column or '*'}" for step in proposal.plan.steps
        ),
        validation_valid=outcome.validation.valid,
        rolled_back=outcome.rolled_back,
        validation_issues=";".join(issue.code for issue in outcome.validation.issues),
        audit_rows=audit_rows,
        generation_seconds=round(generation_seconds, 4),
        propose_seconds=round(propose_seconds, 4),
        execute_validate_seconds=round(execute_validate_seconds, 4),
        diff_seconds=round(diff_seconds, 4),
        report_seconds=round(report_seconds, 4),
        total_seconds=round(total_seconds, 4),
    )

    pd.DataFrame([asdict(result)]).to_csv(
        output_dir / "large_dataset_smoke_summary.csv",
        index=False,
    )
    diff_summary.to_csv(
        output_dir / "large_dataset_smoke_change_summary.csv",
        index=False,
    )
    diff.to_csv(
        output_dir / "large_dataset_smoke_change_audit.csv",
        index=False,
    )
    if report:
        (output_dir / "large_dataset_smoke_report.md").write_text(
            report,
            encoding="utf-8",
        )
    outcome.dataframe.head(1000).to_csv(
        output_dir / "large_dataset_processed_preview.csv",
        index=False,
    )
    dataframe.head(1000).to_csv(
        output_dir / "large_dataset_dirty_preview.csv",
        index=False,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["estimate", "core", "full"], default="full")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--columns", type=int, default=CORE_COLUMNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-fraction", type=float, default=0.02)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument("--memory-guard-mb", type=float, default=8_000)
    parser.add_argument("--max-memory-mb", type=float, default=None)
    parser.add_argument("--audit-sample-rows", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 10_001:
        raise SystemExit("--rows must be greater than 10000 for this smoke test.")
    if args.columns < CORE_COLUMNS:
        raise SystemExit(f"--columns must be at least {CORE_COLUMNS}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "estimate":
        estimate = estimate_capacity(
            rows=args.rows,
            columns=args.columns,
            seed=args.seed,
            duplicate_fraction=args.duplicate_fraction,
            sample_rows=args.sample_rows,
            memory_guard_mb=args.memory_guard_mb,
        )
        frame = pd.DataFrame([asdict(estimate)])
        frame.to_csv(output_dir / "large_dataset_capacity_estimate.csv", index=False)
        print(frame.to_string(index=False))
        print(
            "\nWrote capacity estimate to "
            f"{(output_dir / 'large_dataset_capacity_estimate.csv').resolve()}"
        )
        return

    result = run_large_dataset_smoke(
        rows=args.rows,
        seed=args.seed,
        output_dir=output_dir,
        columns=args.columns,
        duplicate_fraction=args.duplicate_fraction,
        mode=args.mode,
        max_memory_mb=args.max_memory_mb,
        audit_sample_rows=args.audit_sample_rows,
    )
    print(pd.DataFrame([asdict(result)]).to_string(index=False))
    print(f"\nWrote large dataset smoke outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
