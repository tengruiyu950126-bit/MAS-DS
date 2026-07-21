"""CLI for local chunked CSV preprocessing.

Example:

    python -m scripts.run_chunked_preprocessing ^
      --input data/samples/dirty_customers.csv ^
      --output outputs/chunked_cleaned.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.chunked import execute_chunked_csv_atomic
from tools.data_contract import contract_from_json
from tools.policy import load_policy_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAS-DS preprocessing on a CSV file in chunks."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Cleaned output CSV path.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk. Default: 100000.",
    )
    parser.add_argument(
        "--planning-sample-rows",
        type=int,
        default=100_000,
        help="Rows used to propose the rule-based cleaning plan. Default: 100000.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional PreprocessingPolicy JSON path.",
    )
    parser.add_argument(
        "--contract",
        default=None,
        help="Optional local DataContract JSON path.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary CSV path. Defaults to output stem + _summary.csv.",
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help="Optional final cleaning plan JSON path. Defaults to output stem + _plan.json.",
    )
    parser.add_argument(
        "--keep-failed-staging",
        action="store_true",
        help="Debug only: retain this run's failed staging file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = (
        Path(args.summary)
        if args.summary is not None
        else output_path.with_name(f"{output_path.stem}_summary.csv")
    )
    plan_path = (
        Path(args.plan_json)
        if args.plan_json is not None
        else output_path.with_name(f"{output_path.stem}_plan.json")
    )
    policy = load_policy_file(args.policy) if args.policy else None
    contract = (
        contract_from_json(Path(args.contract).read_bytes())
        if args.contract else None
    )

    result = execute_chunked_csv_atomic(
        input_path,
        output_path,
        policy=policy,
        contract=contract,
        chunk_size=args.chunk_size,
        planning_sample_rows=args.planning_sample_rows,
        summary_csv=summary_path,
        plan_json=plan_path,
        keep_failed_staging=args.keep_failed_staging,
    )
    print(f"Transaction ID: {result.transaction_id}")
    print(f"Final status: {result.status}")
    print(f"Rows read: {result.rows_read}")
    print(f"Rows committed: {result.rows_committed}")
    print(f"Previous output preserved: {result.previous_output_preserved}")
    print(f"Staging cleanup succeeded: {result.rollback_cleanup_succeeded}")
    if result.output_fingerprint:
        print(f"Output SHA-256: {result.output_fingerprint}")
    if result.failure_stage:
        print(f"Failure stage: {result.failure_stage}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if result.status == "committed":
        print(f"Committed cleaned CSV to {output_path.resolve()}")
        return 0
    return 2 if result.status == "rolled_back" else 1


if __name__ == "__main__":
    raise SystemExit(main())
