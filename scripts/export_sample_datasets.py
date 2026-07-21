"""Export built-in sample datasets as CSV files for UI demos."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.datasets import available_dataset_names, load_datasets, make_dirty_copy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=available_dataset_names(),
        default=available_dataset_names(),
    )
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--output-dir", default="data/samples")
    parser.add_argument(
        "--include-dirty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export *_dirty.csv files for UI demos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in load_datasets(args.datasets, rows=args.rows):
        path = output_dir / f"{dataset.name}.csv"
        dataset.dataframe.to_csv(path, index=False)
        print(f"Wrote {dataset.name}: {path}")
        if args.include_dirty:
            dirty_path = output_dir / f"{dataset.name}_dirty.csv"
            make_dirty_copy(dataset.dataframe).to_csv(dirty_path, index=False)
            print(f"Wrote {dataset.name} dirty copy: {dirty_path}")


if __name__ == "__main__":
    main()
