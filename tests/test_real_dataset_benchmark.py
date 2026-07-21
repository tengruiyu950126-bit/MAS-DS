from __future__ import annotations

import pandas as pd
import pytest

from evaluation.real_dataset_benchmark import (
    load_public_dataset,
    run_real_dataset_benchmark,
    sklearn_available,
)


pytestmark = pytest.mark.skipif(
    not sklearn_available(),
    reason="scikit-learn bundled public datasets are unavailable",
)


def test_load_public_dataset_adds_auditable_identifier() -> None:
    dataset = load_public_dataset("iris", rows=12)

    assert dataset.name == "iris"
    assert dataset.source == "scikit-learn bundled public dataset"
    assert len(dataset.dataframe) == 12
    assert dataset.dataframe.columns[0] == "public_row_id"
    assert dataset.feature_columns
    assert "target_name" in dataset.dataframe.columns


def test_real_dataset_benchmark_writes_outputs(tmp_path) -> None:
    records = run_real_dataset_benchmark(
        datasets=["iris"],
        runners=["in_memory_rule", "chunked_rule"],
        corruptions=["missing_value", "duplicate_row"],
        seeds=[0],
        rows=30,
        fraction=0.1,
        chunk_size=7,
        planning_sample_rows=12,
        output_dir=tmp_path,
    )

    assert len(records) == 4
    assert all(record.dataset == "iris" for record in records)
    assert all(record.error_message == "" for record in records)
    assert {record.runner for record in records} == {"in_memory_rule", "chunked_rule"}

    results = pd.read_csv(tmp_path / "real_dataset_benchmark_results.csv")
    summary = pd.read_csv(tmp_path / "real_dataset_benchmark_summary.csv")
    report = (tmp_path / "real_dataset_benchmark_report.md").read_text(encoding="utf-8")

    assert len(results) == 4
    assert set(summary["runner"]) == {"in_memory_rule", "chunked_rule"}
    assert "Real Public Dataset Benchmark" in report
    assert (tmp_path / "cases").exists()
