# Real Public Dataset Benchmark

MAS-DS has been benchmarked on real public datasets bundled with scikit-learn.
This benchmark is local, free, and does not download data.

## Datasets

| Dataset | Source | Rows in full dataset | Notes |
|---|---|---:|---|
| Iris | scikit-learn bundled public dataset | 150 | Classic flower measurement dataset |
| Wine | scikit-learn bundled public dataset | 178 | Chemical analysis of wines |
| Breast Cancer | scikit-learn bundled public dataset | 569 | Diagnostic numeric-feature dataset |

## Benchmark command

```powershell
python -m evaluation.real_dataset_benchmark `
  --datasets iris wine breast_cancer `
  --runners in_memory_rule chunked_rule `
  --corruptions missing_value duplicate_row numeric_type `
  --seeds 0 1 2 `
  --fraction 0.05 `
  --chunk-size 100 `
  --planning-sample-rows 100 `
  --output-dir outputs\real_dataset_benchmark
```

## Output files

The benchmark writes:

- `outputs/real_dataset_benchmark/real_dataset_benchmark_results.csv`
- `outputs/real_dataset_benchmark/real_dataset_benchmark_summary.csv`
- `outputs/real_dataset_benchmark/real_dataset_benchmark_report.md`
- `outputs/real_dataset_benchmark/cases/`

## Aggregate result

| Runner | Avg detection F1 | Avg repair success | Avg data preservation | Avg latency seconds | Rollback rate |
|---|---:|---:|---:|---:|---:|
| `in_memory_rule` | 0.6284 | 1.0000 | 1.0000 | 5.6247 | 0.0000 |
| `chunked_rule` | 0.4717 | 1.0000 | 1.0000 | 6.7670 | 0.0000 |

## Interpretation

The key result is that both the standard in-memory preprocessing pipeline and the chunked
CSV path repaired all injected corruptions in this benchmark while preserving
unaffected cells.

Detection F1 is lower than repair success for two reasons:

1. MAS-DS is conservative and often proposes a broader safe operation for a
   column instead of trying to identify every injected cell-level corruption.
2. For duplicate rows, the plan has one dataset-level operation,
   `drop_duplicates`, while the metric counts multiple row-level duplicate
   records. This makes duplicate detection F1 look lower than the practical
   repair result.

This benchmark is stronger than the earlier synthetic-only tests because it
uses real public datasets as clean references, then injects controlled,
reproducible corruptions.

## Current limitation

The benchmark still uses controlled corruptions. The next step would be a
fully naturally dirty public dataset, where the ground truth is not injected by
MAS-DS.
