# Large Dataset Stress Test

This document records the MAS-DS wide-table stress test performed on a synthetic
dataset with more than 80 columns and 1 million base rows.

## Goal

Validate that the local, free MAS-DS preprocessing pipeline can handle a large
wide dataset without relying on paid APIs, cloud services, or remote LLM calls.

## Test shape

| Item | Value |
|---|---:|
| Base rows | 1,000,000 |
| Rows after synthetic duplicates | 1,005,000 |
| Output rows after duplicate removal | 1,000,000 |
| Columns | 85 |
| Duplicate fraction | 0.005 |
| Generated dataframe memory | 532.157 MB |
| Estimated working memory | 2,394.746 MB |
| Memory guard | 8,000 MB |

## Command

```powershell
python -m scripts.run_large_dataset_smoke `
  --mode core `
  --rows 1000000 `
  --columns 85 `
  --duplicate-fraction 0.005 `
  --audit-sample-rows 20000 `
  --max-memory-mb 5000 `
  --output-dir outputs\large_1m_85
```

## Result

| Metric | Value |
|---|---:|
| Mode | core |
| Plan steps | 14 |
| Validation valid | True |
| Rolled back | False |
| Validation issues | 0 |
| Sampled audit rows | 4,292 |
| Generation time | 10.9804 s |
| Proposal/profiling time | 129.7992 s |
| Execution + validation time | 187.0960 s |
| Sampled diff time | 7.3015 s |
| Sampled report time | 8.7434 s |
| Total time | 348.8843 s |

The full core preprocessing flow was executed on the complete 1,005,000-row,
85-column input dataframe. Audit and Markdown report generation were intentionally
sampled on the first 20,000 rows, because full cell-level audit/report generation
for every changed cell at this scale is a separate reporting workload and can
dominate runtime and disk usage.

## Output files

The test writes the following files under `outputs/large_1m_85/`:

- `large_dataset_capacity_estimate.csv`
- `large_dataset_smoke_summary.csv`
- `large_dataset_smoke_change_summary.csv`
- `large_dataset_smoke_change_audit.csv`
- `large_dataset_smoke_report.md`
- `large_dataset_dirty_preview.csv`
- `large_dataset_processed_preview.csv`

## Interpretation

This test confirms that MAS-DS can run its rule-based preprocessing
pipeline on a 1M-row, 85-column in-memory pandas dataset on a local machine.
The most expensive stages are profiling/planning and execution/validation,
which is expected because they scan or transform large columns.

For a future production-grade version, the next important improvement is
chunked or out-of-core execution, so datasets larger than available RAM can be
processed safely.
