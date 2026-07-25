# Chunked / Out-of-Core Preprocessing

MAS-DS now includes a local chunked CSV preprocessing path for datasets that are
too large to comfortably load into memory at once.

## Why this exists

The standard Streamlit workflow reads a CSV into a pandas DataFrame and runs the
full deterministic preprocessing lifecycle in memory. That path is simple, auditable, and good for small to
medium datasets, but large files can stress memory because pandas operations and
validation may create copies.

The chunked path processes CSV files in bounded row chunks.

## Command

```powershell
python -m scripts.run_chunked_preprocessing `
  --input data\samples\customers_dirty.csv `
  --output outputs\chunked_customers_cleaned.csv `
  --chunk-size 100000 `
  --planning-sample-rows 100000
```

Optional outputs:

```powershell
python -m scripts.run_chunked_preprocessing `
  --input path\to\large_dirty.csv `
  --output outputs\large_cleaned.csv `
  --summary outputs\large_cleaned_summary.csv `
  --plan-json outputs\large_cleaned_plan.json
```

Optional local data contract:

```powershell
python -m scripts.run_chunked_preprocessing `
  --input path\to\large_dirty.csv `
  --output outputs\large_cleaned.csv `
  --contract configs\customer_contract.json
```

## What it does

The chunked workflow has five main stages:

1. Plan from a bounded sample.
   - Uses the deterministic rule-based cleaning agent.
   - Does not call paid APIs or cloud LLMs.

2. Scan the full CSV for global parameters.
   - Counts input rows and missing values.
   - Detects exact duplicates across chunk boundaries.
   - Computes global fill values, such as median, mean, and mode.
   - Builds global text normalization mappings where needed.

3. Execute the plan chunk by chunk into a unique staging file beside the final
   destination.
   - Applies only whitelisted MAS-DS operations.
   - Removes duplicates across chunks while preserving first occurrences.
   - Writes the staged CSV incrementally with one header.
   - Never writes processed rows directly to the requested destination.

4. Validate the complete staging file.
   - Reopens and scans it for readability, schema, row count, and practical
     missing/row safety.
   - Runs streaming Data Contract validation, including global uniqueness.
   - Error findings and error-severity `not_evaluated` findings block commit;
     warnings do not.

5. Commit once.
   - Closes staging handles and fingerprints the staged file.
   - Calls `os.replace(staging, destination)` on the same filesystem.
   - Reopens, recounts, and fingerprints the committed output before reporting
     success.
   - Writes optional summary and plan sidecars only after output commit.

## Atomic execution and existing-output protection

`execute_chunked_csv` is atomic by default while retaining its existing summary
return type. `execute_chunked_csv_atomic` returns a typed
`ChunkedTransactionResult` with transaction ID, status history, row counts,
validation, failure stage, cleanup state, and SHA-256 fingerprints.

If a destination exists, it is not edited or deleted during staging.
Pre-commit read, transform, write, validation, contract, and commit-call
failures preserve it byte-for-byte. If the destination did not exist,
pre-commit failure leaves it absent. Cleanup targets only the exact sibling
staging file containing the current random transaction ID and never removes a
directory recursively.

Each destination also has an exclusive sibling lock file for the duration of a
transaction. A second writer to the same output is rejected before staging and
cannot silently replace the first writer's result. A process crash can leave a
stale lock; inspect the owning process and transaction metadata before manually
removing it.

The guarantee relies on `os.replace` for paths in the same directory and
filesystem; there is no delete-then-rename sequence. Local Windows and POSIX
filesystems normally provide atomic replacement. Replacement can still fail
because of restrictive file locks, permissions, a full disk, or nonstandard
network filesystem semantics. A machine/storage failure after replacement but
before verification can leave the new file committed; MAS-DS reports this as
`failed`, not as a successful rollback.

## Transaction statuses and CLI exit codes

- `pending`: path and input preflight.
- `staging`: chunks are written to the private staging file.
- `validating`: staged structure and contract are checked.
- `committed`: replacement and post-commit verification succeeded.
- `rolled_back`: a staged run failed before commit and cleanup was attempted.
- `failed`: failure occurred before staging or after replacement could no
  longer be represented as rollback.

The CLI prints transaction ID, aggregate rows, preservation, cleanup, failure
stage, and fingerprints—never dataframe previews or cell values. Exit `0` means
verified commit, exit `2` means rolled back, and exit `1` means failed.
`--keep-failed-staging` is debug-only; cleanup remains the safe default.

## Supported operations

The chunked executor supports the existing MAS-DS whitelist operations:

- `drop_duplicates`
- `fill_mean`
- `fill_median`
- `fill_mode`
- `convert_numeric`
- `convert_datetime`
- `parse_numeric_text`
- `strip_whitespace`
- `normalize_case`
- `normalize_category_typos`
- `flag_outliers_iqr`
- `leave_unchanged`

## Important design detail

Global statistics are calculated before chunk execution. This avoids a common
large-data cleaning bug where each chunk uses a different median or mode.

Example:

```text
Bad chunked behavior:
chunk 1 missing age -> fill with chunk 1 median
chunk 2 missing age -> fill with chunk 2 median

MAS-DS behavior:
full CSV scan -> global age median
every chunk -> fill missing age with the same global median
```

## Current limitations

- Planning is based on a bounded sample, so rare data-quality issues that appear
  only outside the planning sample may be missed.
- Exact median is computed by collecting values for columns that need median
  imputation. This is much lighter than loading the whole DataFrame, but very
  wide files with many median-imputed columns can still use memory.
- The Streamlit UI still uses the in-memory path. The chunked path is currently
  exposed through the CLI.
- Full cell-level audit for every changed cell is intentionally not generated in
  chunked mode, because the audit can become larger than the cleaned dataset.
- Atomicity cannot cover every power-loss or faulty-filesystem scenario and
  does not add directory durability calls such as Windows directory `fsync`.
- Optional summary and plan sidecars are written after CSV commit. Sidecar
  failure is a warning and does not undo a verified CSV commit.

## Verified behavior

Automated tests cover:

- chunked output matching the in-memory rule pipeline on representative data;
- global median filling instead of per-chunk median filling;
- duplicate removal across chunk boundaries, including when the planning sample
  misses duplicates.
- atomic creation/replacement, output preservation under controlled failures,
  cleanup reporting, fingerprints, transaction serialization, and CLI exits.
