# MAS-DS Release Notes

## Current local release: v0.1.0

MAS-DS is portfolio and research software for local, single-user tabular-data
cleaning. It is not production-ready, does not support public or multi-user
deployment, and does not guarantee semantically correct repairs.

## Current architecture

One authoritative deterministic lifecycle performs profiling, typed planning,
policy approval, fixed-allowlist execution, result and contract validation, and
commit or rollback. Planner modes are:

- rule-based baseline;
- deterministic multi-expert workflow;
- optional bounded LLM-assisted planning through an Ollama-compatible endpoint;
- hybrid deterministic and bounded LLM-assisted planning.

The deterministic multi-expert workflow is not a genuine Multi-Agent System.
Models can propose typed operations but cannot mutate production dataframes,
approve plans, execute code, or bypass deterministic validation and rollback.

## Release highlights

- bounded UTF-8 CSV ingestion with stable failure handling;
- explicit human approval in the Streamlit UI;
- typed cleaning plans, preprocessing policies, and data contracts;
- loopback model endpoints by default, explicit remote opt-in, metadata-only
  planning by default, and bounded optional sample transmission;
- structured value-free lifecycle diagnostics with run IDs;
- atomic chunked output staging, validation, destination locking, and commit;
- value-level repair and unaffected-cell preservation metrics;
- audit exports with bounded detail and spreadsheet-formula neutralization;
- console commands for chunked processing and deterministic evaluation;
- offline CI on supported Python 3.11 and 3.13.

## Verification

The offline suite run on 2026-07-25 completed with:

```text
229 passed, 2 skipped
```

The two skips were optional scikit-learn bundled-dataset tests unavailable in
that local environment. No external model endpoint was required. This result
verifies repository assertions in that environment only; it is not evidence of
general model quality, guaranteed repair, guaranteed privacy, or production
readiness.

Before release, run:

```powershell
python -m pip check
python -m compileall -q agents evaluation models providers scripts tools
python -m scripts.check_text_encoding
python -m scripts.check_secret_patterns
python -m pytest -q -ra
```

## Important commands

Start the local app:

```powershell
python -m streamlit run app.py
```

Run chunked preprocessing:

```powershell
python -m scripts.run_chunked_preprocessing --help
```

Run the deterministic evaluation:

```powershell
python -m evaluation.run_experiment --methods rule --seeds 0 1 2
```

## Public release boundary

Public release content may include reviewed source, tests, CI/configuration,
sample data, policies, and public documentation.

Exclude internal material under `CHECK/` and `Need_Fix/`, ignored private/local
material under `HAND_DS_BOOK/`, generated `outputs/`, local environments,
caches, logs, uploads, processed datasets, generated artifacts, private
documents, databases, model weights, and secrets.

Human privacy, copyright, authorship, dataset-provenance, release-archive, and
Git-history credential reviews remain required before publication.

## Known limitations

- Optional model compatibility and quality depend on the installed endpoint and
  model and are not verified by the offline suite.
- Explicitly opted-in non-loopback endpoints may receive metadata and enabled
  sample rows; remote plaintext HTTP is not transport-confidential.
- Processed CSV preserves exact values, including formula-leading text; audit
  CSV exports neutralize formula-like text.
- Exact global duplicate and median operations in the chunked path can consume
  memory that grows with total data.
- Current public evaluation artifacts are a reviewed aggregate-only schema-2.0
  subset covering seeds 0–4, 100 routing scenarios, 700 ablation runs, and
  seven configurations. Scenario and run-level rows remain ignored review
  evidence. Expected-operation coverage is not repaired-cell correctness;
  timings are environment-specific. A roughly 0.716216 false/unnecessary
  routing rate and 0.0 clean-data no-op accuracy disclose weak routing
  selectivity. These synthetic results do not prove genuine-MAS behavior or
  production readiness.
- Every plan and output requires human review and domain-specific governance.
