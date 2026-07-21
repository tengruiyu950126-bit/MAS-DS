# Contributing to MAS-DS

## Setup and tests

Use Python 3.11 or newer in a dedicated virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,evaluation]"
python -m pytest -q
```

Rule-based and Multi-Expert tests must remain deterministic, local, and free.
Tests must not require Ollama, internet access, paid APIs, or a live Streamlit
server unless explicitly marked as a local runtime check.

## Safe changes

- Specialists may propose only typed `CleaningStep` objects and must never
  mutate the input dataframe directly.
- New operations require a typed plan entry, deterministic executor branch,
  policy checks, Critic behavior, validation coverage, rollback tests, and
  documentation. Never execute generated source code.
- Preserve identifier and contract-protected column behavior at Router,
  specialist, Critic, Arbiter, executor, and validation boundaries.
- Resolve uncertainty conservatively with rejection or `leave_unchanged`.

## Evaluation contributions

Add independently labeled scenarios rather than deriving labels from Router
output. Record explicit seeds, cover clean and protected controls, and report
precision/recall/F1—not accuracy dominated by negative labels. Evaluation-only
ablations must not change production defaults.

Fixtures must be artificial or from already bundled public datasets. Never
commit uploaded data, raw previews, processed user CSVs, cell values in audit
metadata, local paths, credentials, logs, or temporary staging names.

Before submitting a change, run the focused tests and the full suite, validate
documentation links, and inspect the staged diff for generated or sensitive
files.
