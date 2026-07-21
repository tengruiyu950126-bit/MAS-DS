# MAS-DS Release Notes

## Current local release: v0.1.0

Date: 2026-07-11

MAS-DS v0.1.0 is a local multi-agent data preprocessing system focused on safe,
auditable, and explainable tabular-data cleaning.

## Highlights

- Free local rule-based mode.
- Optional local Ollama planner.
- Hybrid rule + local LLM planner.
- Streamlit UI with sample datasets and upload support.
- Human approval before execution.
- Safe whitelist executor.
- Validation and rollback.
- User-configurable safety policies.
- Policy JSON import/export.
- Cell-level before/after change audit.
- Automatic Markdown cleaning report export.
- Experiment summary aggregation.
- Ablation and policy-control experiments.
- Large synthetic dataset smoke test for 10,000+ rows.
- Current verified tests: `116 passed`.

## Core files

| Area | Files |
|---|---|
| App | `app.py` |
| Agents | `agents/` |
| Schemas | `models/` |
| Tools | `tools/` |
| Workflow | `workflow/` |
| Evaluation | `evaluation/` |
| Policy configs | `configs/` |
| Tests | `tests/` |
| Handbooks | `HAND_DS_BOOK/` |

## Important commands

Start the app:

```powershell
streamlit run app.py
```

Run tests:

```powershell
pytest
```

Run rule baseline evaluation:

```powershell
python -m evaluation.run_experiment --methods rule --seeds 0 1 2
```

Run policy evaluation:

```powershell
python -m evaluation.run_policy_experiment
```

Aggregate experiment outputs:

```powershell
python -m evaluation.summarize_outputs
```

Run large dataset smoke test:

```powershell
python -m scripts.run_large_dataset_smoke --rows 20000
```

## GitHub-ready status

The project is ready for a public repository after the owner decides:

1. whether to include Word `.docx` handbooks or only Markdown handbooks;
2. which license to use;
3. whether to include screenshots or a demo GIF.

Recommended before publishing:

- Keep Markdown handbooks.
- Consider excluding generated Word binaries if the repo should stay lightweight.
- Add screenshots under `docs/assets/` if making the GitHub page more visual.
- Run `pytest` once more before pushing.

## Known limitations

- The local LLM planner depends on the installed Ollama model.
- LLM runs can be slow on CPU.
- The cleaning executor intentionally supports a conservative whitelist of operations.
- Evaluation datasets are synthetic but realistic.
- Word document visual render QA requires LibreOffice/`soffice`, which may not be installed.

## Suggested next improvements

1. Add local LLM status/model checks in the UI.
2. Add domain policy presets for customer, sales, and student data.
3. Add charts for experiment summaries.
4. Add screenshots or demo GIFs for GitHub.
5. Prepare a clean release zip excluding `.venv`, caches, and generated outputs.
