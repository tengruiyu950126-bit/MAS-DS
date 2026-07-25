# GitHub release checklist

Use this checklist before publishing or updating MAS-DS.

## Public release content

Include only reviewed project material:

- source packages: `agents/`, `models/`, `tools/`, `evaluation/`, `providers/`,
  and `scripts/`;
- Streamlit entry point: `app.py`;
- tests: `tests/`;
- CI and project configuration: `.github/`, `pyproject.toml`,
  `constraints-tested.txt`, `.env.example`, and `.gitignore`;
- reviewed sample data and policies: `data/samples/` and `configs/`;
- public documentation: `README.md`, `SECURITY.md`, this checklist, `docs/`,
  and the reviewed aggregate-only schema-2.0 evaluation subset under
  `evaluation/results/public/`.

Exclude from the public release:

- internal review material under `CHECK/` and `Need_Fix/`;
- ignored private/local material under `HAND_DS_BOOK/`;
- generated results under `outputs/`;
- local environments, caches, logs, uploads, processed datasets, generated
  artifacts, private documents, databases, model weights, and local secrets.

Ignored files are not automatically approved for publication. Review the final
Git archive for privacy, copyright, authorship, credentials, and dataset rights.

## Current product classification

MAS-DS is a local single-user data-cleaning application with one authoritative
deterministic lifecycle. It provides a deterministic multi-expert workflow and
optional bounded LLM-assisted planning. It is not a genuine Multi-Agent System,
and public or multi-user production deployment is unsupported.

## Verification before pushing

From the repository root, run:

```powershell
python -m pip check
python -m compileall -q agents evaluation models providers scripts tools
python -m scripts.check_text_encoding
python -m scripts.check_secret_patterns
python -m pytest -q -ra
```

The offline run dated 2026-07-25 completed with `229 passed, 2 skipped`. The two
skips were optional scikit-learn bundled-dataset tests unavailable in that local
environment. This result verifies the tested assertions only; it does not prove
general repair correctness, model quality, privacy, or production readiness.

After an authorized push or pull request, confirm both Python jobs in the
`offline-quality` GitHub Actions workflow. Also run an approved history-aware
secret scan with redacted output and inspect a Git-generated release archive.

Optional local UI smoke test:

```powershell
python -m streamlit run app.py
```

Use a sample dataset and the rule-based baseline for an offline, no-model smoke
path. Review every proposed plan and output.

## Release metadata

- Confirm the MIT license and citation metadata are correct.
- Keep screenshots or demo media only when reviewed and intentionally tracked.
- Do not add private handbooks or generated release archives.

Recommended GitHub description:

```text
Local deterministic workflow for auditable tabular-data preprocessing.
```

Suggested topics:

```text
data-cleaning, preprocessing, streamlit, pandas, ollama, python, data-quality
```
