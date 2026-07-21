# GitHub checklist

Use this checklist before publishing or updating the repository.

Current verified test status:

```text
116 passed
```

## Keep

- Source code: `agents/`, `workflow/`, `models/`, `tools/`, `evaluation/`,
  `providers/`, `scripts/`
- Streamlit app: `app.py`
- Project metadata: `pyproject.toml`, `.env.example`, `.gitignore`
- Demo data: `data/samples/*.csv`
- Tests: `tests/`
- Documentation: `README.md`, `GITHUB_CHECKLIST.md`, `docs/`
- Optional handbooks: `HAND_DS_BOOK/*.md`

## Do not upload

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.env`
- local/private CSV files
- generated experiment outputs in `outputs/`
- generated release ZIP files

## Decide before publishing

- License: choose a license before making the repo public.
- Screenshots/GIF: optional but recommended for a stronger GitHub landing page.
- Word files: keep `HAND_DS_BOOK/*.docx` only if you want binary handbook files
  in the repo. Markdown handbooks are usually easier to review on GitHub.

The `.gitignore` already covers these paths.

## Before pushing

Run:

```powershell
pytest
```

Optionally smoke-test the UI:

```powershell
streamlit run app.py
```

For a fast no-model demo, use:

1. **Sample dataset**
2. **Inject demo issues**
3. **Rule-based baseline**
4. **Approve and apply plan**

This path is fully local and costs no money.

## Useful docs

- `docs/DEMO_SCRIPT.md`: step-by-step demo flow.
- `docs/RELEASE_NOTES.md`: current release summary and known limitations.
- `HAND_DS_BOOK/MAS_DS_REPORT_ZH.md`: Chinese project handbook.
- `HAND_DS_BOOK/MAS_DS_REPORT_EN.md`: English project handbook.

## Recommended GitHub description

```text
Free local multi-agent system for safe, auditable tabular-data preprocessing.
```

## Suggested tags

```text
llm-agents, multi-agent-system, data-cleaning, preprocessing, streamlit,
langgraph, ollama, python, data-quality
```
