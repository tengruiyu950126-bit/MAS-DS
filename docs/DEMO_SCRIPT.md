# MAS-DS Demo Script

This script is for recording a short demo video, presenting the project, or
walking someone through the app for the first time.

## 1. Start the app

```powershell
cd "path\to\MAS_DS"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

If the virtual environment is not activated yet and PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Fast free demo path

Use this path when you want a reliable demo with zero monetary cost.

1. In the sidebar, choose **Sample dataset**.
2. Select `demo`, `sales`, `customers`, or `students`.
3. Keep **Inject demo issues** enabled.
4. Select **Rule-based baseline**.
5. Open the **Preview** tab and show the dirty data.
6. Open the **Profile** tab and show missing values / duplicates / dtypes.
7. Open the **Cleaning plan** tab.
8. Show:
   - Plan summary
   - Operation mix
   - Detailed plan
9. Open **Execute & validate**.
10. Click **Approve and apply plan**.
11. Show:
    - Result summary
    - Validation summary
    - Execution audit
    - Before / after change audit
    - Before / after preview
12. Download:
    - processed CSV
    - change audit CSV
    - cleaning report Markdown

## 3. What to say during the demo

Short explanation:

> MAS-DS is a local preprocessing workflow. It profiles a CSV, creates
> a structured cleaning plan, asks for human approval, executes only safe
> whitelisted operations, validates the result, and rolls back unsafe changes.
> The important part is not just automatic cleaning; it is auditability and
> safety.

Point out the deterministic safety design:

- Profiling expert understands the dataset.
- Cleaning planner proposes safe operations.
- Human approval prevents blind execution.
- Safe executor applies only whitelisted transformations.
- Validation expert commits or rolls back the result.

Point out the safety design:

- No model-generated code execution.
- ID columns are protected by default.
- Policies can disable operations.
- Validation can roll back unsafe outputs.
- Cell-level change audit shows exactly what changed.

## 4. Optional local LLM demo

Only use this if Ollama is already installed and a model is downloaded.

1. Start Ollama.
2. Select **Hybrid rule + Ollama**.
3. Enter the installed model tag, for example `qwen3:4b`.
4. Run the same sample dataset flow.

Important explanation:

> Local LLM mode still does not call a paid API. The model proposes a structured
> plan, but execution remains restricted to the same safe whitelist.

## 5. Experiment summary demo

Open the **Experiment summary** tab.

Show:

- `experiment_summary_overview.csv`
- generated Markdown experiment report
- best balanced method
- preservation rate
- policy scenario pass rate

Run manually if needed:

```powershell
python -m evaluation.summarize_outputs
```

## 6. Test proof

Run:

```powershell
pytest
```

Latest local offline verification on 2026-07-25:

```text
229 passed, 2 skipped
```

The two skips require optional scikit-learn bundled public datasets that were
unavailable in the verified environment.

## 7. Suggested 60-second pitch

> This project builds a local preprocessing workflow for tabular data.
> Instead of letting an LLM directly mutate data, MAS-DS separates the workflow
> into profiling, planning, human approval, safe execution, validation, and
> reporting. It supports deterministic rule mode, optional local Ollama mode,
> and hybrid orchestration. The system is fully local, can run for zero monetary
> cost, and includes policy control, rollback, cell-level change audit,
> experiment summaries, and an offline regression suite.
