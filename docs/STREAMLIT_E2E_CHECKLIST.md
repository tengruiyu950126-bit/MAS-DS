# Streamlit end-to-end checklist

These checks use the existing environment without installing or updating anything.

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

The app is local and free. No payment, API key, internet connection, Ollama, or
cloud service is required for Rule-based or Multi-Expert mode. Stop it with
`Ctrl+C` in the server terminal.

## Successful Multi-Expert workflow

1. Keep **Sample dataset**, enable demo issues, select **Multi-Expert**.
2. In **Data contract**, select **Built-in example**.
3. Confirm Dataset overview/Profile and pre-cleaning contract results appear.
4. Open **Multi-Expert Orchestration Trace**. Confirm Router, specialist, Critic,
   and Arbiter tables are readable and read-only.
5. Review the plan, then click **Approve and apply plan**.
6. Expect “Validation passed. Changes were committed.” and a post-contract result.
7. Confirm non-empty download controls for trace JSON, trace Markdown, processed
   CSV, change audit (when changes exist), and cleaning report.

## Contract rollback workflow

Select **Paste JSON** and use:

```json
{"name":"Rollback check","columns":{"age":{"numeric_max":10,"severity":"error"}}}
```

With the demo sample, approve the proposed plan. Expect an error-level post-cleaning
finding, rollback indication, and the original preview/result data preserved.

## Expected audit content

Trace JSON is structured; Markdown is human-readable. Both include provenance but
not raw cell values in provenance. Cleaning reports identify planner, policy,
contract, validation, and rollback. Processed CSV intentionally contains processed
user data and must be handled according to the user's data policy.

## Common local errors

- “Address already in use”: stop the old server or add `--server.port <unused-port>`.
- Streamlit import failure: verify the command uses `.venv\Scripts\python.exe`; do
  not install packages as part of validation.
- Ollama connection errors: choose Rule-based or Multi-Expert; neither needs Ollama.
- A blocked execution means the contract has a structural error such as a missing
  required protected column; correct the data/contract before retrying.
