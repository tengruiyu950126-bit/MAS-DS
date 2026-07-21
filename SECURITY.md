# Security policy

## Supported status

MAS-DS is an early-stage research and portfolio project. Security fixes are
provided on the latest branch only; there is currently no long-term-support
release. Cleaning plans and outputs must be reviewed before use.

## Reporting a vulnerability

Use the repository's private GitHub Security Advisory channel after the public
repository is created. If private vulnerability reporting is unavailable,
open a minimal issue requesting a private contact channel without including
exploit details, credentials, private datasets, or sensitive logs.

Do not attach confidential CSV files, contracts, traces, reports, environment
files, or screenshots containing real data to a public issue.

## Security model

- Rule-based and Multi-Expert processing runs locally.
- Cleaning is limited to typed, whitelisted operations.
- Contracts and planner output are treated as data; MAS-DS does not use
  `eval`, `exec`, dynamic imports, or arbitrary generated-code execution.
- Identifier-like and explicitly protected columns are excluded from automatic
  mutation and checked again during critique and validation.
- Candidate results are validated before commit; failed validation follows the
  rollback path. Chunked CSV output uses same-filesystem staging and atomic
  replacement where the filesystem supports it.
- Local LLM support is optional and uses a user-managed Ollama service. No paid
  or cloud API is required by the project.

These controls reduce risk but do not guarantee correct cleaning, complete
privacy, or fitness for a regulated workflow.

## Secrets and sensitive data

Never commit `.env`, `.streamlit/secrets.toml`, API keys, tokens, passwords,
private connection strings, uploaded datasets, or processed user outputs.
Use environment variables for optional local configuration and review every
trace/report before sharing. If a credential is committed, rotate it first;
deleting it from the working tree does not remove it from Git history.
