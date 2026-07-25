# Security and Privacy

## Supported deployment

MAS-DS supports local single-user execution. Public or multi-tenant hosting is
not supported. There is no authentication, authorization, tenant isolation, or
server-side retention/deletion service.

## Security boundary

Planners and models may only propose typed operations. Deterministic code
enforces policy, protected columns, operation allowlists, execution,
validation, data contracts, and rollback. No generated code is executed.

## Model data

Ollama-compatible endpoints are loopback-only unless a user explicitly opts in
to a remote destination. Model planning is metadata-only by default. Enabling
sample rows transmits bounded cell values to the configured endpoint. Embedded
URL credentials are rejected. Application logs must never contain cells,
prompts, responses, credentials, or endpoint URLs.

## File handling

The Streamlit path enforces documented byte, row, column, header, and cell
limits and accepts UTF-8 CSV only. Larger files belong on the chunked CLI path.
Processed CSV preserves exact values; audit CSV neutralizes formula-like text.

## Sensitive local content

`.env`, uploads, generated/private data, outputs, logs, databases, model
weights, local documents, virtual environments, and caches are ignored. Ignore
rules are not privacy approval. Review local/ignored files and Git history
before publication. Revoke any credential found in history before sanitizing
files.

## Reporting

Report vulnerabilities privately to the repository owner. Include affected
version, reproduction steps using synthetic data, impact, and suggested
mitigation. Never include real user data or complete credentials.

## Non-guarantees

The project does not guarantee correct repair, confidentiality of data sent to
an explicitly configured remote endpoint, filesystem durability under hardware
failure, regulatory compliance, or safe public hosting.
