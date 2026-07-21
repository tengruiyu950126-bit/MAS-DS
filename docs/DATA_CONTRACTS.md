# Data Contracts and Semantic Validation

MAS-DS data contracts are local, typed declarations of dataset requirements.
They protect multi-agent cleaning by separating semantic rules from planner
suggestions. Contract JSON is validated by Pydantic and is never evaluated as
code, imported as a module, or allowed to select filesystem paths.

## Supported rules

A contract has a name, schema version, optional description,
required/optional/protected columns, extra-column policy, and column rules.

| Rule | Meaning |
|---|---|
| `allowed_dtypes` | any, string, number, integer, float, boolean, datetime, or category |
| `nullable` | Whether missing cells are allowed |
| `max_missing_ratio` | Maximum missing fraction from 0 to 1 |
| `numeric_min` / `numeric_max` | Inclusive numeric range |
| `allowed_values` | Categorical allowlist |
| `datetime_min` / `datetime_max` | Inclusive ISO-8601 bounds |
| `unique` | Non-missing values must be globally unique |
| `regex` | Full-match text pattern |
| `text_min_length` / `text_max_length` | Inclusive text-length bounds |

Every column has a default `severity` (`warning` or `error`) and optional
`rule_severities` overrides. Required and unexpected columns have separate
dataset-level severities.

## JSON example

```json
{
  "name": "Customer intake contract",
  "schema_version": "1.0",
  "required_columns": ["customer_id", "age", "status"],
  "optional_columns": ["created_at", "email"],
  "protected_columns": ["customer_id", "email"],
  "allow_extra_columns": true,
  "columns": {
    "customer_id": {
      "allowed_dtypes": ["string"],
      "nullable": false,
      "unique": true
    },
    "age": {
      "allowed_dtypes": ["number"],
      "nullable": false,
      "numeric_min": 0,
      "numeric_max": 120
    },
    "status": {
      "allowed_values": ["active", "inactive"]
    },
    "created_at": {
      "datetime_min": "2020-01-01T00:00:00Z",
      "datetime_max": "2030-12-31T23:59:59Z",
      "severity": "warning"
    },
    "email": {
      "regex": "[^@\\s]+@[^@\\s]+\\.[^@\\s]+",
      "text_max_length": 254,
      "rule_severities": {"regex": "warning"}
    }
  }
}
```

## Structured validation

`ContractValidationResult` contains dimensions, overall validity,
error/warning/not-evaluated counts, `block_execution`, and typed findings.
Every finding records rule ID, column, severity, status, expected condition,
aggregate observed summary, safe message, and whether it is structural.

Findings contain counts and schema summaries only. They never include raw or
sample cell values, including values that violated regexes or allowlists.

## Pre-cleaning validation

The contract is evaluated before graph planning. Missing required columns are
structural. Error-severity structural findings block execution and preserve the
original dataframe. Repairable missing, type, range, category, or text
violations remain visible but may proceed to cleaning.

## Planner safety

Contract protections are merged into `PreprocessingPolicy` without mutation:

- Router marks protected columns as not routed.
- Specialists receive no mutating route for protected columns.
- Rule, local LLM, and hybrid modes inherit policy protection.
- Critic rejects protected changes, incompatible numeric/datetime conversion,
  and uncertain category rewriting under an allowlist.
- Arbiter receives only critic-approved proposals.

Contracts remain data and are never passed to `eval`, `exec`, dynamic imports,
or arbitrary generated-code execution.

## Post-cleaning validation and rollback

After execution, MAS-DS runs its original validator and the active contract.
Contract findings become existing `ValidationIssue` records with
`data_contract:<rule_id>` codes. Remaining error-level findings set the normal
rollback recommendation, so the graph returns the original dataframe.
Warnings are reported but do not force rollback.

`PreprocessingOutcome.contract_caused_rollback` distinguishes contract-caused
rollback. Pre- and post-cleaning results stay separate for UI and reports.

## Streamlit usage

1. Run `streamlit run app.py`.
2. Open **Data contract** in the sidebar.
3. Choose **No contract** (default), **Built-in example**, **Paste JSON**, or
   **Upload JSON**.
4. Review pre-cleaning results in the **Data contract** tab.
5. Review post-cleaning findings in **Execute & validate**.

The built-in example is warning-only. Invalid JSON shows an actionable error
and does not activate a contract.

## Python usage

```python
from tools.data_contract import contract_from_json, validate_dataframe_contract
from workflow.graph import PreprocessingGraphOrchestrator

contract = contract_from_json(contract_json)
pre = validate_dataframe_contract(dataframe, contract)
orchestrator = PreprocessingGraphOrchestrator(contract=contract)
proposal = orchestrator.propose(dataframe)
outcome = orchestrator.execute_approved(dataframe, proposal.plan)
```

## Reports, provenance, and privacy

Cleaning reports include contract name/version, pre/post counts, outcome,
contract-caused rollback, and SHA-256 contract fingerprint. Provenance contains
the same deterministic fingerprint, linking reports and trace exports to the
exact contract. Existing callers without a contract remain compatible.

Everything runs locally and free. No contract finding, trace metadata,
provenance record, or report contract section contains raw dataframe values.

## Chunked CSV validation

`validate_chunked_csv_contract(path, contract, chunk_size=...)` scans without
loading the full dataframe. It checks required/schema rules, inferred types,
global missing ratio, ranges, allowlists, regex, text length, and uniqueness
across chunks. Uniqueness stores SHA-256 digests, not raw values.

CSV does not preserve categorical dtype metadata, so a `category` dtype rule
returns an explicit `not_evaluated` finding. `execute_chunked_csv(...,
contract=contract)` merges protected columns into policy and blocks structural
preflight errors. Chunks are written to a unique sibling staging file.
Error-level post-cleaning findings, including error-severity `not_evaluated`
findings, prevent atomic replacement and preserve any old destination.
Warnings permit commit. The CLI accepts `--contract`; Streamlit large-file
contract execution is not currently exposed.

## Limitations

- Pandas dtype inference can make numeric/date text a repairable type mismatch.
- Regex uses Python full-match semantics.
- Streaming uniqueness retains one fixed-size digest per unique value.
- Contract hashes are integrity links, not digital signatures.
- Cross-column semantic relationships are not yet supported.
