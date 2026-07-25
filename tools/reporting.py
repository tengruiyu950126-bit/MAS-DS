"""Markdown report export for completed preprocessing runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from models.cleaning_plan import CleaningPlan
from models.policy import PreprocessingPolicy
from models.provenance import AuditProvenance
from models.data_contract import ContractValidationResult, DataContract
from models.chunked_transaction import ChunkedTransactionResult
from models.validation import ValidationResult
from tools.cleaning import ExecutionRecord
from tools.diff import diff_summary_frame, plan_diff_frame
from tools.ui_tables import (
    changed_columns_frame,
    dataset_metrics_frame,
    execution_records_frame,
    operation_summary_frame,
    plan_frame,
    plan_summary_frame,
    policy_summary_frame,
    validation_issues_frame,
    validation_summary_frame,
)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    """Render a dataframe as a dependency-free Markdown table."""
    if frame.empty:
        return "_No rows._"
    preview = frame.copy()
    if max_rows is not None:
        preview = preview.head(max_rows)
    preview = preview.fillna("")
    columns = [str(column) for column in preview.columns]
    rows = [
        [_format_value(value) for value in record]
        for record in preview.to_numpy().tolist()
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _safe_section_title(text: str) -> str:
    return text.replace("\n", " ").strip()


def build_cleaning_report(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    plan: CleaningPlan,
    validation: ValidationResult,
    execution_records: list[ExecutionRecord],
    policy: PreprocessingPolicy,
    planner_name: str,
    source_label: str,
    rolled_back: bool,
    generated_at: datetime | None = None,
    max_audit_rows: int = 100,
    provenance: AuditProvenance | dict[str, Any] | None = None,
    contract: DataContract | None = None,
    pre_contract_validation: ContractValidationResult | None = None,
    post_contract_validation: ContractValidationResult | None = None,
    contract_caused_rollback: bool = False,
) -> str:
    """Build a self-contained Markdown report for one preprocessing run."""
    generated_at = generated_at or datetime.now()
    diff = plan_diff_frame(before, plan)
    changed_columns = changed_columns_frame(before, after)
    status = "rolled back" if rolled_back else "committed"
    title = "MAS-DS Cleaning Report"
    provenance_lines: list[str] = []
    if provenance is not None:
        provenance_payload = (
            provenance.model_dump(mode="json")
            if isinstance(provenance, AuditProvenance)
            else dict(provenance)
        )
        provenance_rows = []
        for field, value in provenance_payload.items():
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value) or "none"
            provenance_rows.append({"field": field, "value": value})
        provenance_lines = [
            "## Audit provenance",
            "",
            _markdown_table(pd.DataFrame(provenance_rows)),
            "",
        ]
    contract_lines: list[str] = []
    if contract is not None:
        contract_lines = [
            "## Data contract",
            "",
            _markdown_table(pd.DataFrame([{
                "active": True,
                "name": contract.name,
                "schema_version": contract.schema_version,
                "pre_errors": pre_contract_validation.error_count if pre_contract_validation else "unavailable",
                "pre_warnings": pre_contract_validation.warning_count if pre_contract_validation else "unavailable",
                "post_errors": post_contract_validation.error_count if post_contract_validation else "unavailable",
                "post_warnings": post_contract_validation.warning_count if post_contract_validation else "unavailable",
                "post_valid": post_contract_validation.valid if post_contract_validation else "unavailable",
                "contract_caused_rollback": contract_caused_rollback,
                "contract_fingerprint": (
                    provenance.contract_fingerprint
                    if isinstance(provenance, AuditProvenance) else
                    (provenance or {}).get("contract_fingerprint", "unavailable")
                ),
            }])),
            "",
        ]

    lines = [
        f"# {title}",
        "",
        f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report was generated locally by MAS-DS. It does not call paid APIs.",
        "",
        "## Run summary",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "source": source_label,
                        "planner": planner_name,
                        "status": status,
                        "rolled_back": rolled_back,
                        "plan_steps": len(plan.steps),
                        "validation_issues": len(validation.issues),
                        "audit_rows": len(diff),
                    }
                ]
            )
        ),
        "",
        *provenance_lines,
        *contract_lines,
        "## Dataset metrics",
        "",
        "### Before",
        "",
        _markdown_table(dataset_metrics_frame(before)),
        "",
        "### After committed result",
        "",
        _markdown_table(dataset_metrics_frame(after)),
        "",
        "## Active safety policy",
        "",
        _markdown_table(policy_summary_frame(policy)),
        "",
        "## Cleaning plan",
        "",
        "### Plan summary",
        "",
        _markdown_table(plan_summary_frame(plan)),
        "",
        "### Operation mix",
        "",
        _markdown_table(operation_summary_frame(plan)),
        "",
        "### Detailed plan",
        "",
        _markdown_table(plan_frame(plan)),
        "",
        "## Execution audit",
        "",
        _markdown_table(execution_records_frame(execution_records)),
        "",
        "## Validation",
        "",
        "### Validation summary",
        "",
        _markdown_table(validation_summary_frame(validation)),
        "",
        "### Validation issues",
        "",
        _markdown_table(validation_issues_frame(validation)),
        "",
        "## Change audit",
        "",
    ]

    if rolled_back:
        lines.extend(
            [
                "Validation recommended rollback. The change audit below describes "
                "the candidate changes attempted by the approved plan before the "
                "final output was rolled back to the original data.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The change audit below describes committed candidate changes made "
                "by the approved plan.",
                "",
            ]
        )

    lines.extend(
        [
            "### Change summary",
            "",
            _markdown_table(diff_summary_frame(diff)),
            "",
            f"### Detailed changes (first {max_audit_rows})",
            "",
            _markdown_table(diff, max_rows=max_audit_rows),
            "",
            "### Changed columns in committed result",
            "",
            _markdown_table(changed_columns),
            "",
            "## Interpretation notes",
            "",
            "- `committed` means validation passed and MAS-DS returned the processed data.",
            "- `rolled back` means validation detected risk and returned the original data.",
            "- `value_changed` rows identify exact cell-level edits.",
            "- `dtype_changed` rows identify type conversions even when values look similar.",
            "- `row_removed` rows identify removed duplicate rows.",
            "",
            "## Reproducibility",
            "",
            "To reproduce the run, use the same input CSV, planner mode, local model "
            "configuration if applicable, and policy JSON.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report_filename(source_label: str) -> str:
    """Return a stable report filename from a Streamlit source label."""
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in _safe_section_title(source_label)
    ).strip("_")
    return f"mas_ds_cleaning_report_{safe or 'run'}.md"


def build_chunked_transaction_report(result: ChunkedTransactionResult) -> str:
    """Build a value-free Markdown audit for an atomic chunked transaction."""
    core = {
        "transaction_id": result.transaction_id,
        "status": result.status,
        "started_at_utc": result.started_at_utc,
        "completed_at_utc": result.completed_at_utc,
        "rows_read": result.rows_read,
        "rows_staged": result.rows_staged,
        "rows_committed": result.rows_committed,
        "output_previously_existed": result.output_previously_existed,
        "previous_output_preserved": result.previous_output_preserved,
        "rollback_cleanup_succeeded": result.rollback_cleanup_succeeded,
        "failure_stage": result.failure_stage,
        "error_message": result.error_message,
    }
    fingerprints = {
        "input_fingerprint": result.input_fingerprint,
        "output_fingerprint": result.output_fingerprint,
        "plan_fingerprint": result.plan_fingerprint,
        "policy_fingerprint": result.policy_fingerprint,
        "contract_fingerprint": result.contract_fingerprint,
    }
    validation = result.validation_result.model_dump() if result.validation_result else {}
    if isinstance(validation.get("messages"), list):
        validation["messages"] = "; ".join(validation["messages"]) or "none"
    contract = result.contract_validation_result
    contract_summary = ({
        "contract_name": contract.contract_name,
        "schema_version": contract.schema_version,
        "valid": contract.valid,
        "errors": contract.error_count,
        "warnings": contract.warning_count,
        "not_evaluated": contract.not_evaluated_count,
    } if contract else {})
    return "\n".join([
        "# MAS-DS Atomic Chunked Transaction Report",
        "",
        "This local audit contains aggregate state and fingerprints only; it does not contain dataframe cell values.",
        "",
        "## Transaction",
        "",
        _markdown_table(pd.DataFrame([core])),
        "",
        "## Fingerprints",
        "",
        _markdown_table(pd.DataFrame([fingerprints])),
        "",
        "## Staged-output validation",
        "",
        _markdown_table(pd.DataFrame([validation])) if validation else "_Unavailable._",
        "",
        "## Data-contract validation",
        "",
        _markdown_table(pd.DataFrame([contract_summary])) if contract_summary else "_No active contract._",
        "",
        "## Warnings",
        "",
        "\n".join(f"- {item}" for item in result.warnings) or "_No warnings._",
        "",
    ])
