"""Presentation helpers for the Streamlit UI."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from models.cleaning_plan import CleaningPlan
from models.data_contract import ContractValidationResult, DataContract
from models.orchestration import OrchestrationTrace
from models.policy import PreprocessingPolicy
from models.profile import DataProfile
from models.validation import ValidationResult
from tools.validation import _duplicate_count, _total_missing


ADVISORY_OPERATIONS = {"flag_outliers_iqr", "leave_unchanged"}


def contract_summary_frame(contract: DataContract | None) -> pd.DataFrame:
    if contract is None:
        return pd.DataFrame([{"active": False, "name": "none", "schema_version": "", "required_columns": 0, "protected_columns": 0, "column_rules": 0}])
    return pd.DataFrame([{
        "active": True,
        "name": contract.name,
        "schema_version": contract.schema_version,
        "required_columns": len(contract.required_columns),
        "protected_columns": len(contract.protected_columns),
        "column_rules": len(contract.columns),
    }])


def contract_findings_frame(result: ContractValidationResult | None) -> pd.DataFrame:
    columns = ["rule_id", "column", "severity", "status", "expected", "observed", "message", "structural"]
    if result is None:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([
        {
            "rule_id": item.rule_id, "column": item.column or "*",
            "severity": item.severity, "status": item.status,
            "expected": item.expected, "observed": item.observed,
            "message": item.message, "structural": item.structural,
        }
        for item in result.findings
    ], columns=columns)


def contract_validation_summary_frame(result: ContractValidationResult | None) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame([{"valid": True, "errors": 0, "warnings": 0, "not_evaluated": 0, "blocks_execution": False}])
    return pd.DataFrame([{
        "valid": result.valid, "errors": result.error_count,
        "warnings": result.warning_count, "not_evaluated": result.not_evaluated_count,
        "blocks_execution": result.block_execution,
    }])

ROUTER_TRACE_COLUMNS = [
    "column", "selected_expert", "issue_type", "confidence", "reason"
]
SPECIALIST_TRACE_COLUMNS = [
    "expert", "proposal_count", "operations", "columns"
]
CRITIC_TRACE_COLUMNS = [
    "decision", "operation", "column", "confidence", "detail"
]
ARBITER_TRACE_COLUMNS = [
    "step", "operation", "column", "confidence", "reason"
]


def router_trace_frame(trace: OrchestrationTrace | None) -> pd.DataFrame:
    """Return router decisions with a stable empty schema."""
    if trace is None:
        return pd.DataFrame(columns=ROUTER_TRACE_COLUMNS)
    return pd.DataFrame(
        [
            {
                "column": route.column or "*",
                "selected_expert": route.selected_expert or "Not routed",
                "issue_type": route.issue_type,
                "confidence": round(route.confidence, 3),
                "reason": route.reason,
            }
            for route in trace.routes
        ],
        columns=ROUTER_TRACE_COLUMNS,
    )


def specialist_trace_frame(trace: OrchestrationTrace | None) -> pd.DataFrame:
    """Summarize proposal ownership without exposing editable state."""
    if trace is None:
        return pd.DataFrame(columns=SPECIALIST_TRACE_COLUMNS)
    rows = []
    for expert, steps in trace.specialist_proposals.items():
        rows.append(
            {
                "expert": expert,
                "proposal_count": len(steps),
                "operations": ", ".join(sorted({step.operation for step in steps})) or "none",
                "columns": ", ".join(sorted({step.column or "*" for step in steps})) or "none",
            }
        )
    return pd.DataFrame(rows, columns=SPECIALIST_TRACE_COLUMNS)


def critic_trace_frame(trace: OrchestrationTrace | None) -> pd.DataFrame:
    """Show approvals, rejections, warnings, and review reasons."""
    if trace is None:
        return pd.DataFrame(columns=CRITIC_TRACE_COLUMNS)
    rows = []
    for decision, steps in (
        ("approved", trace.critique.approved_steps),
        ("rejected", trace.critique.rejected_steps),
    ):
        rows.extend(
            {
                "decision": decision,
                "operation": step.operation,
                "column": step.column or "*",
                "confidence": round(step.confidence, 3),
                "detail": step.reason,
            }
            for step in steps
        )
    rows.extend(
        {
            "decision": "warning",
            "operation": "*",
            "column": "*",
            "confidence": None,
            "detail": warning,
        }
        for warning in trace.critique.warnings
    )
    rows.extend(
        {
            "decision": "reason",
            "operation": "*",
            "column": "*",
            "confidence": None,
            "detail": reason,
        }
        for reason in trace.critique.reasons
        if reason not in trace.critique.warnings
    )
    return pd.DataFrame(rows, columns=CRITIC_TRACE_COLUMNS)


def arbiter_trace_frame(trace: OrchestrationTrace | None) -> pd.DataFrame:
    """Return final arbiter-selected steps with a stable empty schema."""
    if trace is None:
        return pd.DataFrame(columns=ARBITER_TRACE_COLUMNS)
    return plan_frame(trace.final_plan).reindex(columns=ARBITER_TRACE_COLUMNS)


def dataset_metrics_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return compact dataframe-level metrics for display."""
    return pd.DataFrame(
        [
            {
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "missing_values": _total_missing(dataframe),
                "duplicate_rows": _duplicate_count(dataframe),
            }
        ]
    )


def profile_frame(profile: DataProfile) -> pd.DataFrame:
    """Convert a profile object into a readable column-level table."""
    rows = []
    for column, details in profile.column_profiles.items():
        rows.append(
            {
                "column": column,
                "dtype": details.dtype,
                "missing": details.missing_count,
                "missing_ratio": round(details.missing_ratio, 4),
                "unique": details.unique_count,
            }
        )
    return pd.DataFrame(rows)


def plan_frame(plan: CleaningPlan) -> pd.DataFrame:
    """Convert a cleaning plan into a readable table."""
    return pd.DataFrame(
        [
            {
                "step": index + 1,
                "operation": step.operation,
                "column": step.column or "*",
                "confidence": round(step.confidence, 3),
                "reason": step.reason,
            }
            for index, step in enumerate(plan.steps)
        ]
    )


def plan_summary_frame(plan: CleaningPlan) -> pd.DataFrame:
    """Return high-level plan counts for UI status cards."""
    targeted_columns = {
        step.column
        for step in plan.steps
        if step.column is not None
    }
    advisory_steps = sum(
        step.operation in ADVISORY_OPERATIONS
        for step in plan.steps
    )
    mutating_steps = len(plan.steps) - advisory_steps
    average_confidence = (
        sum(step.confidence for step in plan.steps) / len(plan.steps)
        if plan.steps
        else 1.0
    )
    return pd.DataFrame(
        [
            {
                "planned_steps": len(plan.steps),
                "mutating_steps": mutating_steps,
                "advisory_steps": advisory_steps,
                "targeted_columns": len(targeted_columns),
                "avg_confidence": round(average_confidence, 3),
            }
        ]
    )


def operation_summary_frame(plan: CleaningPlan) -> pd.DataFrame:
    """Summarize planned operations by operation name."""
    if not plan.steps:
        return pd.DataFrame(columns=["operation", "steps", "columns"])
    frame = plan_frame(plan)
    grouped = (
        frame.groupby("operation", as_index=False)
        .agg(
            steps=("step", "count"),
            columns=(
                "column",
                lambda values: ", ".join(
                    sorted({str(value) for value in values if str(value) != "*"})
                )
                or "*",
            ),
        )
        .sort_values(["steps", "operation"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return grouped


def policy_summary_frame(policy: PreprocessingPolicy) -> pd.DataFrame:
    """Return the active policy in a compact, readable form."""
    allowed_operations = (
        ", ".join(policy.allowed_operations)
        if policy.allowed_operations
        else "all except denied"
    )
    denied_operations = (
        ", ".join(policy.denied_operations)
        if policy.denied_operations
        else "none"
    )
    protected_columns = (
        ", ".join(policy.protected_columns)
        if policy.protected_columns
        else "none"
    )
    column_overrides = (
        ", ".join(sorted(policy.columns))
        if policy.columns
        else "none"
    )
    return pd.DataFrame(
        [
            {
                "setting": "Auto-protect ID columns",
                "value": "on" if policy.protect_identifier_columns else "off",
                "impact": "Prevents accidental changes to id/customer_id/order_id-like columns.",
            },
            {
                "setting": "Protected columns",
                "value": protected_columns,
                "impact": "Columns that planners should leave unchanged.",
            },
            {
                "setting": "Allowed operations",
                "value": allowed_operations,
                "impact": "Whitelist boundary for cleaning suggestions.",
            },
            {
                "setting": "Disabled operations",
                "value": denied_operations,
                "impact": "Operations blocked by the current policy.",
            },
            {
                "setting": "Outlier handling",
                "value": policy.outlier_action,
                "impact": "Outliers are advisory when set to flag; they are not auto-modified.",
            },
            {
                "setting": "Column overrides",
                "value": column_overrides,
                "impact": "Optional per-column policy overrides.",
            },
        ]
    )


def validation_summary_frame(validation: ValidationResult) -> pd.DataFrame:
    """Return before/after validation metrics in a compact table."""
    return pd.DataFrame(
        [
            {
                "metric": "rows",
                "before": validation.rows_before,
                "after": validation.rows_after,
            },
            {
                "metric": "columns",
                "before": validation.columns_before,
                "after": validation.columns_after,
            },
            {
                "metric": "missing values",
                "before": validation.missing_before,
                "after": validation.missing_after,
            },
            {
                "metric": "duplicate rows",
                "before": validation.duplicates_before,
                "after": validation.duplicates_after,
            },
        ]
    )


def validation_issues_frame(validation: ValidationResult) -> pd.DataFrame:
    """Convert validation issues into a table."""
    return pd.DataFrame(
        [
            {
                "severity": issue.severity,
                "code": issue.code,
                "column": issue.column or "*",
                "message": issue.message,
            }
            for issue in validation.issues
        ]
    )


def execution_records_frame(records: list[object]) -> pd.DataFrame:
    """Convert execution audit dataclasses into a table."""
    return pd.DataFrame([asdict(record) for record in records])


def changed_columns_frame(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Summarize columns whose values or dtypes changed."""
    shared_columns = [column for column in before.columns if column in after.columns]
    rows = []
    for column in shared_columns:
        before_series = before[column].reset_index(drop=True)
        after_series = after[column].reset_index(drop=True)
        comparable = min(len(before_series), len(after_series))
        changed_values = 0
        if comparable:
            changed_values = int(
                (
                    before_series.iloc[:comparable].astype("object")
                    != after_series.iloc[:comparable].astype("object")
                ).sum()
            )
        changed_values += abs(len(before_series) - len(after_series))
        dtype_changed = str(before[column].dtype) != str(after[column].dtype)
        if changed_values or dtype_changed:
            rows.append(
                {
                    "column": column,
                    "before_dtype": str(before[column].dtype),
                    "after_dtype": str(after[column].dtype),
                    "changed_or_removed_rows": changed_values,
                    "dtype_changed": dtype_changed,
                }
            )
    return pd.DataFrame(rows)
