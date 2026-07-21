"""Deterministic local SHA-256 fingerprints for MAS-DS audit provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from models.cleaning_plan import CleaningPlan
from models.data_contract import DataContract
from models.orchestration import OrchestrationTrace
from models.policy import PreprocessingPolicy
from models.provenance import AuditProvenance
from tools.data_contract import contract_fingerprint


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Hash dataframe schema and ordered values while ignoring its row index.

    Row order is intentionally significant because execution operates on the
    presented row order. The canonical payload is used only inside SHA-256 and
    is never returned or placed in provenance metadata.
    """
    schema = {
        "columns": [str(column) for column in dataframe.columns],
        "dtypes": [str(dtype) for dtype in dataframe.dtypes],
        "rows": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
    }
    values = dataframe.to_json(
        orient="split",
        index=False,
        date_format="iso",
        date_unit="ns",
        double_precision=15,
        force_ascii=False,
        default_handler=str,
    )
    digest = hashlib.sha256()
    digest.update(_canonical_json(schema))
    digest.update(b"\n")
    digest.update(values.encode("utf-8"))
    return digest.hexdigest()


def cleaning_plan_fingerprint(plan: CleaningPlan) -> str:
    """Hash the ordered, validated cleaning steps."""
    return _sha256_bytes(_canonical_json(plan.model_dump(mode="json")))


def policy_fingerprint(policy: PreprocessingPolicy) -> str:
    """Hash the complete validated preprocessing policy."""
    return _sha256_bytes(_canonical_json(policy.model_dump(mode="json")))


def orchestration_trace_fingerprint(trace: OrchestrationTrace) -> str:
    """Hash routing, proposals, critique, and final plan deterministically."""
    return _sha256_bytes(_canonical_json(trace.model_dump(mode="json")))


def _utc_timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_audit_provenance(
    *,
    dataframe: pd.DataFrame | None,
    plan: CleaningPlan | None,
    policy: PreprocessingPolicy | None,
    trace: OrchestrationTrace | None = None,
    contract: DataContract | None = None,
    source_label: str | None = None,
    planner_name: str | None = None,
    planner_mode: str | None = None,
    created_at_utc: datetime | None = None,
) -> AuditProvenance:
    """Build safe provenance metadata without embedding dataframe values."""
    notes: list[str] = []
    if dataframe is None:
        notes.append("Dataset fingerprint unavailable: no dataframe was provided.")
    if plan is None:
        notes.append("Cleaning plan fingerprint unavailable: no plan was provided.")
    if policy is None:
        notes.append("Policy fingerprint unavailable: no policy was provided.")
    if trace is None:
        notes.append("Orchestration trace fingerprint unavailable: no trace was provided.")
    if contract is None:
        notes.append("Data contract fingerprint unavailable: no contract was provided.")
    if not source_label:
        notes.append("Source label unavailable.")
    if not planner_name and not planner_mode:
        notes.append("Planner name and mode unavailable.")

    return AuditProvenance(
        created_at_utc=_utc_timestamp(created_at_utc),
        source_label=source_label,
        planner_name=planner_name,
        planner_mode=planner_mode,
        dataset_rows=int(len(dataframe)) if dataframe is not None else None,
        dataset_columns=(
            int(len(dataframe.columns)) if dataframe is not None else None
        ),
        dataset_fingerprint=(
            dataframe_fingerprint(dataframe) if dataframe is not None else None
        ),
        cleaning_plan_fingerprint=(
            cleaning_plan_fingerprint(plan) if plan is not None else None
        ),
        policy_fingerprint=(
            policy_fingerprint(policy) if policy is not None else None
        ),
        orchestration_trace_fingerprint=(
            orchestration_trace_fingerprint(trace) if trace is not None else None
        ),
        contract_fingerprint=(
            contract_fingerprint(contract) if contract is not None else None
        ),
        final_plan_steps=len(plan.steps) if plan is not None else None,
        notes=notes,
    )
