"""Structured metadata linking MAS-DS audit artifacts to their inputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuditProvenance(BaseModel):
    """JSON-serializable fingerprints and context for one planning run."""

    schema_version: int = 1
    created_at_utc: str
    source_label: str | None = None
    planner_name: str | None = None
    planner_mode: str | None = None
    dataset_rows: int | None = Field(default=None, ge=0)
    dataset_columns: int | None = Field(default=None, ge=0)
    dataset_fingerprint: str | None = None
    cleaning_plan_fingerprint: str | None = None
    policy_fingerprint: str | None = None
    orchestration_trace_fingerprint: str | None = None
    contract_fingerprint: str | None = None
    final_plan_steps: int | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)
