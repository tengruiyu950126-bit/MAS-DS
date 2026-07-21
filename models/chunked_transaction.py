"""Typed audit contracts for atomic chunked CSV transactions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from models.data_contract import ContractValidationResult


TransactionStatus = Literal[
    "pending", "staging", "validating", "committed", "rolled_back", "failed"
]


class ChunkedOutputValidation(BaseModel):
    valid: bool
    readable: bool
    header_matches: bool
    row_count_matches: bool
    expected_rows: int = Field(ge=0)
    observed_rows: int = Field(ge=0)
    expected_columns: int = Field(ge=0)
    observed_columns: int = Field(ge=0)
    messages: list[str] = Field(default_factory=list)


class ChunkedTransactionResult(BaseModel):
    schema_version: int = 1
    transaction_id: str
    source_path: str
    requested_output_path: str
    staging_path: str | None = None
    status: TransactionStatus
    status_history: list[TransactionStatus]
    started_at_utc: str
    completed_at_utc: str | None = None
    rows_read: int = Field(default=0, ge=0)
    rows_staged: int = Field(default=0, ge=0)
    rows_committed: int = Field(default=0, ge=0)
    output_previously_existed: bool = False
    previous_output_preserved: bool = True
    rollback_cleanup_succeeded: bool = True
    validation_result: ChunkedOutputValidation | None = None
    contract_validation_result: ContractValidationResult | None = None
    failure_stage: str | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    input_fingerprint: str | None = None
    output_fingerprint: str | None = None
    plan_fingerprint: str | None = None
    policy_fingerprint: str | None = None
    contract_fingerprint: str | None = None
    summary: dict[str, Any] | None = None
