"""Shared typed state passed between preprocessing experts."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from models.cleaning_plan import CleaningPlan
from models.data_contract import ContractValidationResult
from models.profile import DataProfile
from models.validation import ValidationResult
from tools.cleaning import ExecutionRecord


class WorkflowState(TypedDict, total=False):
    dataframe: pd.DataFrame
    profile: DataProfile
    cleaning_plan: CleaningPlan
    approved: bool | None
    candidate: pd.DataFrame
    result: pd.DataFrame
    execution_records: list[ExecutionRecord]
    validation_result: ValidationResult
    rolled_back: bool
    status: str
    contract_validation_result: ContractValidationResult
