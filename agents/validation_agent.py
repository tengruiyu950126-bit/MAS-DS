"""Validation expert for commit-or-rollback decisions."""

from __future__ import annotations

import pandas as pd

from models.cleaning_plan import CleaningPlan
from models.data_contract import ContractValidationResult, DataContract
from models.policy import PreprocessingPolicy
from models.validation import ValidationIssue, ValidationResult
from tools.validation import validate_preprocessing
from tools.data_contract import validate_dataframe_contract


class ValidationAgent:
    """Applies deterministic safety policy to a candidate dataset."""

    name = "validation_expert"

    def __init__(self, policy: PreprocessingPolicy | None = None, contract: DataContract | None = None) -> None:
        self._policy = policy or PreprocessingPolicy()
        self._contract = contract
        self.last_contract_result: ContractValidationResult | None = None

    def run(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        plan: CleaningPlan,
    ) -> ValidationResult:
        result = validate_preprocessing(before, after, plan, policy=self._policy)
        if self._contract is None:
            self.last_contract_result = None
            return result
        self.last_contract_result = validate_dataframe_contract(after, self._contract)
        contract_issues = [
            ValidationIssue(
                code=f"data_contract:{finding.rule_id}",
                severity=finding.severity,
                column=finding.column,
                message=finding.message,
            )
            for finding in self.last_contract_result.findings
            if finding.status == "failed"
        ]
        issues = [*result.issues, *contract_issues]
        has_error = any(issue.severity == "error" for issue in issues)
        return result.model_copy(update={
            "issues": issues,
            "valid": not has_error,
            "recommend_rollback": has_error,
        })
