"""Authoritative in-memory preprocessing lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

import pandas as pd

from agents.cleaning_agent import CleaningPlanner, RuleBasedCleaningAgent
from models.cleaning_plan import CleaningPlan
from models.data_contract import ContractValidationResult, DataContract
from models.policy import PreprocessingPolicy
from models.profile import DataProfile
from models.validation import ValidationIssue, ValidationResult
from tools.cleaning import ExecutionRecord, execute_plan
from tools.validation import apply_rollback_policy, validate_preprocessing
from tools.data_contract import validate_dataframe_contract
from tools.policy import filter_plan_by_policy, policy_with_contract
from tools.profiler import profile_dataframe
from tools.observability import log_event


@dataclass(frozen=True)
class PreprocessingProposal:
    run_id: str
    profile: DataProfile
    plan: CleaningPlan
    contract_validation: ContractValidationResult | None = None


@dataclass(frozen=True)
class PreprocessingOutcome:
    run_id: str
    dataframe: pd.DataFrame
    execution_records: list[ExecutionRecord]
    validation: ValidationResult
    rolled_back: bool
    contract_validation: ContractValidationResult | None = None
    contract_caused_rollback: bool = False


class PreprocessingOrchestrator:
    """Own profiling, planning, deterministic execution, and rollback."""

    def __init__(
        self,
        cleaning_agent: CleaningPlanner | None = None,
        policy: PreprocessingPolicy | None = None,
        contract: DataContract | None = None,
    ) -> None:
        self._contract = contract
        self._policy = policy_with_contract(policy, contract)
        self._cleaning_agent = cleaning_agent or RuleBasedCleaningAgent(
            policy=self._policy,
        )

    def propose(self, dataframe: pd.DataFrame) -> PreprocessingProposal:
        run_id = str(uuid4())
        started = perf_counter()
        log_event(
            "planning_started",
            run_id=run_id,
            rows=len(dataframe),
            columns=len(dataframe.columns),
            planner=type(self._cleaning_agent).__name__,
        )
        profile = profile_dataframe(dataframe)
        plan = self._cleaning_agent.propose(dataframe.copy(deep=True), profile)
        contract_validation = (
            validate_dataframe_contract(dataframe, self._contract)
            if self._contract is not None else None
        )
        log_event(
            "planning_completed",
            run_id=run_id,
            operation_count=len(plan.steps),
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        return PreprocessingProposal(
            run_id=run_id,
            profile=profile,
            plan=plan,
            contract_validation=contract_validation,
        )

    def execute_approved(
        self,
        dataframe: pd.DataFrame,
        plan: CleaningPlan,
        *,
        run_id: str | None = None,
    ) -> PreprocessingOutcome:
        run_id = run_id or str(uuid4())
        started = perf_counter()
        log_event(
            "execution_started",
            run_id=run_id,
            rows=len(dataframe),
            columns=len(dataframe.columns),
            operation_count=len(plan.steps),
        )
        pre_contract = (
            validate_dataframe_contract(dataframe, self._contract)
            if self._contract is not None else None
        )
        approved_plan = filter_plan_by_policy(plan, self._policy)
        if approved_plan != plan:
            validation, _ = self._validate_candidate(
                dataframe,
                dataframe.copy(deep=True),
                approved_plan,
            )
            validation = validation.model_copy(
                update={
                    "issues": [
                        *validation.issues,
                        ValidationIssue(
                            code="policy_rejected_operation",
                            severity="error",
                            message=(
                                "The approved plan contains an operation denied "
                                "by the active preprocessing policy."
                            ),
                        ),
                    ],
                    "valid": False,
                    "recommend_rollback": True,
                }
            )
            log_event(
                "execution_completed",
                run_id=run_id,
                rolled_back=True,
                validation_valid=False,
                validation_issue_count=len(validation.issues),
                contract_error_count=(
                    pre_contract.error_count if pre_contract is not None else 0
                ),
                duration_ms=round((perf_counter() - started) * 1000, 3),
                stop_reason="policy_rejected",
            )
            return PreprocessingOutcome(
                run_id=run_id,
                dataframe=dataframe.copy(deep=True),
                execution_records=[],
                validation=validation,
                rolled_back=True,
                contract_validation=pre_contract,
                contract_caused_rollback=False,
            )
        if pre_contract is not None and pre_contract.block_execution:
            validation, _ = self._validate_candidate(
                dataframe,
                dataframe.copy(deep=True),
                plan,
            )
            return PreprocessingOutcome(
                run_id=run_id,
                dataframe=dataframe.copy(deep=True), execution_records=[], validation=validation,
                rolled_back=True, contract_validation=pre_contract,
                contract_caused_rollback=True,
            )
        candidate, records = execute_plan(dataframe, plan)
        validation, contract_validation = self._validate_candidate(
            dataframe,
            candidate,
            plan,
        )
        committed, rolled_back = apply_rollback_policy(
            dataframe,
            candidate,
            validation,
        )
        outcome = PreprocessingOutcome(
            run_id=run_id,
            dataframe=committed,
            execution_records=records,
            validation=validation,
            rolled_back=rolled_back,
            contract_validation=contract_validation,
            contract_caused_rollback=bool(
                rolled_back
                and contract_validation is not None
                and contract_validation.error_count > 0
            ),
        )
        log_event(
            "execution_completed",
            run_id=run_id,
            rolled_back=rolled_back,
            validation_valid=validation.valid,
            validation_issue_count=len(validation.issues),
            contract_error_count=(
                contract_validation.error_count
                if contract_validation is not None
                else 0
            ),
            duration_ms=round((perf_counter() - started) * 1000, 3),
            stop_reason="validation_failed" if rolled_back else "completed",
        )
        return outcome

    def _validate_candidate(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        plan: CleaningPlan,
    ) -> tuple[ValidationResult, ContractValidationResult | None]:
        validation = validate_preprocessing(
            before,
            after,
            plan,
            policy=self._policy,
        )
        if self._contract is None:
            return validation, None
        contract_validation = validate_dataframe_contract(after, self._contract)
        contract_issues = [
            ValidationIssue(
                code=f"data_contract:{finding.rule_id}",
                severity=finding.severity,
                column=finding.column,
                message=finding.message,
            )
            for finding in contract_validation.findings
            if finding.status == "failed"
        ]
        issues = [*validation.issues, *contract_issues]
        has_error = any(issue.severity == "error" for issue in issues)
        return (
            validation.model_copy(
                update={
                    "issues": issues,
                    "valid": not has_error,
                    "recommend_rollback": has_error,
                }
            ),
            contract_validation,
        )
