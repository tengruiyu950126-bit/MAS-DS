"""Coordinates profiling, planning, execution, and validation experts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.cleaning_agent import CleaningPlanner, RuleBasedCleaningAgent
from agents.profiling_agent import ProfilingAgent
from agents.validation_agent import ValidationAgent
from models.cleaning_plan import CleaningPlan
from models.data_contract import ContractValidationResult, DataContract
from models.policy import PreprocessingPolicy
from models.profile import DataProfile
from models.validation import ValidationResult
from tools.cleaning import ExecutionRecord, execute_plan
from tools.validation import apply_rollback_policy
from tools.data_contract import validate_dataframe_contract
from tools.policy import policy_with_contract


@dataclass(frozen=True)
class PreprocessingProposal:
    profile: DataProfile
    plan: CleaningPlan
    contract_validation: ContractValidationResult | None = None


@dataclass(frozen=True)
class PreprocessingOutcome:
    dataframe: pd.DataFrame
    execution_records: list[ExecutionRecord]
    validation: ValidationResult
    rolled_back: bool
    contract_validation: ContractValidationResult | None = None
    contract_caused_rollback: bool = False


class PreprocessingOrchestrator:
    """Small orchestration facade used by the UI and future graph workflow."""

    def __init__(
        self,
        profiling_agent: ProfilingAgent | None = None,
        cleaning_agent: CleaningPlanner | None = None,
        validation_agent: ValidationAgent | None = None,
        policy: PreprocessingPolicy | None = None,
        contract: DataContract | None = None,
    ) -> None:
        self._contract = contract
        self._policy = policy_with_contract(policy, contract)
        self._profiling_agent = profiling_agent or ProfilingAgent()
        self._cleaning_agent = cleaning_agent or RuleBasedCleaningAgent(
            policy=self._policy,
        )
        self._validation_agent = validation_agent or ValidationAgent(
            policy=self._policy,
            contract=contract,
        )

    def propose(self, dataframe: pd.DataFrame) -> PreprocessingProposal:
        profile = self._profiling_agent.run(dataframe)
        plan = self._cleaning_agent.propose(dataframe, profile)
        contract_validation = (
            validate_dataframe_contract(dataframe, self._contract)
            if self._contract is not None else None
        )
        return PreprocessingProposal(profile=profile, plan=plan, contract_validation=contract_validation)

    def execute_approved(
        self,
        dataframe: pd.DataFrame,
        plan: CleaningPlan,
    ) -> PreprocessingOutcome:
        pre_contract = (
            validate_dataframe_contract(dataframe, self._contract)
            if self._contract is not None else None
        )
        if pre_contract is not None and pre_contract.block_execution:
            validation = self._validation_agent.run(dataframe, dataframe.copy(deep=True), plan)
            return PreprocessingOutcome(
                dataframe=dataframe.copy(deep=True), execution_records=[], validation=validation,
                rolled_back=True, contract_validation=pre_contract,
                contract_caused_rollback=True,
            )
        candidate, records = execute_plan(dataframe, plan)
        validation = self._validation_agent.run(dataframe, candidate, plan)
        committed, rolled_back = apply_rollback_policy(
            dataframe,
            candidate,
            validation,
        )
        return PreprocessingOutcome(
            dataframe=committed,
            execution_records=records,
            validation=validation,
            rolled_back=rolled_back,
            contract_validation=self._validation_agent.last_contract_result,
            contract_caused_rollback=bool(
                rolled_back and self._validation_agent.last_contract_result is not None
                and self._validation_agent.last_contract_result.error_count > 0
            ),
        )
