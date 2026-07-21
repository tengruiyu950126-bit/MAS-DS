"""LangGraph orchestration for profile, approval, execution, and rollback."""

from __future__ import annotations

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.cleaning_agent import CleaningPlanner, RuleBasedCleaningAgent
from agents.orchestrator import PreprocessingOutcome, PreprocessingProposal
from agents.profiling_agent import ProfilingAgent
from agents.validation_agent import ValidationAgent
from models.cleaning_plan import CleaningPlan
from models.data_contract import ContractValidationResult, DataContract
from models.policy import PreprocessingPolicy
from tools.cleaning import execute_plan
from tools.data_contract import validate_dataframe_contract
from tools.policy import policy_with_contract
from workflow.state import WorkflowState


class PreprocessingGraphOrchestrator:
    """Runs a traceable graph while preserving the UI-facing API."""

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
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("profile", self._profile_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("await_approval", self._await_approval_node)
        builder.add_node("reject", self._reject_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("validate", self._validate_node)
        builder.add_node("commit", self._commit_node)
        builder.add_node("rollback", self._rollback_node)

        builder.add_conditional_edges(
            START,
            self._route_start,
            {
                "profile": "profile",
                "execute": "execute",
                "reject": "reject",
            },
        )
        builder.add_edge("profile", "plan")
        builder.add_conditional_edges(
            "plan",
            self._route_approval,
            {
                "await_approval": "await_approval",
                "execute": "execute",
                "reject": "reject",
            },
        )
        builder.add_edge("await_approval", END)
        builder.add_edge("reject", END)
        builder.add_edge("execute", "validate")
        builder.add_conditional_edges(
            "validate",
            self._route_validation,
            {"commit": "commit", "rollback": "rollback"},
        )
        builder.add_edge("commit", END)
        builder.add_edge("rollback", END)
        return builder.compile()

    @staticmethod
    def _route_start(state: WorkflowState) -> str:
        if "cleaning_plan" not in state:
            return "profile"
        if state.get("approved") is False:
            return "reject"
        if state.get("approved") is True:
            return "execute"
        return "profile"

    @staticmethod
    def _route_approval(state: WorkflowState) -> str:
        if state.get("approved") is True:
            return "execute"
        if state.get("approved") is False:
            return "reject"
        return "await_approval"

    @staticmethod
    def _route_validation(state: WorkflowState) -> str:
        if state["validation_result"].recommend_rollback:
            return "rollback"
        return "commit"

    def _profile_node(self, state: WorkflowState) -> WorkflowState:
        return {"profile": self._profiling_agent.run(state["dataframe"])}

    def _plan_node(self, state: WorkflowState) -> WorkflowState:
        return {
            "cleaning_plan": self._cleaning_agent.propose(
                state["dataframe"],
                state["profile"],
            )
        }

    @staticmethod
    def _await_approval_node(state: WorkflowState) -> WorkflowState:
        return {"status": "awaiting_approval"}

    @staticmethod
    def _reject_node(state: WorkflowState) -> WorkflowState:
        return {
            "result": state["dataframe"].copy(deep=True),
            "rolled_back": False,
            "status": "rejected",
        }

    @staticmethod
    def _execute_node(state: WorkflowState) -> WorkflowState:
        candidate, records = execute_plan(
            state["dataframe"],
            state["cleaning_plan"],
        )
        return {
            "candidate": candidate,
            "execution_records": records,
            "status": "executed",
        }

    def _validate_node(self, state: WorkflowState) -> WorkflowState:
        validation = self._validation_agent.run(
                state["dataframe"],
                state["candidate"],
                state["cleaning_plan"],
            )
        result = {
            "validation_result": validation,
            "status": "validated",
        }
        if self._validation_agent.last_contract_result is not None:
            result["contract_validation_result"] = self._validation_agent.last_contract_result
        return result

    @staticmethod
    def _commit_node(state: WorkflowState) -> WorkflowState:
        return {
            "result": state["candidate"].copy(deep=True),
            "rolled_back": False,
            "status": "completed",
        }

    @staticmethod
    def _rollback_node(state: WorkflowState) -> WorkflowState:
        return {
            "result": state["dataframe"].copy(deep=True),
            "rolled_back": True,
            "status": "rolled_back",
        }

    def propose(self, dataframe: pd.DataFrame) -> PreprocessingProposal:
        contract_validation = (
            validate_dataframe_contract(dataframe, self._contract)
            if self._contract is not None else None
        )
        state = self.graph.invoke(
            {"dataframe": dataframe.copy(deep=True), "approved": None}
        )
        return PreprocessingProposal(
            profile=state["profile"],
            plan=state["cleaning_plan"],
            contract_validation=contract_validation,
        )

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
        state = self.graph.invoke(
            {
                "dataframe": dataframe.copy(deep=True),
                "cleaning_plan": plan,
                "approved": True,
            }
        )
        return PreprocessingOutcome(
            dataframe=state["result"],
            execution_records=state["execution_records"],
            validation=state["validation_result"],
            rolled_back=state["rolled_back"],
            contract_validation=state.get("contract_validation_result"),
            contract_caused_rollback=bool(
                state["rolled_back"] and state.get("contract_validation_result") is not None
                and state["contract_validation_result"].error_count > 0
            ),
        )
