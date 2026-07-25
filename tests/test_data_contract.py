import json

import pandas as pd
import pytest

from agents.multi_expert import CriticAgent, MultiExpertCleaningAgent, RouterAgent
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.data_contract import ColumnContract, DataContract
from models.policy import PreprocessingPolicy
from tools.data_contract import (
    ContractLoadError,
    contract_fingerprint,
    contract_from_dict,
    contract_from_json,
    validate_chunked_csv_contract,
    validate_dataframe_contract,
)
from tools.chunked import ChunkedTransactionError, execute_chunked_csv
from tools.policy import policy_with_contract
from tools.ui_tables import contract_findings_frame, contract_summary_frame
from agents.orchestrator import PreprocessingOrchestrator


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(operation=operation, column=column, reason="contract test", confidence=1.0)


def test_valid_contract_parsing_and_deterministic_fingerprint() -> None:
    payload = {
        "name": "Customer contract",
        "schema_version": "1.0",
        "required_columns": ["customer_id"],
        "protected_columns": ["customer_id"],
        "columns": {"age": {"allowed_dtypes": ["number"], "numeric_min": 0}},
    }
    first = contract_from_dict(payload)
    second = contract_from_json(json.dumps(payload))

    assert first == second
    assert contract_fingerprint(first) == contract_fingerprint(second)
    assert len(contract_fingerprint(first)) == 64


def test_invalid_contract_rejection_is_actionable_and_safe() -> None:
    with pytest.raises(ContractLoadError, match="numeric_min"):
        contract_from_dict({"name": "bad", "columns": {"age": {"numeric_min": 10, "numeric_max": 1}}})
    with pytest.raises(ContractLoadError, match="JSON is invalid"):
        contract_from_json('{"name":')
    with pytest.raises(ContractLoadError, match="JSON object"):
        contract_from_json("[]")


def test_missing_required_column_is_structural_and_blocks_execution() -> None:
    contract = DataContract(name="required", required_columns=["customer_id"], protected_columns=["customer_id"])
    result = validate_dataframe_contract(pd.DataFrame({"city": ["SG"]}), contract)

    assert not result.valid
    assert result.block_execution
    assert result.findings[0].rule_id == "required_column"
    assert result.findings[0].structural


def test_nullable_and_missing_ratio_constraints() -> None:
    contract = DataContract(name="missing", columns={
        "age": ColumnContract(nullable=False, max_missing_ratio=0.2)
    })
    result = validate_dataframe_contract(pd.DataFrame({"age": [1, None, None]}), contract)

    assert {item.rule_id for item in result.findings} == {"nullable", "missing_ratio"}
    assert result.error_count == 2
    assert not result.block_execution


def test_numeric_category_and_datetime_constraints() -> None:
    contract = DataContract(name="semantic", columns={
        "age": ColumnContract(numeric_min=0, numeric_max=100),
        "status": ColumnContract(allowed_values=["active", "inactive"]),
        "created": ColumnContract(datetime_min="2025-01-01", datetime_max="2025-12-31"),
    })
    dataframe = pd.DataFrame({
        "age": [-1, 101], "status": ["active", "secret-status"],
        "created": ["2024-12-31", "2026-01-01"],
    })
    result = validate_dataframe_contract(dataframe, contract)

    assert {item.rule_id for item in result.findings} == {
        "numeric_min", "numeric_max", "allowed_values", "datetime_min", "datetime_max"
    }
    serialized = result.model_dump_json()
    assert "secret-status" not in serialized


def test_unique_regex_and_text_length_constraints() -> None:
    contract = DataContract(name="text", columns={
        "code": ColumnContract(unique=True, regex=r"[A-Z]{3}", text_min_length=3, text_max_length=3)
    })
    result = validate_dataframe_contract(pd.DataFrame({"code": ["ABC", "ABC", "x", "TOOLONG"]}), contract)

    assert {item.rule_id for item in result.findings} == {
        "unique", "regex", "text_min_length", "text_max_length"
    }


def test_warning_severity_does_not_make_result_invalid() -> None:
    contract = DataContract(name="warning", columns={
        "age": ColumnContract(numeric_min=0, severity="warning")
    })
    result = validate_dataframe_contract(pd.DataFrame({"age": [-1]}), contract)

    assert result.valid
    assert result.warning_count == 1
    assert result.error_count == 0


def test_contract_protected_columns_merge_into_policy_and_router() -> None:
    contract = DataContract(name="protect", protected_columns=["email"])
    policy = policy_with_contract(PreprocessingPolicy(), contract)
    dataframe = pd.DataFrame({"email": [" a@example.com ", None], "amount": [1, 2]})
    routes = RouterAgent(policy).route(dataframe)

    assert "email" in policy.protected_columns
    assert [route.selected_expert for route in routes if route.column == "email"] == [None]


def test_critic_rejects_contract_conflicting_conversion() -> None:
    contract = DataContract(name="types", columns={
        "code": ColumnContract(allowed_dtypes=["string"])
    })
    dataframe = pd.DataFrame({"code": ["10", "20"]})
    proposal = step("convert_numeric", "code")
    result = CriticAgent(PreprocessingPolicy(), contract).review(dataframe, [proposal])

    assert result.rejected_steps == [proposal]
    assert any("contract" in reason for reason in result.reasons)


def test_multi_expert_contract_protection_prevents_proposals() -> None:
    contract = DataContract(name="protect", protected_columns=["account"])
    dataframe = pd.DataFrame({"account": [" A ", None], "amount": [1, 2]})
    plan = MultiExpertCleaningAgent(contract=contract).propose(dataframe)

    assert all(item.column != "account" for item in plan.steps)


def test_precleaning_result_is_attached_and_repairable_error_can_execute() -> None:
    contract = DataContract(name="non-null", columns={"age": ColumnContract(nullable=False)})
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    orchestrator = PreprocessingOrchestrator(contract=contract)
    proposal = orchestrator.propose(dataframe)
    outcome = orchestrator.execute_approved(dataframe, proposal.plan)

    assert proposal.contract_validation is not None
    assert proposal.contract_validation.error_count == 1
    assert not proposal.contract_validation.block_execution
    assert not outcome.rolled_back
    assert outcome.contract_validation is not None and outcome.contract_validation.valid


def test_postcleaning_contract_error_triggers_existing_rollback() -> None:
    contract = DataContract(name="range", columns={"age": ColumnContract(numeric_max=100)})
    dataframe = pd.DataFrame({"age": [10, 500]})
    orchestrator = PreprocessingOrchestrator(contract=contract)
    outcome = orchestrator.execute_approved(dataframe, CleaningPlan(steps=[]))

    assert outcome.rolled_back
    assert outcome.contract_caused_rollback
    assert outcome.dataframe.equals(dataframe)
    assert any(issue.code == "data_contract:numeric_max" for issue in outcome.validation.issues)


def test_warning_only_postcleaning_contract_does_not_trigger_rollback() -> None:
    contract = DataContract(name="range", columns={
        "age": ColumnContract(numeric_max=100, severity="warning")
    })
    dataframe = pd.DataFrame({"age": [10, 500]})
    outcome = PreprocessingOrchestrator(contract=contract).execute_approved(
        dataframe, CleaningPlan(steps=[])
    )

    assert not outcome.rolled_back
    assert not outcome.contract_caused_rollback
    assert outcome.contract_validation is not None and outcome.contract_validation.warning_count == 1


def test_missing_required_column_blocks_plan_execution_without_running_steps() -> None:
    contract = DataContract(name="structure", required_columns=["id"], protected_columns=["id"])
    dataframe = pd.DataFrame({"age": [1, None]})
    plan = CleaningPlan(steps=[step("fill_median", "age")])
    outcome = PreprocessingOrchestrator(contract=contract).execute_approved(dataframe, plan)

    assert outcome.rolled_back
    assert outcome.contract_caused_rollback
    assert outcome.execution_records == []
    assert outcome.dataframe.equals(dataframe)


def test_no_contract_preserves_existing_behavior() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    orchestrator = PreprocessingOrchestrator()
    proposal = orchestrator.propose(dataframe)
    outcome = orchestrator.execute_approved(dataframe, proposal.plan)

    assert proposal.contract_validation is None
    assert outcome.contract_validation is None
    assert not outcome.contract_caused_rollback
    assert not outcome.rolled_back


def test_chunked_validation_checks_global_uniqueness_across_chunks(tmp_path) -> None:
    path = tmp_path / "global.csv"
    pd.DataFrame({"code": ["A", "B", "C", "A"]}).to_csv(path, index=False)
    contract = DataContract(name="global", columns={"code": ColumnContract(unique=True)})

    result = validate_chunked_csv_contract(path, contract, chunk_size=2)

    assert any(item.rule_id == "unique" for item in result.findings)
    assert result.dataset_rows == 4


def test_chunked_category_dtype_is_explicitly_not_evaluated(tmp_path) -> None:
    path = tmp_path / "category.csv"
    pd.DataFrame({"segment": ["A", "B"]}).to_csv(path, index=False)
    contract = DataContract(name="category", columns={
        "segment": ColumnContract(allowed_dtypes=["category"])
    })

    result = validate_chunked_csv_contract(path, contract, chunk_size=1)

    assert result.not_evaluated_count == 1
    assert result.findings[0].status == "not_evaluated"


def test_chunked_execution_respects_contract_protection_and_reports_validation(tmp_path) -> None:
    input_path = tmp_path / "protected.csv"
    output_path = tmp_path / "protected_out.csv"
    pd.DataFrame({"code": [" A ", "B"]}).to_csv(input_path, index=False)
    contract = DataContract(
        name="protected chunked",
        protected_columns=["code"],
        columns={"code": ColumnContract(regex=r".*")},
    )
    plan = CleaningPlan(steps=[step("strip_whitespace", "code")])

    summary = execute_chunked_csv(
        input_path, output_path, plan=plan, contract=contract, chunk_size=1
    )
    output = pd.read_csv(output_path)

    assert output["code"].tolist() == [" A ", "B"]
    assert summary.plan_steps == 0
    assert summary.contract_name == "protected chunked"
    assert summary.contract_post_valid is True


def test_chunked_structural_contract_error_blocks_before_output(tmp_path) -> None:
    input_path = tmp_path / "missing_required.csv"
    output_path = tmp_path / "should_not_exist.csv"
    pd.DataFrame({"value": [1]}).to_csv(input_path, index=False)
    contract = DataContract(name="required", required_columns=["id"])

    with pytest.raises(ChunkedTransactionError, match="Structural data-contract"):
        execute_chunked_csv(input_path, output_path, contract=contract, chunk_size=1)

    assert not output_path.exists()


def test_contract_ui_tables_are_server_independent() -> None:
    contract = DataContract(name="ui", required_columns=["id"])
    result = validate_dataframe_contract(pd.DataFrame({"value": [1]}), contract)

    assert contract_summary_frame(contract).iloc[0]["name"] == "ui"
    assert contract_findings_frame(result).iloc[0]["rule_id"] == "required_column"
