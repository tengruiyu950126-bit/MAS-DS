from datetime import datetime

import pandas as pd

from models.cleaning_plan import CleaningPlan, CleaningStep
from models.data_contract import ColumnContract, DataContract
from models.policy import PreprocessingPolicy
from tools.cleaning import execute_plan
from tools.reporting import build_cleaning_report, build_report_filename
from tools.provenance import build_audit_provenance
from tools.validation import apply_rollback_policy, validate_preprocessing
from tools.data_contract import validate_dataframe_contract


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="report test",
        confidence=0.9,
    )


def test_cleaning_report_contains_core_audit_sections() -> None:
    before = pd.DataFrame({"age": [10.0, None, 30.0], "city": ["A", "B", "A"]})
    plan = CleaningPlan(steps=[step("fill_median", "age")])
    after, records = execute_plan(before, plan)
    validation = validate_preprocessing(before, after, plan)

    report = build_cleaning_report(
        before=before,
        after=after,
        plan=plan,
        validation=validation,
        execution_records=records,
        policy=PreprocessingPolicy(protected_columns=["city"]),
        planner_name="Rule-based baseline",
        source_label="sample:demo:dirty",
        rolled_back=False,
        generated_at=datetime(2026, 7, 11, 10, 0, 0),
    )

    assert "# MAS-DS Cleaning Report" in report
    assert "Generated at: 2026-07-11 10:00:00" in report
    assert "Rule-based baseline" in report
    assert "sample:demo:dirty" in report
    assert "## Active safety policy" in report
    assert "## Cleaning plan" in report
    assert "## Validation" in report
    assert "## Change audit" in report
    assert "fill_median" in report
    assert "value_changed" in report
    assert "<missing>" in report


def test_cleaning_report_explains_rollback_candidate_changes() -> None:
    before = pd.DataFrame({"city": ["Singapore", "Bangkok", "Jakarta"]})
    plan = CleaningPlan(steps=[step("convert_numeric", "city")])
    candidate, records = execute_plan(before, plan)
    validation = validate_preprocessing(before, candidate, plan)
    committed, rolled_back = apply_rollback_policy(before, candidate, validation)

    report = build_cleaning_report(
        before=before,
        after=committed,
        plan=plan,
        validation=validation,
        execution_records=records,
        policy=PreprocessingPolicy(),
        planner_name="Unsafe test planner",
        source_label="upload:risky.csv",
        rolled_back=rolled_back,
        generated_at=datetime(2026, 7, 11, 10, 0, 0),
    )

    assert rolled_back
    assert "rolled back" in report
    assert "candidate changes attempted" in report
    assert "dtype_changed" in report
    assert "value_changed" in report


def test_report_filename_is_safe_for_download() -> None:
    assert (
        build_report_filename("sample:demo dirty/file.csv")
        == "mas_ds_cleaning_report_sample_demo_dirty_file_csv.md"
    )
    assert build_report_filename(" ") == "mas_ds_cleaning_report_run.md"


def test_cleaning_report_includes_optional_audit_provenance() -> None:
    before = pd.DataFrame({"age": [10.0, None, 30.0]})
    plan = CleaningPlan(steps=[step("fill_median", "age")])
    policy = PreprocessingPolicy()
    after, records = execute_plan(before, plan)
    validation = validate_preprocessing(before, after, plan)
    provenance = build_audit_provenance(
        dataframe=before,
        plan=plan,
        policy=policy,
        source_label="sample:demo:dirty",
        planner_name="Rule-based baseline",
        planner_mode="Rule-based baseline",
        created_at_utc=datetime(2026, 7, 20, 12, 0, 0),
    )

    report = build_cleaning_report(
        before=before,
        after=after,
        plan=plan,
        validation=validation,
        execution_records=records,
        policy=policy,
        planner_name="Rule-based baseline",
        source_label="sample:demo:dirty",
        rolled_back=False,
        provenance=provenance,
    )

    assert "## Audit provenance" in report
    assert provenance.dataset_fingerprint in report
    assert provenance.cleaning_plan_fingerprint in report
    assert provenance.policy_fingerprint in report
    assert "Rule-based baseline" in report


def test_cleaning_report_includes_data_contract_audit() -> None:
    before = pd.DataFrame({"age": [10, 500]})
    plan = CleaningPlan(steps=[])
    policy = PreprocessingPolicy()
    contract = DataContract(name="Age contract", schema_version="2.0", columns={
        "age": ColumnContract(numeric_max=100)
    })
    validation = validate_preprocessing(before, before.copy(), plan)
    contract_result = validate_dataframe_contract(before, contract)
    provenance = build_audit_provenance(
        dataframe=before, plan=plan, policy=policy, contract=contract
    )

    report = build_cleaning_report(
        before=before, after=before, plan=plan, validation=validation,
        execution_records=[], policy=policy, planner_name="Multi-Expert",
        source_label="sample:age", rolled_back=True, provenance=provenance,
        contract=contract, pre_contract_validation=contract_result,
        post_contract_validation=contract_result, contract_caused_rollback=True,
    )

    assert "## Data contract" in report
    assert "Age contract" in report
    assert "2.0" in report
    assert provenance.contract_fingerprint in report
    assert "contract_caused_rollback" in report
