from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

import tools.chunked as chunked
from models.chunked_transaction import ChunkedOutputValidation
from models.data_contract import ColumnContract, DataContract
from models.cleaning_plan import CleaningPlan, CleaningStep
from scripts import run_chunked_preprocessing
from tools.chunked import execute_chunked_csv_atomic
from tools.reporting import build_chunked_transaction_report


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stages(directory: Path) -> list[Path]:
    return list(directory.glob(".*.masds-stage-*.tmp"))


def make_input(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "age": [10.0, None, 30.0, 40.0],
            "city": ["A", "B", "B", "C"],
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_atomic_success_creates_new_output_and_serializable_audit(tmp_path) -> None:
    source = tmp_path / "source data.csv"
    output = tmp_path / "new output.csv"
    make_input(source)

    result = execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert result.status == "committed"
    assert result.status_history == ["pending", "staging", "validating", "committed"]
    assert output.exists()
    assert result.rows_read == 4
    assert result.rows_staged == result.rows_committed == 4
    assert result.output_fingerprint == sha256(output)
    assert result.plan_fingerprint and result.policy_fingerprint
    assert result.validation_result is not None and result.validation_result.valid
    assert result.staging_path == Path(result.staging_path).name
    assert not stages(tmp_path)
    assert "committed" in result.model_dump_json()
    report = build_chunked_transaction_report(result)
    assert "# MAS-DS Atomic Chunked Transaction Report" in report
    assert result.transaction_id in report
    assert result.output_fingerprint in report


def test_atomic_success_replaces_existing_only_after_validation(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "existing.csv"
    make_input(source)
    old = b"old,valid\n1,yes\n"
    output.write_bytes(old)
    original_validator = chunked._validate_staged_csv
    calls = 0

    def observing_validator(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert output.read_bytes() == old
        return original_validator(*args, **kwargs)

    monkeypatch.setattr(chunked, "_validate_staged_csv", observing_validator)
    result = execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert result.status == "committed"
    assert output.read_bytes() != old
    assert calls == 2
    assert not stages(tmp_path)


def test_same_resolved_source_and_output_is_rejected_without_change(tmp_path) -> None:
    source = tmp_path / "same.csv"
    make_input(source)
    before = sha256(source)

    result = execute_chunked_csv_atomic(source, source)

    assert result.status == "failed"
    assert result.failure_stage == "path_validation"
    assert result.previous_output_preserved
    assert sha256(source) == before
    assert not stages(tmp_path)


def test_midway_transformation_failure_preserves_existing_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "secret-source.csv"
    output = tmp_path / "existing.csv"
    make_input(source)
    output.write_bytes(b"stable\nold\n")
    before = sha256(output)
    original = chunked._execute_plan_on_chunk
    calls = 0

    def fail_second(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("raw-secret-value-must-not-leak")
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(chunked, "_execute_plan_on_chunk", fail_second)
    result = execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert result.status == "rolled_back"
    assert result.failure_stage == "processing"
    assert result.previous_output_preserved
    assert sha256(output) == before
    assert "raw-secret-value" not in result.model_dump_json()
    assert not stages(tmp_path)


def test_staging_write_failure_does_not_create_new_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "never-created.csv"
    make_input(source)
    original = pd.DataFrame.to_csv

    def fail_stage(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf and "masds-stage" in str(path_or_buf):
            raise OSError(28, "No space left on device")
        return original(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_stage)
    result = execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert result.status == "rolled_back"
    assert result.failure_stage == "staging_write"
    assert "No space left" in result.error_message
    assert not output.exists()
    assert result.previous_output_preserved
    assert not stages(tmp_path)


def test_malformed_input_preserves_existing_output(tmp_path) -> None:
    source = tmp_path / "truncated.csv"
    output = tmp_path / "existing.csv"
    source.write_text('a,b\n1,"unterminated\n2,x', encoding="utf-8")
    output.write_bytes(b"safe\nvalue\n")
    before = sha256(output)

    result = execute_chunked_csv_atomic(source, output, chunk_size=1)

    assert result.status == "rolled_back"
    assert sha256(output) == before
    assert result.previous_output_preserved
    assert not stages(tmp_path)


def test_error_contract_rolls_back_and_warning_contract_commits(tmp_path) -> None:
    source = tmp_path / "source.csv"
    error_output = tmp_path / "error.csv"
    warning_output = tmp_path / "warning.csv"
    pd.DataFrame({"age": [10, 500]}).to_csv(source, index=False)
    error_contract = DataContract(
        name="error", columns={"age": ColumnContract(numeric_max=100)}
    )
    warning_contract = DataContract(
        name="warning",
        columns={"age": ColumnContract(numeric_max=100, severity="warning")},
    )

    failed = execute_chunked_csv_atomic(source, error_output, contract=error_contract, chunk_size=1)
    committed = execute_chunked_csv_atomic(source, warning_output, contract=warning_contract, chunk_size=1)

    assert failed.status == "rolled_back"
    assert failed.failure_stage == "contract_validation"
    assert not error_output.exists()
    assert failed.contract_validation_result.error_count == 1
    assert committed.status == "committed"
    assert committed.contract_validation_result.warning_count == 1
    assert warning_output.exists()
    assert not stages(tmp_path)


def test_global_uniqueness_across_chunks_blocks_commit_and_preserves_old(tmp_path) -> None:
    source = tmp_path / "unique.csv"
    output = tmp_path / "existing.csv"
    pd.DataFrame({"code": ["A", "B", "C", "A"], "sequence": [1, 2, 3, 4]}).to_csv(source, index=False)
    output.write_bytes(b"previous\nvalid\n")
    before = sha256(output)
    contract = DataContract(
        name="unique", columns={"code": ColumnContract(unique=True)}
    )

    result = execute_chunked_csv_atomic(source, output, contract=contract, chunk_size=2)

    assert result.status == "rolled_back"
    assert result.contract_validation_result.error_count == 1
    assert sha256(output) == before
    assert result.previous_output_preserved


def test_existing_structural_validation_failure_preserves_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "existing.csv"
    make_input(source)
    output.write_bytes(b"old\ncontent\n")
    before = sha256(output)
    invalid = ChunkedOutputValidation(
        valid=False, readable=True, header_matches=True, row_count_matches=False,
        expected_rows=4, observed_rows=3, expected_columns=3, observed_columns=3,
        messages=["controlled row mismatch"],
    )
    monkeypatch.setattr(chunked, "_validate_staged_csv", lambda *a, **k: invalid)

    result = execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert result.status == "rolled_back"
    assert result.failure_stage == "validation"
    assert sha256(output) == before
    assert not stages(tmp_path)


def test_existing_missing_value_safety_validation_blocks_commit(tmp_path) -> None:
    source = tmp_path / "numeric_text.csv"
    output = tmp_path / "output.csv"
    pd.DataFrame({"amount": ["10", "not-numeric"]}).to_csv(source, index=False)
    plan = CleaningPlan(steps=[CleaningStep(
        operation="convert_numeric", column="amount",
        reason="controlled unsafe conversion", confidence=1.0,
    )])

    result = execute_chunked_csv_atomic(source, output, plan=plan, chunk_size=1)

    assert result.status == "rolled_back"
    assert result.failure_stage == "validation"
    assert any("missing values" in item for item in result.validation_result.messages)
    assert not output.exists()


def test_error_severity_not_evaluated_contract_rule_blocks_commit(tmp_path) -> None:
    source = tmp_path / "category.csv"
    output = tmp_path / "category_out.csv"
    pd.DataFrame({"segment": ["A", "B"]}).to_csv(source, index=False)
    contract = DataContract(
        name="category metadata",
        columns={"segment": ColumnContract(allowed_dtypes=["category"])},
    )

    result = execute_chunked_csv_atomic(source, output, contract=contract, chunk_size=1)

    assert result.status == "rolled_back"
    assert result.failure_stage == "contract_validation"
    assert result.contract_validation_result.not_evaluated_count == 1
    assert not output.exists()


def test_commit_failure_preserves_existing_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "existing.csv"
    make_input(source)
    output.write_bytes(b"old\nstable\n")
    before = sha256(output)

    def fail_replace(*args, **kwargs):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(chunked.os, "replace", fail_replace)
    result = execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert result.status == "rolled_back"
    assert result.failure_stage == "commit"
    assert result.previous_output_preserved
    assert sha256(output) == before
    assert not stages(tmp_path)


def test_cleanup_failure_is_reported_without_claiming_success(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    make_input(source)
    original_execute = chunked._execute_plan_on_chunk
    original_unlink = Path.unlink

    def fail_transform(*args, **kwargs):
        raise RuntimeError("controlled")

    def fail_stage_unlink(self, *args, **kwargs):
        if "masds-stage" in self.name:
            raise PermissionError(13, "Permission denied")
        return original_unlink(self, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(chunked, "_execute_plan_on_chunk", fail_transform)
        context.setattr(Path, "unlink", fail_stage_unlink)
        result = execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert result.status == "rolled_back"
    assert not result.rollback_cleanup_succeeded
    assert any("Could not remove staging file" in item for item in result.warnings)
    leftovers = stages(tmp_path)
    assert len(leftovers) == 1
    leftovers[0].unlink()
    assert not stages(tmp_path)
    monkeypatch.setattr(chunked, "_execute_plan_on_chunk", original_execute)


def test_keyboard_interrupt_cleans_stage_and_is_reraised(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    make_input(source)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(chunked, "_execute_plan_on_chunk", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute_chunked_csv_atomic(source, output, chunk_size=2)

    assert not output.exists()
    assert not stages(tmp_path)


def test_cli_returns_zero_only_for_commit_and_nonzero_for_contract_failure(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "source.csv"
    success_output = tmp_path / "success.csv"
    failed_output = tmp_path / "failed.csv"
    contract_path = tmp_path / "contract.json"
    pd.DataFrame({"age": [10, 500]}).to_csv(source, index=False)
    contract_path.write_text(
        DataContract(name="limit", columns={"age": ColumnContract(numeric_max=100)}).model_dump_json(),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["chunked", "--input", str(source), "--output", str(success_output), "--chunk-size", "1"])
    assert run_chunked_preprocessing.main() == 0
    success_text = capsys.readouterr().out
    assert "Final status: committed" in success_text
    assert "Transaction ID:" in success_text

    monkeypatch.setattr(sys, "argv", ["chunked", "--input", str(source), "--output", str(failed_output), "--chunk-size", "1", "--contract", str(contract_path)])
    assert run_chunked_preprocessing.main() != 0
    failure_text = capsys.readouterr().out
    assert "Final status: rolled_back" in failure_text
    assert not failed_output.exists()


def test_existing_output_lock_rejects_concurrent_writer(tmp_path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    source.write_text("value\n1\n2\n", encoding="utf-8")
    output.write_text("value\noriginal\n", encoding="utf-8")
    lock = tmp_path / ".output.csv.masds.lock"
    lock.write_text("other-transaction", encoding="utf-8")

    result = execute_chunked_csv_atomic(source, output, chunk_size=1)

    assert result.status == "failed"
    assert result.failure_stage == "output_lock"
    assert "Another chunked transaction" in (result.error_message or "")
    assert output.read_text(encoding="utf-8") == "value\noriginal\n"
    assert lock.read_text(encoding="utf-8") == "other-transaction"
