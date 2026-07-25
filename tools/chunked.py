"""Chunked CSV preprocessing for large local datasets.

The regular MAS-DS workflow is dataframe-oriented and intentionally easy to
inspect. This module adds a streaming-oriented path for large CSV files:

1. Build a conservative plan from a bounded planning sample.
2. Scan the full CSV in chunks to compute global fill values and mappings.
3. Execute the plan chunk by chunk while tracking duplicates across chunks.

The implementation stays local and deterministic. It does not call a model,
does not execute generated code, and does not require a paid API.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from agents.cleaning_agent import RuleBasedCleaningAgent
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.chunked_transaction import ChunkedOutputValidation, ChunkedTransactionResult
from models.data_contract import DataContract
from models.policy import PreprocessingPolicy
from tools.cleaning import CleaningExecutionError
from tools.policy import operation_allowed, policy_with_contract
from tools.data_contract import validate_chunked_csv_contract
from tools.data_contract import contract_fingerprint
from tools.provenance import cleaning_plan_fingerprint, policy_fingerprint
from tools.quality import (
    parse_numeric_text_series,
    strip_whitespace_series,
)


TEXT_MAPPING_OPERATIONS = {"normalize_case", "normalize_category_typos"}
FILL_OPERATIONS = {"fill_mean", "fill_median", "fill_mode"}
MISSING_CELL = ("__MAS_DS_MISSING__",)


class ChunkedTransactionError(RuntimeError):
    """Raised by the compatibility API when an atomic transaction does not commit."""

    def __init__(self, result: ChunkedTransactionResult) -> None:
        self.result = result
        super().__init__(result.error_message or f"Chunked transaction {result.status}.")


class ChunkedPathError(ValueError):
    """Safe, user-actionable path preflight failure."""


@dataclass(frozen=True)
class ChunkedPreprocessingSummary:
    """Run summary for a chunked preprocessing job."""

    input_path: str
    output_path: str
    chunk_size: int
    planning_sample_rows: int
    chunks_processed: int
    rows_input: int
    rows_output: int
    rows_removed_as_duplicates: int
    columns: int
    missing_before: int
    missing_after: int
    plan_steps: int
    operations: str
    global_fill_values_json: str
    global_text_mappings: int
    elapsed_seconds: float
    contract_name: str | None = None
    contract_pre_errors: int = 0
    contract_post_errors: int = 0
    contract_post_valid: bool | None = None
    final_plan_json: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


@dataclass(frozen=True)
class ChunkedExecutionParameters:
    """Global parameters used by chunked execution."""

    fill_values: dict[str, Any]
    case_mappings: dict[str, dict[str, str]]
    typo_mappings: dict[str, dict[str, str]]
    duplicate_rows: int
    rows_input: int
    columns: int
    missing_before: int


def propose_chunked_plan(
    input_csv: str | Path,
    *,
    sample_rows: int = 100_000,
    policy: PreprocessingPolicy | None = None,
    contract: DataContract | None = None,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> CleaningPlan:
    """Propose a conservative rule-based plan from a bounded CSV sample."""
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive.")
    kwargs = dict(read_csv_kwargs or {})
    sample = pd.read_csv(input_csv, nrows=sample_rows, **kwargs)
    return RuleBasedCleaningAgent(policy=policy_with_contract(policy, contract)).propose(sample)


def _execute_chunked_csv_to_path(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    plan: CleaningPlan | None = None,
    policy: PreprocessingPolicy | None = None,
    contract: DataContract | None = None,
    chunk_size: int = 100_000,
    planning_sample_rows: int = 100_000,
    summary_csv: str | Path | None = None,
    plan_json: str | Path | None = None,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> ChunkedPreprocessingSummary:
    """Run chunked preprocessing from ``input_csv`` to ``output_csv``.

    ``plan`` may be supplied directly. When omitted, MAS-DS builds a rule-based
    plan from the first ``planning_sample_rows`` rows, then augments it with a
    duplicate-removal step if the full scan detects duplicates that the sample
    missed.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    started = time.perf_counter()
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    read_kwargs = dict(read_csv_kwargs or {})
    active_policy = policy_with_contract(policy, contract)
    pre_contract = (
        validate_chunked_csv_contract(input_path, contract, chunk_size=chunk_size, read_csv_kwargs=read_kwargs)
        if contract is not None else None
    )
    if pre_contract is not None and pre_contract.block_execution:
        raise CleaningExecutionError(
            "Chunked execution blocked by structural data-contract errors."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_plan = plan or propose_chunked_plan(
        input_path,
        sample_rows=planning_sample_rows,
        policy=active_policy,
        contract=contract,
        read_csv_kwargs=read_kwargs,
    )
    active_plan = CleaningPlan(
        steps=[
            step for step in active_plan.steps
            if operation_allowed(step, active_policy)
        ]
    )

    parameters = collect_chunked_execution_parameters(
        input_path,
        active_plan,
        chunk_size=chunk_size,
        policy=active_policy,
        read_csv_kwargs=read_kwargs,
    )
    active_plan = _ensure_duplicate_step_when_needed(
        active_plan,
        duplicate_rows=parameters.duplicate_rows,
        policy=active_policy,
    )

    chunks_processed = 0
    rows_output = 0
    missing_after = 0
    wrote_header = False
    seen_rows: set[tuple[Any, ...]] = set()

    for chunk in _read_chunks(input_path, chunk_size, read_kwargs):
        chunks_processed += 1
        processed = _execute_plan_on_chunk(chunk, active_plan, parameters, seen_rows)
        rows_output += len(processed)
        missing_after += int(processed.isna().sum().sum())
        processed.to_csv(
            output_path,
            index=False,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
        )
        wrote_header = True

    if not wrote_header:
        pd.DataFrame().to_csv(output_path, index=False)

    post_contract = (
        validate_chunked_csv_contract(output_path, contract, chunk_size=chunk_size, read_csv_kwargs=read_kwargs)
        if contract is not None else None
    )
    summary = ChunkedPreprocessingSummary(
        input_path=str(input_path.resolve()),
        output_path=str(output_path.resolve()),
        chunk_size=chunk_size,
        planning_sample_rows=planning_sample_rows,
        chunks_processed=chunks_processed,
        rows_input=parameters.rows_input,
        rows_output=rows_output,
        rows_removed_as_duplicates=parameters.rows_input - rows_output,
        columns=parameters.columns,
        missing_before=parameters.missing_before,
        missing_after=missing_after,
        plan_steps=len(active_plan.steps),
        operations=";".join(
            f"{step.operation}:{step.column or '*'}" for step in active_plan.steps
        ),
        global_fill_values_json=json.dumps(
            _json_safe_mapping(parameters.fill_values),
            ensure_ascii=False,
            sort_keys=True,
        ),
        global_text_mappings=sum(
            len(mapping)
            for collection in (parameters.case_mappings, parameters.typo_mappings)
            for mapping in collection.values()
        ),
        elapsed_seconds=round(time.perf_counter() - started, 4),
        contract_name=contract.name if contract is not None else None,
        contract_pre_errors=pre_contract.error_count if pre_contract is not None else 0,
        contract_post_errors=post_contract.error_count if post_contract is not None else 0,
        contract_post_valid=post_contract.valid if post_contract is not None else None,
        final_plan_json=active_plan.model_dump_json(),
    )

    if summary_csv is not None:
        Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_frame().to_csv(summary_csv, index=False)
    if plan_json is not None:
        Path(plan_json).parent.mkdir(parents=True, exist_ok=True)
        Path(plan_json).write_text(active_plan.model_dump_json(indent=2), encoding="utf-8")

    return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _sanitize_transaction_error(exc: Exception, *paths: Path) -> str:
    if isinstance(exc, ChunkedPathError):
        message = " ".join(str(exc).splitlines()).strip()
    elif isinstance(exc, OSError):
        detail = exc.strerror or "filesystem operation failed"
        code = f"errno {exc.errno}: " if exc.errno is not None else ""
        message = f"{code}{detail}"
    else:
        message = "operation failed safely; raw exception details were suppressed"
    for path in paths:
        message = message.replace(str(path), path.name)
    message = message[:300]
    return f"{type(exc).__name__}: {message or 'operation failed'}"


def _validate_staged_csv(
    source_path: Path,
    staged_path: Path,
    *,
    expected_rows: int,
    chunk_size: int,
    read_csv_kwargs: dict[str, Any],
) -> ChunkedOutputValidation:
    messages: list[str] = []
    readable = True
    observed_rows = 0
    observed_columns = 0
    header_matches = False
    try:
        source_header = pd.read_csv(source_path, nrows=0, **read_csv_kwargs)
        staged_header = pd.read_csv(staged_path, nrows=0, **read_csv_kwargs)
        observed_columns = len(staged_header.columns)
        header_matches = list(source_header.columns) == list(staged_header.columns)
        if not header_matches:
            messages.append("Staged CSV header does not match the source schema.")
        for chunk in pd.read_csv(
            staged_path,
            chunksize=chunk_size,
            **read_csv_kwargs,
        ):
            observed_rows += len(chunk)
            if list(chunk.columns) != list(source_header.columns):
                header_matches = False
    except Exception as exc:
        readable = False
        messages.append(_sanitize_transaction_error(exc, source_path, staged_path))
    row_count_matches = observed_rows == expected_rows
    if not row_count_matches:
        messages.append(
            f"Staged row count mismatch: expected {expected_rows}, observed {observed_rows}."
        )
    valid = readable and header_matches and row_count_matches
    return ChunkedOutputValidation(
        valid=valid,
        readable=readable,
        header_matches=header_matches,
        row_count_matches=row_count_matches,
        expected_rows=expected_rows,
        observed_rows=observed_rows,
        expected_columns=(len(source_header.columns) if "source_header" in locals() else 0),
        observed_columns=observed_columns,
        messages=messages,
    )


def _cleanup_staging_file(
    staging_path: Path,
    output_parent: Path,
    transaction_id: str,
    warnings: list[str],
) -> bool:
    try:
        if staging_path.parent.resolve() != output_parent.resolve():
            warnings.append("Refused staging cleanup outside the validated output directory.")
            return False
        if transaction_id not in staging_path.name:
            warnings.append("Refused staging cleanup because the transaction ID did not match.")
            return False
        if staging_path.exists():
            if not staging_path.is_file():
                warnings.append(f"Staging cleanup refused for non-file {staging_path.name!r}.")
                return False
            staging_path.unlink()
        return True
    except Exception as exc:
        warnings.append(
            f"Could not remove staging file {staging_path.name!r}: "
            f"{_sanitize_transaction_error(exc, staging_path)}"
        )
        return False


def execute_chunked_csv_atomic(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    plan: CleaningPlan | None = None,
    policy: PreprocessingPolicy | None = None,
    contract: DataContract | None = None,
    chunk_size: int = 100_000,
    planning_sample_rows: int = 100_000,
    summary_csv: str | Path | None = None,
    plan_json: str | Path | None = None,
    read_csv_kwargs: dict[str, Any] | None = None,
    keep_failed_staging: bool = False,
) -> ChunkedTransactionResult:
    """Stage, validate, and atomically commit one chunked CSV transaction."""
    transaction_id = uuid.uuid4().hex
    started = _utc_now()
    history: list[str] = ["pending"]
    warnings: list[str] = []
    source_raw = Path(input_csv)
    output_raw = Path(output_csv)
    source_path = source_raw.resolve(strict=False)
    output_path = output_raw.resolve(strict=False)
    lock_path = output_path.parent / f".{output_path.name}.masds.lock"
    lock_acquired = False
    staging_path: Path | None = None
    previous_hash: str | None = None
    existed = False
    rows_read = rows_staged = 0
    validation: ChunkedOutputValidation | None = None
    contract_result = None
    summary_payload: dict[str, Any] | None = None
    failure_stage: str | None = None
    error_message: str | None = None
    commit_completed = False
    active_policy = policy_with_contract(policy, contract)
    plan_hash: str | None = cleaning_plan_fingerprint(plan) if plan is not None else None
    input_hash: str | None = None

    def finish(status: str, *, cleanup_ok: bool = True, output_hash: str | None = None) -> ChunkedTransactionResult:
        nonlocal lock_acquired
        if lock_acquired:
            try:
                lock_path.unlink(missing_ok=True)
                lock_acquired = False
            except OSError:
                warnings.append(
                    "Could not remove the output lock file; manual review is required."
                )
        try:
            preserved = (
                output_path.exists() is False
                if not existed
                else output_path.is_file() and previous_hash is not None and _file_sha256(output_path) == previous_hash
            )
        except OSError as exc:
            preserved = False
            warnings.append(
                "Could not verify previous-output preservation: "
                f"{_sanitize_transaction_error(exc, output_path)}"
            )
        return ChunkedTransactionResult(
            transaction_id=transaction_id,
            source_path=str(source_path),
            requested_output_path=str(output_path),
            staging_path=staging_path.name if staging_path is not None else None,
            status=status,
            status_history=history,
            started_at_utc=started,
            completed_at_utc=_utc_now(),
            rows_read=rows_read,
            rows_staged=rows_staged,
            rows_committed=rows_staged if status == "committed" or commit_completed else 0,
            output_previously_existed=existed,
            previous_output_preserved=preserved if status != "committed" else not existed or False,
            rollback_cleanup_succeeded=cleanup_ok,
            validation_result=validation,
            contract_validation_result=contract_result,
            failure_stage=failure_stage,
            error_message=error_message,
            warnings=warnings,
            input_fingerprint=input_hash,
            output_fingerprint=output_hash,
            plan_fingerprint=plan_hash,
            policy_fingerprint=policy_fingerprint(active_policy),
            contract_fingerprint=contract_fingerprint(contract) if contract is not None else None,
            summary=summary_payload,
        )

    try:
        failure_stage = "path_validation"
        if not source_path.exists() or not source_path.is_file():
            raise ChunkedPathError("Source path must be an existing file.")
        if output_path.exists() and output_path.is_dir():
            raise ChunkedPathError("Requested output path is a directory, not a file.")
        existed = output_path.exists()
        previous_hash = _file_sha256(output_path) if existed else None
        input_hash = _file_sha256(source_path)
        if source_path == output_path or (output_path.exists() and os.path.samefile(source_path, output_path)):
            raise ChunkedPathError("Source and output paths must resolve to different files.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        failure_stage = "output_lock"
        try:
            with lock_path.open("x", encoding="utf-8") as lock_file:
                lock_file.write(transaction_id)
            lock_acquired = True
        except FileExistsError as exc:
            raise ChunkedPathError(
                "Another chunked transaction already owns this output path."
            ) from exc
        existed = output_path.exists()
        previous_hash = _file_sha256(output_path) if existed else None

        if contract is not None:
            failure_stage = "pre_contract_validation"
            contract_result = validate_chunked_csv_contract(
                source_path, contract, chunk_size=chunk_size,
                read_csv_kwargs=read_csv_kwargs,
            )
            if contract_result.block_execution:
                error_message = "Structural data-contract errors blocked chunked execution."
                history.append("failed")
                return finish("failed")

        failure_stage = "staging_create"
        staging_path = output_path.parent / (
            f".{output_path.name}.masds-stage-{transaction_id}.tmp"
        )
        with staging_path.open("xb"):
            pass
        history.append("staging")

        failure_stage = "processing"
        summary = _execute_chunked_csv_to_path(
            source_path,
            staging_path,
            plan=plan,
            policy=active_policy,
            contract=contract,
            chunk_size=chunk_size,
            planning_sample_rows=planning_sample_rows,
            summary_csv=None,
            plan_json=None,
            read_csv_kwargs=read_csv_kwargs,
        )
        rows_read = summary.rows_input
        rows_staged = summary.rows_output
        summary_payload = asdict(summary)
        summary_payload["input_path"] = str(source_path)
        summary_payload["output_path"] = str(output_path)
        final_plan = CleaningPlan.model_validate_json(summary.final_plan_json)
        plan_hash = cleaning_plan_fingerprint(final_plan)

        failure_stage = "validation"
        history.append("validating")
        validation = _validate_staged_csv(
            source_path,
            staging_path,
            expected_rows=rows_staged,
            chunk_size=chunk_size,
            read_csv_kwargs=dict(read_csv_kwargs or {}),
        )
        safety_messages = list(validation.messages)
        if rows_staged > rows_read:
            safety_messages.append("Chunked processing added rows unexpectedly.")
        if summary.missing_after > summary.missing_before:
            safety_messages.append("Chunked processing introduced unresolved missing values.")
        if len(safety_messages) != len(validation.messages):
            validation = validation.model_copy(update={
                "valid": False,
                "messages": safety_messages,
            })
        if not validation.valid:
            error_message = "Staged output failed structural validation."
            history.append("rolled_back")
            cleanup_ok = True if keep_failed_staging else _cleanup_staging_file(staging_path, output_path.parent, transaction_id, warnings)
            if keep_failed_staging:
                cleanup_ok = False
                warnings.append(f"Debug retention enabled for staging file {staging_path.name!r}.")
            return finish("rolled_back", cleanup_ok=cleanup_ok)

        if contract is not None:
            failure_stage = "contract_validation"
            contract_result = validate_chunked_csv_contract(
                staging_path, contract, chunk_size=chunk_size,
                read_csv_kwargs=read_csv_kwargs,
            )
            blocking_not_evaluated = any(
                item.status == "not_evaluated" and item.severity == "error"
                for item in contract_result.findings
            )
            if not contract_result.valid or blocking_not_evaluated:
                error_message = "Staged output failed error-level data-contract validation."
                history.append("rolled_back")
                cleanup_ok = True if keep_failed_staging else _cleanup_staging_file(staging_path, output_path.parent, transaction_id, warnings)
                if keep_failed_staging:
                    cleanup_ok = False
                    warnings.append(f"Debug retention enabled for staging file {staging_path.name!r}.")
                return finish("rolled_back", cleanup_ok=cleanup_ok)

        failure_stage = "precommit_fingerprint"
        staged_hash = _file_sha256(staging_path)
        failure_stage = "commit"
        os.replace(staging_path, output_path)
        commit_completed = True

        failure_stage = "verification"
        if not output_path.exists() or not output_path.is_file():
            raise OSError("Committed output file is unavailable after replacement.")
        committed_hash = _file_sha256(output_path)
        if committed_hash != staged_hash:
            raise OSError("Committed output fingerprint differs from staged output.")
        final_validation = _validate_staged_csv(
            source_path, output_path, expected_rows=rows_staged,
            chunk_size=chunk_size, read_csv_kwargs=dict(read_csv_kwargs or {}),
        )
        if not final_validation.valid:
            raise OSError("Committed output could not be verified as readable and complete.")

        history.append("committed")
        failure_stage = None
        if summary_csv is not None:
            try:
                Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([summary_payload]).to_csv(summary_csv, index=False)
            except Exception as exc:
                warnings.append(f"Committed output, but summary export failed: {_sanitize_transaction_error(exc, Path(summary_csv))}")
        if plan_json is not None:
            try:
                exported_plan = CleaningPlan.model_validate_json(
                    summary.final_plan_json
                )
                Path(plan_json).parent.mkdir(parents=True, exist_ok=True)
                Path(plan_json).write_text(exported_plan.model_dump_json(indent=2), encoding="utf-8")
            except Exception as exc:
                warnings.append(f"Committed output, but plan export failed: {_sanitize_transaction_error(exc, Path(plan_json))}")
        return finish("committed", output_hash=committed_hash)
    except KeyboardInterrupt:
        if staging_path is not None and not keep_failed_staging:
            _cleanup_staging_file(staging_path, output_path.parent, transaction_id, warnings)
        if lock_acquired:
            lock_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if failure_stage == "processing" and isinstance(exc, OSError):
            failure_stage = "staging_write"
        error_message = _sanitize_transaction_error(exc, source_path, output_path, *( [staging_path] if staging_path else [] ))
        status = "failed" if staging_path is None or commit_completed else "rolled_back"
        history.append(status)
        cleanup_ok = True
        if staging_path is not None:
            if keep_failed_staging:
                cleanup_ok = False
                warnings.append(f"Debug retention enabled for staging file {staging_path.name!r}.")
            else:
                cleanup_ok = _cleanup_staging_file(staging_path, output_path.parent, transaction_id, warnings)
        return finish(status, cleanup_ok=cleanup_ok)


def execute_chunked_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    **kwargs: Any,
) -> ChunkedPreprocessingSummary:
    """Backward-compatible atomic chunked execution returning the old summary."""
    result = execute_chunked_csv_atomic(input_csv, output_csv, **kwargs)
    if result.status != "committed" or result.summary is None:
        raise ChunkedTransactionError(result)
    return ChunkedPreprocessingSummary(**result.summary)


def collect_chunked_execution_parameters(
    input_csv: str | Path,
    plan: CleaningPlan,
    *,
    chunk_size: int = 100_000,
    policy: PreprocessingPolicy | None = None,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> ChunkedExecutionParameters:
    """Collect global parameters needed to execute ``plan`` by chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    read_kwargs = dict(read_csv_kwargs or {})
    active_policy = policy or PreprocessingPolicy()
    active_plan = CleaningPlan(
        steps=[step for step in plan.steps if operation_allowed(step, active_policy)]
    )

    text_columns = _columns_with_operations(active_plan, TEXT_MAPPING_OPERATIONS)
    median_columns = _columns_with_operations(active_plan, {"fill_median"})
    mean_columns = _columns_with_operations(active_plan, {"fill_mean"})

    raw_text_counts: dict[str, Counter[str]] = {
        column: Counter() for column in text_columns
    }
    numeric_values: dict[str, list[float]] = {
        column: [] for column in median_columns
    }
    numeric_sums: dict[str, float] = {column: 0.0 for column in mean_columns}
    numeric_counts: dict[str, int] = {column: 0 for column in mean_columns}

    rows_input = 0
    columns = 0
    missing_before = 0
    duplicate_rows = 0
    seen_rows: set[tuple[Any, ...]] = set()
    stats_seen_rows: set[tuple[Any, ...]] = set()
    use_deduped_stats = _drop_duplicates_precedes_global_stats(active_plan)

    for chunk in _read_chunks(input_csv, chunk_size, read_kwargs):
        rows_input += len(chunk)
        columns = len(chunk.columns)
        missing_before += int(chunk.isna().sum().sum())
        duplicate_rows += _count_duplicate_rows(chunk, seen_rows)
        stats_chunk = (
            _drop_seen_duplicates(chunk, stats_seen_rows)
            if use_deduped_stats
            else chunk
        )

        for column in text_columns:
            if column not in stats_chunk.columns:
                continue
            raw_text_counts[column].update(
                _string_values_for_mapping(stats_chunk[column])
            )

        for column in median_columns:
            if column not in stats_chunk.columns:
                continue
            step_index = _first_step_index(active_plan, column, "fill_median")
            prepared = _apply_steps_before_fill(
                stats_chunk[column],
                active_plan,
                column,
                step_index,
                case_mappings={},
                typo_mappings={},
            )
            numeric = pd.to_numeric(prepared, errors="coerce").dropna()
            numeric_values[column].extend(float(value) for value in numeric.tolist())

        for column in mean_columns:
            if column not in stats_chunk.columns:
                continue
            step_index = _first_step_index(active_plan, column, "fill_mean")
            prepared = _apply_steps_before_fill(
                stats_chunk[column],
                active_plan,
                column,
                step_index,
                case_mappings={},
                typo_mappings={},
            )
            numeric = pd.to_numeric(prepared, errors="coerce").dropna()
            numeric_sums[column] += float(numeric.sum())
            numeric_counts[column] += int(len(numeric))

    case_mappings = {
        column: _case_mapping_from_counter(counter)
        for column, counter in raw_text_counts.items()
    }
    typo_mappings = {
        column: _typo_mapping_from_counter(
            _apply_counter_mapping(raw_text_counts[column], case_mappings[column])
        )
        for column in raw_text_counts
    }

    fill_values: dict[str, Any] = {}
    for column, values in numeric_values.items():
        if values:
            fill_values[column] = float(pd.Series(values, dtype="float64").median())
    for column, total in numeric_sums.items():
        count = numeric_counts[column]
        if count:
            fill_values[column] = total / count

    mode_counters = _collect_mode_counters(
        input_csv,
        active_plan,
        chunk_size=chunk_size,
        case_mappings=case_mappings,
        typo_mappings=typo_mappings,
        read_csv_kwargs=read_kwargs,
    )
    for column, counter in mode_counters.items():
        if counter:
            fill_values[column] = _counter_mode(counter)

    return ChunkedExecutionParameters(
        fill_values=fill_values,
        case_mappings=case_mappings,
        typo_mappings=typo_mappings,
        duplicate_rows=duplicate_rows,
        rows_input=rows_input,
        columns=columns,
        missing_before=missing_before,
    )


def _read_chunks(
    input_csv: str | Path,
    chunk_size: int,
    read_csv_kwargs: dict[str, Any],
):
    yield from pd.read_csv(input_csv, chunksize=chunk_size, **read_csv_kwargs)


def _ensure_duplicate_step_when_needed(
    plan: CleaningPlan,
    *,
    duplicate_rows: int,
    policy: PreprocessingPolicy,
) -> CleaningPlan:
    if duplicate_rows <= 0:
        return plan
    if any(step.operation == "drop_duplicates" for step in plan.steps):
        return plan
    step = CleaningStep(
        operation="drop_duplicates",
        reason=(
            "Detected duplicate rows during the full chunked scan; remove exact "
            "duplicates while preserving first occurrences."
        ),
        confidence=1.0,
    )
    if not operation_allowed(step, policy):
        return plan
    return CleaningPlan(steps=[step, *plan.steps])


def _columns_with_operations(plan: CleaningPlan, operations: set[str]) -> set[str]:
    return {
        str(step.column)
        for step in plan.steps
        if step.column is not None and step.operation in operations
    }


def _drop_duplicates_precedes_global_stats(plan: CleaningPlan) -> bool:
    drop_index = _first_global_step_index(plan, {"drop_duplicates"})
    stats_index = _first_global_step_index(plan, TEXT_MAPPING_OPERATIONS | FILL_OPERATIONS)
    return drop_index < stats_index


def _first_global_step_index(plan: CleaningPlan, operations: set[str]) -> int:
    for index, step in enumerate(plan.steps):
        if step.operation in operations:
            return index
    return len(plan.steps)


def _first_step_index(plan: CleaningPlan, column: str, operation: str) -> int:
    for index, step in enumerate(plan.steps):
        if step.column == column and step.operation == operation:
            return index
    return len(plan.steps)


def _count_duplicate_rows(
    chunk: pd.DataFrame,
    seen_rows: set[tuple[Any, ...]],
) -> int:
    duplicates = 0
    local_seen: set[tuple[Any, ...]] = set()
    for row in _row_keys(chunk):
        if row in seen_rows or row in local_seen:
            duplicates += 1
        else:
            local_seen.add(row)
    seen_rows.update(local_seen)
    return duplicates


def _drop_seen_duplicates(
    chunk: pd.DataFrame,
    seen_rows: set[tuple[Any, ...]],
) -> pd.DataFrame:
    keep: list[bool] = []
    local_seen: set[tuple[Any, ...]] = set()
    for row in _row_keys(chunk):
        if row in seen_rows or row in local_seen:
            keep.append(False)
        else:
            keep.append(True)
            local_seen.add(row)
    seen_rows.update(local_seen)
    return chunk.loc[keep].copy()


def _row_keys(chunk: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [
        tuple(_canonical_cell(value) for value in row)
        for row in chunk.itertuples(index=False, name=None)
    ]


def _canonical_cell(value: Any) -> Any:
    try:
        if bool(pd.isna(value)):
            return MISSING_CELL
    except (TypeError, ValueError):
        pass
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _execute_plan_on_chunk(
    chunk: pd.DataFrame,
    plan: CleaningPlan,
    parameters: ChunkedExecutionParameters,
    seen_rows: set[tuple[Any, ...]],
) -> pd.DataFrame:
    result = chunk.copy(deep=True)
    for step in plan.steps:
        if step.operation == "leave_unchanged":
            continue
        if step.operation == "drop_duplicates":
            result = _drop_seen_duplicates(result, seen_rows)
            result = result.reset_index(drop=True)
            continue

        column = _require_column(result, step)
        if step.operation in {"fill_mean", "fill_median", "fill_mode"}:
            if column not in parameters.fill_values:
                raise CleaningExecutionError(
                    f"Cannot calculate a global fill value for {column!r}."
                )
            result[column] = result[column].fillna(parameters.fill_values[column])
            continue
        if step.operation == "convert_numeric":
            result[column] = pd.to_numeric(result[column], errors="coerce")
            continue
        if step.operation == "convert_datetime":
            result[column] = pd.to_datetime(result[column], errors="coerce")
            continue
        if step.operation == "parse_numeric_text":
            result[column] = parse_numeric_text_series(result[column])
            continue
        if step.operation == "strip_whitespace":
            result[column] = strip_whitespace_series(result[column])
            continue
        if step.operation == "normalize_case":
            result[column] = _apply_case_mapping(
                result[column],
                parameters.case_mappings.get(column, {}),
            )
            continue
        if step.operation == "normalize_category_typos":
            normalized = _apply_case_mapping(
                result[column],
                parameters.case_mappings.get(column, {}),
            )
            result[column] = _apply_value_mapping(
                normalized,
                parameters.typo_mappings.get(column, {}),
            )
            continue
        if step.operation == "flag_outliers_iqr":
            # Advisory operation. The dataframe remains unchanged.
            continue
        raise CleaningExecutionError(f"Unsupported operation: {step.operation!r}.")
    return result


def _require_column(dataframe: pd.DataFrame, step: CleaningStep) -> str:
    if not step.column:
        raise CleaningExecutionError(f"Operation '{step.operation}' requires a column.")
    if step.column not in dataframe.columns:
        raise CleaningExecutionError(f"Unknown column: {step.column!r}.")
    return step.column


def _apply_steps_before_fill(
    series: pd.Series,
    plan: CleaningPlan,
    column: str,
    fill_step_index: int,
    *,
    case_mappings: dict[str, dict[str, str]],
    typo_mappings: dict[str, dict[str, str]],
) -> pd.Series:
    result = series.copy(deep=True)
    for index, step in enumerate(plan.steps):
        if index >= fill_step_index:
            break
        if step.column != column:
            continue
        if step.operation == "convert_numeric":
            result = pd.to_numeric(result, errors="coerce")
        elif step.operation == "convert_datetime":
            result = pd.to_datetime(result, errors="coerce")
        elif step.operation == "parse_numeric_text":
            result = parse_numeric_text_series(result)
        elif step.operation == "strip_whitespace":
            result = strip_whitespace_series(result)
        elif step.operation == "normalize_case":
            result = _apply_case_mapping(result, case_mappings.get(column, {}))
        elif step.operation == "normalize_category_typos":
            normalized = _apply_case_mapping(result, case_mappings.get(column, {}))
            result = _apply_value_mapping(normalized, typo_mappings.get(column, {}))
    return result


def _collect_mode_counters(
    input_csv: str | Path,
    plan: CleaningPlan,
    *,
    chunk_size: int,
    case_mappings: dict[str, dict[str, str]],
    typo_mappings: dict[str, dict[str, str]],
    read_csv_kwargs: dict[str, Any],
) -> dict[str, Counter[Any]]:
    mode_columns = _columns_with_operations(plan, {"fill_mode"})
    counters: dict[str, Counter[Any]] = {column: Counter() for column in mode_columns}
    if not mode_columns:
        return counters

    mode_index = min(_first_step_index(plan, column, "fill_mode") for column in mode_columns)
    use_deduped_stats = _first_global_step_index(plan, {"drop_duplicates"}) < mode_index
    stats_seen_rows: set[tuple[Any, ...]] = set()

    for chunk in _read_chunks(input_csv, chunk_size, read_csv_kwargs):
        stats_chunk = (
            _drop_seen_duplicates(chunk, stats_seen_rows)
            if use_deduped_stats
            else chunk
        )
        for column in mode_columns:
            if column not in stats_chunk.columns:
                continue
            step_index = _first_step_index(plan, column, "fill_mode")
            prepared = _apply_steps_before_fill(
                stats_chunk[column],
                plan,
                column,
                step_index,
                case_mappings=case_mappings,
                typo_mappings=typo_mappings,
            )
            counters[column].update(prepared.dropna().tolist())
    return counters


def _string_values_for_mapping(series: pd.Series) -> list[str]:
    values: list[str] = []
    for value in series.dropna().tolist():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                values.append(stripped)
    return values


def _style_score(value: str) -> int:
    if len(value) <= 5 and value.isupper():
        return 4
    if value.istitle():
        return 3
    if not value.islower() and not value.isupper():
        return 2
    if value.isupper():
        return 1
    return 0


def _preferred_variant_from_counter(counter: Counter[str]) -> str:
    return max(
        counter,
        key=lambda value: (counter[value], _style_score(value), -len(value), value),
    )


def _case_mapping_from_counter(counter: Counter[str]) -> dict[str, str]:
    groups: dict[str, Counter[str]] = {}
    for value, count in counter.items():
        groups.setdefault(value.casefold(), Counter())[value] += count
    return {
        variant: preferred
        for variants in groups.values()
        for preferred in [_preferred_variant_from_counter(variants)]
        for variant in variants
        if variant != preferred
    }


def _apply_counter_mapping(
    counter: Counter[str],
    mapping: dict[str, str],
) -> Counter[str]:
    mapped: Counter[str] = Counter()
    for value, count in counter.items():
        mapped[mapping.get(value, value)] += count
    return mapped


def _typo_mapping_from_counter(counter: Counter[str]) -> dict[str, str]:
    if len(counter) < 2:
        return {}
    common = sorted(counter, key=lambda value: (-counter[value], value))
    mapping: dict[str, str] = {}
    for candidate, candidate_count in counter.items():
        if candidate_count > 1 or len(candidate) < 5:
            continue
        for target in common:
            if target == candidate:
                continue
            if counter[target] < 2 or len(target) < 5:
                continue
            if target[0].casefold() != candidate[0].casefold():
                continue
            if abs(len(target) - len(candidate)) > 2:
                continue
            similarity = SequenceMatcher(
                None,
                target.casefold(),
                candidate.casefold(),
            ).ratio()
            if similarity >= 0.88:
                mapping[candidate] = target
                break
    return mapping


def _apply_case_mapping(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    stripped = strip_whitespace_series(series)
    if not mapping:
        return stripped
    return stripped.map(
        lambda value: mapping.get(value, value) if isinstance(value, str) else value
    )


def _apply_value_mapping(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    if not mapping:
        return series
    return series.map(
        lambda value: mapping.get(value, value) if isinstance(value, str) else value
    )


def _counter_mode(counter: Counter[Any]) -> Any:
    max_count = max(counter.values())
    candidates = [value for value, count in counter.items() if count == max_count]
    return sorted(candidates, key=lambda value: str(value))[0]


def _json_safe_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in mapping.items()}


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
