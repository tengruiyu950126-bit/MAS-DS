"""Safe data-contract parsing, validation, and deterministic fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_categorical_dtype, is_datetime64_any_dtype,
    is_float_dtype, is_integer_dtype, is_numeric_dtype, is_object_dtype,
    is_string_dtype,
)
from pydantic import ValidationError

from models.data_contract import (
    ColumnContract, ContractFinding, ContractValidationResult, DataContract,
    SemanticType,
)


class ContractLoadError(ValueError):
    """Raised when local contract content is malformed or invalid."""


def contract_from_dict(payload: dict[str, Any]) -> DataContract:
    if not isinstance(payload, dict):
        raise ContractLoadError("Data contract must be a JSON object.")
    try:
        return DataContract.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_url=False)
        )
        raise ContractLoadError(f"Data contract is invalid: {details}") from exc


def contract_from_json(raw: str | bytes) -> DataContract:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractLoadError("Data contract must be valid UTF-8 JSON.") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractLoadError(f"Data contract JSON is invalid: {exc.msg} at line {exc.lineno}.") from exc
    return contract_from_dict(payload)


def contract_to_json(contract: DataContract) -> str:
    return contract.model_dump_json(indent=2)


def contract_fingerprint(contract: DataContract) -> str:
    canonical = json.dumps(
        contract.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _semantic_dtype_matches(series: pd.Series, expected: SemanticType) -> bool:
    if expected == "any": return True
    if expected == "string": return bool(is_string_dtype(series) or is_object_dtype(series))
    if expected == "number": return bool(is_numeric_dtype(series) and not is_bool_dtype(series))
    if expected == "integer": return bool(is_integer_dtype(series))
    if expected == "float": return bool(is_float_dtype(series))
    if expected == "boolean": return bool(is_bool_dtype(series))
    if expected == "datetime": return bool(is_datetime64_any_dtype(series))
    if expected == "category": return bool(is_categorical_dtype(series.dtype))
    return False


def _finding(rule: str, column: str | None, severity: str, expected: str, observed: str, message: str, *, structural: bool = False, status: str = "failed") -> ContractFinding:
    return ContractFinding(rule_id=rule, column=column, severity=severity, status=status, expected=expected, observed=observed, message=message, structural=structural)


def _validate_column(series: pd.Series, column: str, rule: ColumnContract) -> list[ContractFinding]:
    findings: list[ContractFinding] = []
    non_missing = series.dropna()
    missing = int(series.isna().sum())
    ratio = float(series.isna().mean()) if len(series) else 0.0
    if rule.allowed_dtypes and not any(_semantic_dtype_matches(series, item) for item in rule.allowed_dtypes):
        findings.append(_finding("dtype", column, rule.severity_for("dtype"), f"one of {rule.allowed_dtypes}", f"dtype family is {series.dtype}", f"Column {column!r} has an incompatible data type."))
    if rule.nullable is False and missing:
        findings.append(_finding("nullable", column, rule.severity_for("nullable"), "no missing values", f"{missing} missing values", f"Column {column!r} is non-nullable but contains missing values."))
    if rule.max_missing_ratio is not None and ratio > rule.max_missing_ratio:
        findings.append(_finding("missing_ratio", column, rule.severity_for("missing_ratio"), f"missing ratio <= {rule.max_missing_ratio:.4f}", f"missing ratio is {ratio:.4f}", f"Column {column!r} exceeds its maximum missing-value ratio."))
    numeric = pd.to_numeric(non_missing, errors="coerce")
    if rule.numeric_min is not None:
        count = int((numeric.dropna() < rule.numeric_min).sum())
        if count: findings.append(_finding("numeric_min", column, rule.severity_for("numeric_min"), f"values >= {rule.numeric_min}", f"{count} values below minimum", f"Column {column!r} contains values below its numeric minimum."))
    if rule.numeric_max is not None:
        count = int((numeric.dropna() > rule.numeric_max).sum())
        if count: findings.append(_finding("numeric_max", column, rule.severity_for("numeric_max"), f"values <= {rule.numeric_max}", f"{count} values above maximum", f"Column {column!r} contains values above its numeric maximum."))
    if rule.allowed_values is not None:
        count = int((~non_missing.isin(rule.allowed_values)).sum())
        if count: findings.append(_finding("allowed_values", column, rule.severity_for("allowed_values"), f"values from an allowlist of {len(rule.allowed_values)} entries", f"{count} values outside allowlist", f"Column {column!r} contains disallowed categorical values."))
    if rule.datetime_min is not None or rule.datetime_max is not None:
        parsed = pd.to_datetime(non_missing, errors="coerce", format="mixed", utc=True)
        if rule.datetime_min is not None:
            bound = pd.to_datetime(rule.datetime_min, errors="coerce", utc=True)
            count = int((parsed.dropna() < bound).sum()) if not pd.isna(bound) else 0
            if count: findings.append(_finding("datetime_min", column, rule.severity_for("datetime_min"), f"datetimes >= {rule.datetime_min}", f"{count} datetimes below minimum", f"Column {column!r} contains datetimes before its minimum."))
        if rule.datetime_max is not None:
            bound = pd.to_datetime(rule.datetime_max, errors="coerce", utc=True)
            count = int((parsed.dropna() > bound).sum()) if not pd.isna(bound) else 0
            if count: findings.append(_finding("datetime_max", column, rule.severity_for("datetime_max"), f"datetimes <= {rule.datetime_max}", f"{count} datetimes above maximum", f"Column {column!r} contains datetimes after its maximum."))
    if rule.unique:
        count = int(non_missing.duplicated(keep=False).sum())
        if count > 0: findings.append(_finding("unique", column, rule.severity_for("unique"), "all non-missing values unique", f"{count} rows participate in duplicate values", f"Column {column!r} violates its uniqueness constraint."))
    text = non_missing.map(str)
    if rule.regex is not None:
        pattern = re.compile(rule.regex)
        count = int((~text.map(lambda value: bool(pattern.fullmatch(value)))).sum())
        if count: findings.append(_finding("regex", column, rule.severity_for("regex"), "all text matches the configured regex", f"{count} values do not match", f"Column {column!r} violates its text pattern."))
    lengths = text.str.len()
    if rule.text_min_length is not None:
        count = int((lengths < rule.text_min_length).sum())
        if count: findings.append(_finding("text_min_length", column, rule.severity_for("text_min_length"), f"text length >= {rule.text_min_length}", f"{count} values are too short", f"Column {column!r} contains text shorter than allowed."))
    if rule.text_max_length is not None:
        count = int((lengths > rule.text_max_length).sum())
        if count: findings.append(_finding("text_max_length", column, rule.severity_for("text_max_length"), f"text length <= {rule.text_max_length}", f"{count} values are too long", f"Column {column!r} contains text longer than allowed."))
    return findings


def validate_dataframe_contract(dataframe: pd.DataFrame, contract: DataContract) -> ContractValidationResult:
    findings: list[ContractFinding] = []
    present = {str(column) for column in dataframe.columns}
    for column in contract.required_columns:
        if column not in present:
            findings.append(_finding("required_column", column, contract.required_columns_severity, "required column is present", "column is missing", f"Required column {column!r} is missing.", structural=True))
    if not contract.allow_extra_columns:
        declared = set(contract.required_columns) | set(contract.optional_columns) | set(contract.columns)
        for column in sorted(present - declared):
            findings.append(_finding("unexpected_column", column, contract.unexpected_columns_severity, "only declared columns", "undeclared column is present", f"Unexpected column {column!r} is present.", structural=False))
    for column, rule in contract.columns.items():
        if column in dataframe.columns:
            findings.extend(_validate_column(dataframe[column], column, rule))
    errors = sum(item.severity == "error" and item.status == "failed" for item in findings)
    warnings = sum(item.severity == "warning" and item.status == "failed" for item in findings)
    not_evaluated = sum(item.status == "not_evaluated" for item in findings)
    block = any(item.structural and item.severity == "error" for item in findings)
    return ContractValidationResult(valid=errors == 0, contract_name=contract.name, schema_version=contract.schema_version, dataset_rows=len(dataframe), dataset_columns=len(dataframe.columns), findings=findings, error_count=errors, warning_count=warnings, not_evaluated_count=not_evaluated, block_execution=block)


def validate_chunked_csv_contract(
    input_csv: str | Path,
    contract: DataContract,
    *,
    chunk_size: int = 100_000,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> ContractValidationResult:
    """Validate global CSV contract statistics without loading the full dataset."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    kwargs = dict(read_csv_kwargs or {})
    iterator = pd.read_csv(input_csv, chunksize=chunk_size, **kwargs)
    findings: list[ContractFinding] = []
    rows = 0
    columns: list[str] = []
    stats: dict[str, dict[str, Any]] = {
        column: {
            "missing": 0, "dtype_mismatch": 0, "numeric_min": 0, "numeric_max": 0,
            "allowed_values": 0, "datetime_min": 0, "datetime_max": 0,
            "regex": 0, "text_min_length": 0, "text_max_length": 0,
            "seen": set(), "duplicates": 0,
        }
        for column in contract.columns
    }
    chunks = 0
    for chunk in iterator:
        chunks += 1
        rows += len(chunk)
        if not columns:
            columns = [str(item) for item in chunk.columns]
        for column, rule in contract.columns.items():
            if column not in chunk.columns:
                continue
            series = chunk[column]
            state = stats[column]
            non_missing = series.dropna()
            state["missing"] += int(series.isna().sum())
            if rule.allowed_dtypes and "category" not in rule.allowed_dtypes and not any(_semantic_dtype_matches(series, item) for item in rule.allowed_dtypes):
                state["dtype_mismatch"] += 1
            numeric = pd.to_numeric(non_missing, errors="coerce").dropna()
            if rule.numeric_min is not None: state["numeric_min"] += int((numeric < rule.numeric_min).sum())
            if rule.numeric_max is not None: state["numeric_max"] += int((numeric > rule.numeric_max).sum())
            if rule.allowed_values is not None: state["allowed_values"] += int((~non_missing.isin(rule.allowed_values)).sum())
            if rule.datetime_min is not None or rule.datetime_max is not None:
                parsed = pd.to_datetime(non_missing, errors="coerce", format="mixed", utc=True).dropna()
                if rule.datetime_min is not None:
                    bound = pd.to_datetime(rule.datetime_min, errors="coerce", utc=True)
                    if not pd.isna(bound): state["datetime_min"] += int((parsed < bound).sum())
                if rule.datetime_max is not None:
                    bound = pd.to_datetime(rule.datetime_max, errors="coerce", utc=True)
                    if not pd.isna(bound): state["datetime_max"] += int((parsed > bound).sum())
            text = non_missing.map(str)
            if rule.regex is not None:
                pattern = re.compile(rule.regex)
                state["regex"] += int((~text.map(lambda value: bool(pattern.fullmatch(value)))).sum())
            lengths = text.str.len()
            if rule.text_min_length is not None: state["text_min_length"] += int((lengths < rule.text_min_length).sum())
            if rule.text_max_length is not None: state["text_max_length"] += int((lengths > rule.text_max_length).sum())
            if rule.unique:
                for value in non_missing.tolist():
                    digest = hashlib.sha256(json.dumps(value, default=str, ensure_ascii=False, sort_keys=True).encode("utf-8")).digest()
                    if digest in state["seen"]: state["duplicates"] += 1
                    else: state["seen"].add(digest)

    present = set(columns)
    for column in contract.required_columns:
        if column not in present:
            findings.append(_finding("required_column", column, contract.required_columns_severity, "required column is present", "column is missing", f"Required column {column!r} is missing.", structural=True))
    if not contract.allow_extra_columns:
        declared = set(contract.required_columns) | set(contract.optional_columns) | set(contract.columns)
        for column in sorted(present - declared):
            findings.append(_finding("unexpected_column", column, contract.unexpected_columns_severity, "only declared columns", "undeclared column is present", f"Unexpected column {column!r} is present."))
    for column, rule in contract.columns.items():
        if column not in present:
            continue
        state = stats[column]
        ratio = state["missing"] / rows if rows else 0.0
        if rule.allowed_dtypes and "category" in rule.allowed_dtypes:
            findings.append(_finding("dtype", column, rule.severity_for("dtype"), f"one of {rule.allowed_dtypes}", "CSV chunks do not preserve categorical dtype metadata", f"Categorical dtype for {column!r} was not evaluated in chunked CSV mode.", status="not_evaluated"))
        elif state["dtype_mismatch"]:
            findings.append(_finding("dtype", column, rule.severity_for("dtype"), f"one of {rule.allowed_dtypes}", f"{state['dtype_mismatch']} chunks had incompatible inferred dtype", f"Column {column!r} has an incompatible inferred type in chunked mode."))
        if rule.nullable is False and state["missing"]: findings.append(_finding("nullable", column, rule.severity_for("nullable"), "no missing values", f"{state['missing']} missing values", f"Column {column!r} is non-nullable but contains missing values."))
        if rule.max_missing_ratio is not None and ratio > rule.max_missing_ratio: findings.append(_finding("missing_ratio", column, rule.severity_for("missing_ratio"), f"missing ratio <= {rule.max_missing_ratio:.4f}", f"missing ratio is {ratio:.4f}", f"Column {column!r} exceeds its maximum missing-value ratio."))
        specs = {
            "numeric_min": (rule.numeric_min, f"values >= {rule.numeric_min}", "values below minimum"),
            "numeric_max": (rule.numeric_max, f"values <= {rule.numeric_max}", "values above maximum"),
            "allowed_values": (rule.allowed_values, f"values from configured allowlist", "values outside allowlist"),
            "datetime_min": (rule.datetime_min, f"datetimes >= {rule.datetime_min}", "datetimes below minimum"),
            "datetime_max": (rule.datetime_max, f"datetimes <= {rule.datetime_max}", "datetimes above maximum"),
            "regex": (rule.regex, "all text matches configured regex", "values do not match"),
            "text_min_length": (rule.text_min_length, f"text length >= {rule.text_min_length}", "values are too short"),
            "text_max_length": (rule.text_max_length, f"text length <= {rule.text_max_length}", "values are too long"),
        }
        for rule_id, (configured, expected, label) in specs.items():
            if configured is not None and state[rule_id]:
                findings.append(_finding(rule_id, column, rule.severity_for(rule_id), expected, f"{state[rule_id]} {label}", f"Column {column!r} violates its {rule_id.replace('_', ' ')} rule."))
        if rule.unique and state["duplicates"]:
            findings.append(_finding("unique", column, rule.severity_for("unique"), "all non-missing values unique across chunks", f"{state['duplicates']} repeated values", f"Column {column!r} violates its global uniqueness constraint."))
    errors = sum(item.severity == "error" and item.status == "failed" for item in findings)
    warnings = sum(item.severity == "warning" and item.status == "failed" for item in findings)
    not_evaluated = sum(item.status == "not_evaluated" for item in findings)
    block = any(item.structural and item.severity == "error" for item in findings)
    return ContractValidationResult(valid=errors == 0, contract_name=contract.name, schema_version=contract.schema_version, dataset_rows=rows, dataset_columns=len(columns), findings=findings, error_count=errors, warning_count=warnings, not_evaluated_count=not_evaluated, block_execution=block)
