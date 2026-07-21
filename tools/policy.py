"""Policy helpers shared by planners and UI."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from models.cleaning_plan import CleaningPlan, CleaningStep, Operation
from models.data_contract import DataContract
from models.policy import PreprocessingPolicy
from tools.quality import is_protected_column


DEFAULT_POLICY = PreprocessingPolicy()


def policy_with_contract(
    policy: PreprocessingPolicy | None,
    contract: DataContract | None,
) -> PreprocessingPolicy:
    """Merge contract protections into policy without mutating either model."""
    active = policy or PreprocessingPolicy()
    if contract is None:
        return active
    protected = list(dict.fromkeys([*active.protected_columns, *contract.protected_columns]))
    return active.model_copy(update={"protected_columns": protected})


class PolicyLoadError(ValueError):
    """Raised when a JSON policy cannot be parsed or validated."""


def policy_to_json(policy: PreprocessingPolicy) -> str:
    """Serialize a policy to stable, human-readable JSON."""
    return policy.model_dump_json(indent=2)


def policy_from_json(raw: str | bytes) -> PreprocessingPolicy:
    """Parse and validate a JSON policy string or bytes payload."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PolicyLoadError("Policy file must be valid UTF-8 JSON.") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyLoadError(f"Policy file is not valid JSON: {exc.msg}.") from exc
    try:
        return PreprocessingPolicy.model_validate(payload)
    except ValidationError as exc:
        raise PolicyLoadError(
            "Policy JSON does not match the PreprocessingPolicy schema."
        ) from exc


def load_policy_file(path: str | Path) -> PreprocessingPolicy:
    return policy_from_json(Path(path).read_text(encoding="utf-8"))


def write_policy_file(policy: PreprocessingPolicy, path: str | Path) -> None:
    Path(path).write_text(policy_to_json(policy), encoding="utf-8")


def normalize_column_list(raw: str) -> list[str]:
    """Parse comma/newline-separated column names from UI or config text."""
    return [
        item.strip()
        for chunk in raw.splitlines()
        for item in chunk.split(",")
        if item.strip()
    ]


def column_is_protected(column: str, policy: PreprocessingPolicy | None = None) -> bool:
    policy = policy or DEFAULT_POLICY
    if column in policy.protected_columns:
        return True
    column_policy = policy.columns.get(column)
    if column_policy and column_policy.protected:
        return True
    return bool(policy.protect_identifier_columns and is_protected_column(column))


def _allowed_by_list(
    operation: Operation,
    allowed: list[Operation] | None,
    denied: list[Operation],
) -> bool:
    if operation in denied:
        return False
    if allowed is not None and operation not in allowed:
        return False
    return True


def operation_allowed(
    step: CleaningStep,
    policy: PreprocessingPolicy | None = None,
) -> bool:
    """Return whether a cleaning step is allowed under the policy."""
    policy = policy or DEFAULT_POLICY
    if step.operation == "leave_unchanged":
        return True

    if not _allowed_by_list(
        step.operation,
        policy.allowed_operations,
        policy.denied_operations,
    ):
        return False

    if step.column is None:
        return True

    if column_is_protected(step.column, policy):
        return False

    column_policy = policy.columns.get(step.column)
    if column_policy is None:
        return True

    return _allowed_by_list(
        step.operation,
        column_policy.allowed_operations,
        column_policy.denied_operations,
    )


def outlier_action_for_column(
    column: str,
    policy: PreprocessingPolicy | None = None,
) -> str:
    policy = policy or DEFAULT_POLICY
    column_policy = policy.columns.get(column)
    if column_policy and column_policy.outlier_action is not None:
        return column_policy.outlier_action
    return policy.outlier_action


def filter_plan_by_policy(
    plan: CleaningPlan,
    policy: PreprocessingPolicy | None = None,
) -> CleaningPlan:
    return CleaningPlan(
        steps=[
            step
            for step in plan.steps
            if operation_allowed(step, policy)
        ]
    )
