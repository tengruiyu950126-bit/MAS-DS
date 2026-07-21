"""Local-LLM cleaning expert with strict structured-output validation."""

from __future__ import annotations

import json
from typing import Any, Protocol

import pandas as pd
from pydantic import ValidationError

from models.cleaning_plan import CleaningPlan
from models.policy import PreprocessingPolicy
from models.profile import DataProfile
from tools.policy import filter_plan_by_policy
from tools.profiler import profile_dataframe


class JSONModelClient(Protocol):
    model: str

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class LocalLLMPlanningError(ValueError):
    """Raised when a model proposes an invalid or unsafe plan."""


SYSTEM_PROMPT = """You are a conservative tabular-data cleaning expert.
Return exactly one JSON object with a `steps` array. Each step must contain:
`column`, `operation`, `reason`, and `confidence`.

Allowed operations only:
- drop_duplicates (column must be null)
- fill_mean
- fill_median
- fill_mode
- convert_numeric
- convert_datetime
- parse_numeric_text
- strip_whitespace
- normalize_case
- normalize_category_typos
- flag_outliers_iqr
- leave_unchanged

Never output Python, SQL, markdown, or an operation outside this list.
Prefer leave_unchanged when evidence is insufficient. Do not invent values.
For column operations, use a column name exactly as provided in the profile.
Do not clean identifier-like columns such as id, customer_id, order_id, or uuid.
Use confidence from 0.0 to 1.0.
"""


class LocalLLMCleaningAgent:
    """Builds cleaning plans with a local model while preserving hard guards."""

    name = "local_llm_cleaning_expert"

    def __init__(
        self,
        client: JSONModelClient,
        sample_rows: int = 5,
        policy: PreprocessingPolicy | None = None,
    ) -> None:
        if sample_rows < 0 or sample_rows > 20:
            raise ValueError("sample_rows must be between 0 and 20.")
        self._client = client
        self._sample_rows = sample_rows
        self._policy = policy or PreprocessingPolicy()

    @property
    def model_name(self) -> str:
        return self._client.model

    def propose(
        self,
        dataframe: pd.DataFrame,
        profile: DataProfile | None = None,
    ) -> CleaningPlan:
        profile = profile or profile_dataframe(dataframe)
        sample_json = dataframe.head(self._sample_rows).to_json(
            orient="records",
            date_format="iso",
        )
        user_payload = {
            "profile": profile.model_dump(mode="json"),
            "sample_rows": json.loads(sample_json),
        }

        user_prompt = (
            "Analyze the dataset evidence below and propose a conservative "
            "cleaning plan. Do not repeat the input. Return only a CleaningPlan "
            "object whose top-level key is `steps`.\n\nDATASET EVIDENCE:\n"
            + json.dumps(user_payload, ensure_ascii=False)
        )
        raw_plan = self._client.generate_json(
            SYSTEM_PROMPT,
            user_prompt,
            json_schema=self._build_json_schema(list(map(str, dataframe.columns))),
        )
        try:
            plan = CleaningPlan.model_validate(raw_plan)
        except ValidationError as exc:
            raise LocalLLMPlanningError(
                "Local model returned a plan that violates the schema."
            ) from exc

        self._validate_semantics(plan, set(map(str, dataframe.columns)))
        return filter_plan_by_policy(plan, self._policy)

    @staticmethod
    def _build_json_schema(columns: list[str]) -> dict[str, Any]:
        """Build a grammar that prevents unknown or missing column names."""
        if not columns:
            return {
                "type": "object",
                "properties": {"steps": {"type": "array", "maxItems": 0}},
                "required": ["steps"],
                "additionalProperties": False,
            }

        reason = {"type": "string", "minLength": 1}
        confidence = {"type": "number", "minimum": 0, "maximum": 1}
        required = ["column", "operation", "reason", "confidence"]
        step_variants = [
            {
                "type": "object",
                "properties": {
                    "column": {"type": "null"},
                    "operation": {"const": "drop_duplicates"},
                    "reason": reason,
                    "confidence": confidence,
                },
                "required": required,
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "enum": columns},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "fill_mean",
                            "fill_median",
                            "fill_mode",
                            "convert_numeric",
                            "convert_datetime",
                            "parse_numeric_text",
                            "strip_whitespace",
                            "normalize_case",
                            "normalize_category_typos",
                            "flag_outliers_iqr",
                        ],
                    },
                    "reason": reason,
                    "confidence": confidence,
                },
                "required": required,
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "column": {
                        "anyOf": [
                            {"type": "string", "enum": columns},
                            {"type": "null"},
                        ]
                    },
                    "operation": {"const": "leave_unchanged"},
                    "reason": reason,
                    "confidence": confidence,
                },
                "required": required,
                "additionalProperties": False,
            },
        ]
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"oneOf": step_variants},
                    "maxItems": max(20, len(columns) * 2 + 1),
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        }

    @staticmethod
    def _validate_semantics(plan: CleaningPlan, columns: set[str]) -> None:
        if len(plan.steps) > max(20, len(columns) * 2 + 1):
            raise LocalLLMPlanningError("Local model proposed too many steps.")

        seen: set[tuple[str, str | None]] = set()
        column_operations = {
            "fill_mean",
            "fill_median",
            "fill_mode",
            "convert_numeric",
            "convert_datetime",
            "parse_numeric_text",
            "strip_whitespace",
            "normalize_case",
            "normalize_category_typos",
            "flag_outliers_iqr",
        }
        for step in plan.steps:
            key = (step.operation, step.column)
            if key in seen:
                raise LocalLLMPlanningError(f"Duplicate plan step: {key!r}.")
            seen.add(key)

            if step.operation == "drop_duplicates" and step.column is not None:
                raise LocalLLMPlanningError(
                    "drop_duplicates must apply to full rows (column=null)."
                )
            if step.operation in column_operations:
                if not step.column or step.column not in columns:
                    raise LocalLLMPlanningError(
                        f"Plan references an unknown column: {step.column!r}."
                    )
            if step.operation == "leave_unchanged":
                if step.column is not None and step.column not in columns:
                    raise LocalLLMPlanningError(
                        f"Plan references an unknown column: {step.column!r}."
                    )
