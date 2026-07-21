"""Hybrid planning that combines deterministic and local-LLM experts."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from agents.cleaning_agent import CleaningPlanner, RuleBasedCleaningAgent
from agents.local_llm_cleaning_agent import LocalLLMPlanningError
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.policy import PreprocessingPolicy
from models.profile import DataProfile
from providers.ollama import OllamaError
from tools.policy import filter_plan_by_policy, operation_allowed
from tools.profiler import profile_dataframe
from tools.quality import (
    can_parse_numeric_text_losslessly,
    has_case_inconsistency,
    has_category_typo_issue,
    has_whitespace_issue,
    has_iqr_outliers,
)


FILL_OPERATIONS = {"fill_mean", "fill_median", "fill_mode"}


class HybridCleaningAgent:
    """Preserve deterministic safeguards while accepting useful LLM steps."""

    name = "hybrid_rule_llm_cleaning_expert"

    def __init__(
        self,
        llm_agent: CleaningPlanner,
        rule_agent: RuleBasedCleaningAgent | None = None,
        policy: PreprocessingPolicy | None = None,
    ) -> None:
        self._policy = policy or PreprocessingPolicy()
        self._llm_agent = llm_agent
        self._rule_agent = rule_agent or RuleBasedCleaningAgent(policy=self._policy)
        self.last_llm_error: str | None = None

    def propose(
        self,
        dataframe: pd.DataFrame,
        profile: DataProfile | None = None,
    ) -> CleaningPlan:
        profile = profile or profile_dataframe(dataframe)
        rule_plan = self._rule_agent.propose(dataframe, profile)

        try:
            llm_plan = self._llm_agent.propose(dataframe, profile)
        except (OllamaError, LocalLLMPlanningError) as exc:
            self.last_llm_error = str(exc)
            return rule_plan

        self.last_llm_error = None
        return self._merge(rule_plan, llm_plan, profile, dataframe, self._policy)

    @staticmethod
    def _merge(
        rule_plan: CleaningPlan,
        llm_plan: CleaningPlan,
        profile: DataProfile,
        dataframe: pd.DataFrame,
        policy: PreprocessingPolicy | None = None,
    ) -> CleaningPlan:
        merged: list[CleaningStep] = []

        # Exact duplicate detection is deterministic and must not be omitted.
        rule_drop = next(
            (step for step in rule_plan.steps if step.operation == "drop_duplicates"),
            None,
        )
        if rule_drop is not None:
            merged.append(rule_drop)
        elif any(step.operation == "drop_duplicates" for step in llm_plan.steps):
            merged.extend(
                step
                for step in llm_plan.steps
                if step.operation == "drop_duplicates"
            )

        columns = list(profile.column_profiles)
        for column in columns:
            column_profile = profile.column_profiles[column]
            rule_steps = [
                step for step in rule_plan.steps if step.column == column
            ]
            llm_steps = [
                step
                for step in llm_plan.steps
                if step.column == column and step.operation != "drop_duplicates"
            ]

            # For an entirely missing column, never permit invented data.
            if column_profile.unique_count == 0:
                merged.extend(rule_steps)
                continue

            llm_actions = [
                step
                for step in llm_steps
                if step.operation != "leave_unchanged"
                and HybridCleaningAgent._is_compatible(step, dataframe, policy)
            ]
            merged.extend(llm_actions)
            llm_keys = {(step.operation, step.column) for step in llm_actions}

            if column_profile.missing_count > 0:
                llm_has_fill = any(
                    step.operation in FILL_OPERATIONS for step in llm_actions
                )
                merged.extend(
                    step
                    for step in rule_steps
                    if (step.operation, step.column) not in llm_keys
                    and (
                        step.operation not in FILL_OPERATIONS
                        or not llm_has_fill
                    )
                )
            elif not llm_actions:
                merged.extend(rule_steps)
                merged.extend(
                    step for step in llm_steps if step.operation == "leave_unchanged"
                )
            else:
                merged.extend(
                    step
                    for step in rule_steps
                    if (step.operation, step.column) not in llm_keys
                    and step.operation != "leave_unchanged"
                )

        # Retain any valid model steps that target no column and are not global
        # duplicate steps, such as an explicit dataset-level no-op.
        merged.extend(
            step
            for step in llm_plan.steps
            if step.column is None
            and step.operation not in {"drop_duplicates", "leave_unchanged"}
        )

        unique: list[CleaningStep] = []
        seen: set[tuple[str, str | None]] = set()
        for step in merged:
            key = (step.operation, step.column)
            if key not in seen:
                seen.add(key)
                unique.append(step)
        return filter_plan_by_policy(CleaningPlan(steps=unique), policy)

    @staticmethod
    def _is_compatible(
        step: CleaningStep,
        dataframe: pd.DataFrame,
        policy: PreprocessingPolicy | None = None,
    ) -> bool:
        """Reject model actions that are known to fail or lose observed data."""
        if not step.column or step.column not in dataframe.columns:
            return False
        if not operation_allowed(step, policy):
            return False
        series = dataframe[step.column]
        if step.operation in {"fill_mean", "fill_median"}:
            return is_numeric_dtype(series)
        if step.operation == "fill_mode":
            return bool(series.notna().any())
        if step.operation == "convert_numeric":
            converted = pd.to_numeric(series, errors="coerce")
            return int(converted.notna().sum()) == int(series.notna().sum())
        if step.operation == "convert_datetime":
            converted = pd.to_datetime(series, errors="coerce")
            return int(converted.notna().sum()) == int(series.notna().sum())
        if step.operation == "parse_numeric_text":
            return can_parse_numeric_text_losslessly(series)
        if step.operation == "strip_whitespace":
            return has_whitespace_issue(series)
        if step.operation == "normalize_case":
            return has_case_inconsistency(series)
        if step.operation == "normalize_category_typos":
            return has_category_typo_issue(series)
        if step.operation == "flag_outliers_iqr":
            return is_numeric_dtype(series) and has_iqr_outliers(series)
        return False
