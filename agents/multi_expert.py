"""Deterministic router, specialists, critic, arbiter, and planner.

Experts only propose whitelisted :class:`CleaningStep` objects. They never
execute transformations. The existing executor and validation/rollback layer
remain the only path that can change a dataframe.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from models.cleaning_plan import CleaningPlan, CleaningStep
from models.data_contract import DataContract
from models.orchestration import CriticResult, OrchestrationTrace, RoutingDecision
from models.policy import PreprocessingPolicy
from models.profile import DataProfile
from tools.policy import column_is_protected, operation_allowed, outlier_action_for_column, policy_with_contract
from tools.profiler import profile_dataframe
from tools.quality import (
    can_parse_numeric_text_losslessly,
    count_iqr_outliers,
    has_case_inconsistency,
    has_category_typo_issue,
    has_whitespace_issue,
    is_text_like,
    needs_robust_numeric_parsing,
    parse_numeric_text_series,
)


_DATE_NAME = re.compile(r"(?:^|_)(?:date|time|timestamp|datetime|created|updated)(?:$|_)", re.I)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$")
_FILL_OPERATIONS = {"fill_mean", "fill_median", "fill_mode"}
_CONVERSION_OPERATIONS = {"convert_numeric", "parse_numeric_text", "convert_datetime"}


def _step_key(step: CleaningStep) -> tuple[str | None, str]:
    return step.column, step.operation


def _strong_datetime_evidence(column: str, series: pd.Series) -> bool:
    """Require lossless parsing plus semantic or strict ISO evidence."""
    if is_datetime64_any_dtype(series):
        return False
    values = series.dropna()
    if len(values) < 3 or not is_text_like(series):
        return False
    text = values.astype(str).str.strip()
    semantic_hint = bool(_DATE_NAME.search(str(column)))
    iso_evidence = bool(text.map(lambda value: bool(_ISO_DATE.match(value))).all())
    if not (semantic_hint or iso_evidence):
        return False
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    return bool(parsed.notna().all())


class RouterAgent:
    """Route observable issues to bounded deterministic specialists."""

    name = "router_agent"

    def __init__(self, policy: PreprocessingPolicy | None = None) -> None:
        self._policy = policy or PreprocessingPolicy()

    def route(
        self, dataframe: pd.DataFrame, profile: DataProfile | None = None
    ) -> list[RoutingDecision]:
        profile = profile or profile_dataframe(dataframe)
        routes: list[RoutingDecision] = []
        if profile.duplicate_rows:
            routes.append(RoutingDecision(
                selected_expert="DuplicateExpert",
                issue_type="duplicate_rows",
                reason=f"Detected {profile.duplicate_rows} exact duplicate rows.",
                confidence=1.0,
            ))

        for column, column_profile in profile.column_profiles.items():
            series = dataframe[column]
            if column_is_protected(column, self._policy):
                routes.append(RoutingDecision(
                    column=column,
                    selected_expert=None,
                    issue_type="protected_identifier",
                    reason="Identifier-like or policy-protected column is excluded from automatic expert cleaning.",
                    confidence=1.0,
                ))
                continue

            if column_profile.missing_count:
                routes.append(RoutingDecision(
                    column=column,
                    selected_expert="MissingValueExpert",
                    issue_type="missing_values",
                    reason=f"Detected {column_profile.missing_count} missing values.",
                    confidence=1.0,
                ))

            if is_numeric_dtype(series):
                routes.append(RoutingDecision(
                    column=column,
                    selected_expert="NumericExpert",
                    issue_type="numeric_analysis",
                    reason="Numeric dtype is within the numeric specialist boundary.",
                    confidence=1.0,
                ))
                continue

            if can_parse_numeric_text_losslessly(series):
                routes.append(RoutingDecision(
                    column=column,
                    selected_expert="NumericExpert",
                    issue_type="numeric_text",
                    reason="All observed non-missing text values parse as numbers.",
                    confidence=0.95,
                ))
                continue

            if _strong_datetime_evidence(column, series):
                routes.append(RoutingDecision(
                    column=column,
                    selected_expert="DatetimeExpert",
                    issue_type="datetime_text",
                    reason="Observed values parse losslessly with strong datetime evidence.",
                    confidence=0.95,
                ))
                continue

            if has_whitespace_issue(series) or (
                is_text_like(series) and column_profile.unique_count > max(20, profile.rows // 5)
            ):
                routes.append(RoutingDecision(
                    column=column,
                    selected_expert="TextExpert",
                    issue_type="text_cleanup",
                    reason="Text evidence is suitable for bounded whitespace cleanup.",
                    confidence=0.9,
                ))

            if is_text_like(series) and column_profile.unique_count <= max(20, profile.rows // 5):
                routes.append(RoutingDecision(
                    column=column,
                    selected_expert="CategoricalExpert",
                    issue_type="categorical_consistency",
                    reason="Low-cardinality text is suitable for conservative category analysis.",
                    confidence=0.85,
                ))
        return routes


class SpecialistExpert(Protocol):
    name: str

    def propose(
        self, dataframe: pd.DataFrame, profile: DataProfile, routes: list[RoutingDecision]
    ) -> list[CleaningStep]: ...


def _routed_columns(routes: list[RoutingDecision], expert: str) -> list[str]:
    return [route.column for route in routes if route.selected_expert == expert and route.column is not None]


class DuplicateExpert:
    name = "DuplicateExpert"

    def propose(self, dataframe: pd.DataFrame, profile: DataProfile, routes: list[RoutingDecision]) -> list[CleaningStep]:
        if profile.duplicate_rows <= 0 or not any(route.selected_expert == self.name for route in routes):
            return []
        return [CleaningStep(operation="drop_duplicates", reason=f"Remove {profile.duplicate_rows} exact duplicate rows.", confidence=1.0)]


class MissingValueExpert:
    name = "MissingValueExpert"

    def propose(self, dataframe: pd.DataFrame, profile: DataProfile, routes: list[RoutingDecision]) -> list[CleaningStep]:
        steps: list[CleaningStep] = []
        for column in _routed_columns(routes, self.name):
            column_profile = profile.column_profiles[column]
            if column_profile.unique_count == 0:
                steps.append(CleaningStep(column=column, operation="leave_unchanged", reason="Column is entirely missing; imputation would invent unsupported data.", confidence=1.0))
            elif is_numeric_dtype(dataframe[column]) or can_parse_numeric_text_losslessly(dataframe[column]):
                steps.append(CleaningStep(column=column, operation="fill_median", reason="Use the observed median for missing numeric values; it is robust to outliers.", confidence=0.95))
            else:
                steps.append(CleaningStep(column=column, operation="fill_mode", reason="Use the most frequent observed category for missing text values.", confidence=0.85))
        return steps


class NumericExpert:
    name = "NumericExpert"

    def __init__(self, policy: PreprocessingPolicy | None = None) -> None:
        self._policy = policy or PreprocessingPolicy()

    def propose(self, dataframe: pd.DataFrame, profile: DataProfile, routes: list[RoutingDecision]) -> list[CleaningStep]:
        steps: list[CleaningStep] = []
        for column in _routed_columns(routes, self.name):
            series = dataframe[column]
            parsed = series
            if can_parse_numeric_text_losslessly(series):
                operation = "parse_numeric_text" if needs_robust_numeric_parsing(series) else "convert_numeric"
                steps.append(CleaningStep(column=column, operation=operation, reason="Convert losslessly parseable numeric text before numeric cleaning.", confidence=0.95))
                parsed = parse_numeric_text_series(series)
            outliers = count_iqr_outliers(parsed)
            if outliers and outlier_action_for_column(column, self._policy) == "flag":
                steps.append(CleaningStep(column=column, operation="flag_outliers_iqr", reason=f"Flag {outliers} potential IQR outliers without changing their values.", confidence=0.75))
        return steps


class CategoricalExpert:
    name = "CategoricalExpert"

    def propose(self, dataframe: pd.DataFrame, profile: DataProfile, routes: list[RoutingDecision]) -> list[CleaningStep]:
        steps: list[CleaningStep] = []
        for column in _routed_columns(routes, self.name):
            series = dataframe[column]
            if has_case_inconsistency(series):
                steps.append(CleaningStep(column=column, operation="normalize_case", reason="Normalize observed case variants to the strongest existing spelling.", confidence=0.85))
            if has_category_typo_issue(series):
                steps.append(CleaningStep(column=column, operation="normalize_category_typos", reason="Normalize only rare labels that closely match a frequent observed category.", confidence=0.78))
        return steps


class TextExpert:
    name = "TextExpert"

    def propose(self, dataframe: pd.DataFrame, profile: DataProfile, routes: list[RoutingDecision]) -> list[CleaningStep]:
        return [
            CleaningStep(column=column, operation="strip_whitespace", reason="Remove leading and trailing whitespace from text cells only.", confidence=0.9)
            for column in _routed_columns(routes, self.name)
            if has_whitespace_issue(dataframe[column])
        ]


class DatetimeExpert:
    name = "DatetimeExpert"

    def propose(self, dataframe: pd.DataFrame, profile: DataProfile, routes: list[RoutingDecision]) -> list[CleaningStep]:
        return [
            CleaningStep(column=column, operation="convert_datetime", reason="Convert text only after lossless parsing with strong datetime evidence.", confidence=0.95)
            for column in _routed_columns(routes, self.name)
            if _strong_datetime_evidence(column, dataframe[column])
        ]


class CriticAgent:
    """Reject unsafe, invalid, redundant, or conflicting proposals."""

    name = "critic_agent"

    def __init__(self, policy: PreprocessingPolicy | None = None, contract: DataContract | None = None) -> None:
        self._policy = policy or PreprocessingPolicy()
        self._contract = contract

    def _contract_conflict(self, step: CleaningStep) -> str | None:
        if self._contract is None or step.column is None:
            return None
        rule = self._contract.columns.get(step.column)
        if rule is None:
            return None
        allowed = set(rule.allowed_dtypes or [])
        if step.operation in {"convert_numeric", "parse_numeric_text"} and allowed and not allowed.intersection({"number", "integer", "float", "any"}):
            return f"Rejected {step.operation}: contract requires a non-numeric type for {step.column!r}."
        if step.operation == "convert_datetime" and allowed and not allowed.intersection({"datetime", "any"}):
            return f"Rejected convert_datetime: contract does not allow datetime for {step.column!r}."
        if step.operation in {"normalize_case", "normalize_category_typos"} and rule.allowed_values is not None:
            return f"Rejected {step.operation}: contract allowlist makes automatic category rewriting unsafe for {step.column!r}."
        return None

    def review(self, dataframe: pd.DataFrame, proposals: list[CleaningStep]) -> CriticResult:
        profile = profile_dataframe(dataframe)
        rejected: list[CleaningStep] = []
        candidates: list[CleaningStep] = []
        warnings: list[str] = []
        reasons: list[str] = []
        seen: set[tuple[str | None, str]] = set()

        for step in proposals:
            key = _step_key(step)
            reason: str | None = None
            if key in seen:
                reason = f"Rejected duplicate step {step.operation} on {step.column!r}."
                warnings.append(reason)
            elif step.column is not None and step.column not in dataframe.columns:
                reason = f"Rejected {step.operation}: unknown column {step.column!r}."
            elif step.column is not None and column_is_protected(step.column, self._policy) and step.operation != "leave_unchanged":
                reason = f"Rejected {step.operation}: {step.column!r} is protected."
            elif not operation_allowed(step, self._policy):
                reason = f"Rejected {step.operation}: preprocessing policy does not allow it."
            elif self._contract_conflict(step) is not None:
                reason = self._contract_conflict(step)
            elif step.operation in _FILL_OPERATIONS and step.column is not None and profile.column_profiles[step.column].unique_count == 0:
                reason = f"Rejected {step.operation}: {step.column!r} is entirely missing."
            elif step.operation in {"fill_mean", "fill_median", "flag_outliers_iqr"} and step.column is not None and not (
                is_numeric_dtype(dataframe[step.column]) or can_parse_numeric_text_losslessly(dataframe[step.column])
            ):
                reason = f"Rejected {step.operation}: {step.column!r} lacks safe numeric evidence."
            elif step.operation in {"convert_numeric", "parse_numeric_text"} and step.column is not None and not can_parse_numeric_text_losslessly(dataframe[step.column]):
                reason = f"Rejected unsafe numeric conversion for {step.column!r}."
            elif step.operation == "convert_datetime" and step.column is not None and not _strong_datetime_evidence(step.column, dataframe[step.column]):
                reason = f"Rejected unsafe datetime conversion for {step.column!r}."
            elif step.operation == "normalize_category_typos" and step.column is not None and not has_category_typo_issue(dataframe[step.column]):
                reason = f"Rejected aggressive category normalization without typo evidence for {step.column!r}."

            seen.add(key)
            if reason:
                rejected.append(step)
                reasons.append(reason)
            else:
                candidates.append(step)

        conflicts: set[int] = set()
        by_column: dict[str, list[tuple[int, CleaningStep]]] = defaultdict(list)
        for index, step in enumerate(candidates):
            if step.column is not None:
                by_column[step.column].append((index, step))
        for column, indexed_steps in by_column.items():
            fills = [(i, s) for i, s in indexed_steps if s.operation in _FILL_OPERATIONS]
            conversions = [(i, s) for i, s in indexed_steps if s.operation in _CONVERSION_OPERATIONS]
            mutating = [(i, s) for i, s in indexed_steps if s.operation not in {"leave_unchanged", "flag_outliers_iqr"}]
            unchanged = [(i, s) for i, s in indexed_steps if s.operation == "leave_unchanged"]
            conflict_groups = []
            if len({s.operation for _, s in fills}) > 1:
                conflict_groups.append(fills)
            if len({s.operation for _, s in conversions}) > 1:
                conflict_groups.append(conversions)
            if unchanged and mutating:
                conflict_groups.append(mutating)
            for group in conflict_groups:
                conflicts.update(i for i, _ in group)
                detail = ", ".join(step.operation for _, step in group)
                message = f"Rejected conflicting operations on {column!r}: {detail}."
                if message not in reasons:
                    reasons.append(message)
                    warnings.append(message)

        approved = [step for index, step in enumerate(candidates) if index not in conflicts]
        rejected.extend(step for index, step in enumerate(candidates) if index in conflicts)
        return CriticResult(approved_steps=approved, rejected_steps=rejected, warnings=warnings, reasons=reasons)


class Arbiter:
    """Assemble a deterministic executor-compatible plan conservatively."""

    name = "arbiter"
    _ORDER = {
        "drop_duplicates": 0,
        "strip_whitespace": 10,
        "normalize_case": 20,
        "normalize_category_typos": 30,
        "convert_numeric": 40,
        "parse_numeric_text": 40,
        "convert_datetime": 40,
        "fill_mean": 50,
        "fill_median": 50,
        "fill_mode": 50,
        "flag_outliers_iqr": 60,
        "leave_unchanged": 70,
    }

    def resolve(self, review: CriticResult) -> CleaningPlan:
        unique: dict[tuple[str | None, str], CleaningStep] = {}
        for step in review.approved_steps:
            key = _step_key(step)
            existing = unique.get(key)
            if existing is None or step.confidence > existing.confidence:
                unique[key] = step

        by_column: dict[str, list[CleaningStep]] = defaultdict(list)
        global_steps: list[CleaningStep] = []
        for step in unique.values():
            (global_steps if step.column is None else by_column[step.column]).append(step)

        safe: list[CleaningStep] = list(global_steps)
        for column, steps in by_column.items():
            fills = [step for step in steps if step.operation in _FILL_OPERATIONS]
            conversions = [step for step in steps if step.operation in _CONVERSION_OPERATIONS]
            if len({step.operation for step in fills}) > 1 or len({step.operation for step in conversions}) > 1:
                safe.extend(step for step in steps if step.operation == "leave_unchanged")
                continue
            if any(step.operation == "leave_unchanged" for step in steps):
                safe.extend(step for step in steps if step.operation == "leave_unchanged")
            else:
                safe.extend(steps)
        safe.sort(key=lambda step: (self._ORDER[step.operation], step.column or ""))
        return CleaningPlan(steps=safe)


class MultiExpertCleaningAgent:
    """Planner implementing Router -> Specialists -> Critic -> Arbiter."""

    name = "multi_expert_cleaning_agent"

    def __init__(
        self,
        policy: PreprocessingPolicy | None = None,
        router: RouterAgent | None = None,
        specialists: list[SpecialistExpert] | None = None,
        critic: CriticAgent | None = None,
        arbiter: Arbiter | None = None,
        contract: DataContract | None = None,
    ) -> None:
        self._contract = contract
        self._policy = policy_with_contract(policy, contract)
        self.router = router or RouterAgent(self._policy)
        self.specialists = specialists or [
            DuplicateExpert(), MissingValueExpert(), NumericExpert(self._policy),
            CategoricalExpert(), TextExpert(), DatetimeExpert(),
        ]
        self.critic = critic or CriticAgent(self._policy, contract)
        self.arbiter = arbiter or Arbiter()
        self.last_trace: OrchestrationTrace | None = None

    def propose(self, dataframe: pd.DataFrame, profile: DataProfile | None = None) -> CleaningPlan:
        profile = profile or profile_dataframe(dataframe)
        routes = self.router.route(dataframe, profile)
        specialist_proposals = {
            expert.name: expert.propose(dataframe, profile, routes)
            for expert in self.specialists
        }
        all_steps = [step for steps in specialist_proposals.values() for step in steps]
        critique = self.critic.review(dataframe, all_steps)
        final_plan = self.arbiter.resolve(critique)
        self.last_trace = OrchestrationTrace(
            routes=routes,
            specialist_proposals=specialist_proposals,
            critique=critique,
            final_plan=final_plan,
        )
        return final_plan
