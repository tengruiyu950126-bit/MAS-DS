"""Structured contracts for deterministic multi-expert orchestration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.cleaning_plan import CleaningPlan, CleaningStep


class RoutingDecision(BaseModel):
    """One auditable assignment from dataframe evidence to an expert."""

    column: str | None = None
    selected_expert: str | None = None
    issue_type: str
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class CriticResult(BaseModel):
    """Structured review of specialist proposals."""

    approved_steps: list[CleaningStep] = Field(default_factory=list)
    rejected_steps: list[CleaningStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class OrchestrationTrace(BaseModel):
    """Inspectable output from every planning stage."""

    routes: list[RoutingDecision]
    specialist_proposals: dict[str, list[CleaningStep]]
    critique: CriticResult
    final_plan: CleaningPlan
