"""Deterministic metrics for routing and orchestration experiments."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean, median, pstdev
from typing import Iterable

from evaluation.multi_expert_schema import MULTI_EXPERT_EVALUATION_SCHEMA_VERSION
from evaluation.multi_expert_scenarios import EXPERTS, RoutingScenario
from models.orchestration import RoutingDecision


def _ratio(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def route_set(routes: Iterable[RoutingDecision]) -> set[tuple[str, str]]:
    return {(r.column or "*", r.selected_expert) for r in routes if r.selected_expert}


def routing_metrics(
    cases: list[tuple[RoutingScenario, list[RoutingDecision]]],
    *,
    schema_version: str = MULTI_EXPERT_EVALUATION_SCHEMA_VERSION,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    counts = {expert: {"tp": 0, "fp": 0, "fn": 0} for expert in EXPERTS}
    exact = clean_total = clean_ok = protected_total = protected_ok = conflicts = routed_columns = 0
    predictions: list[dict[str, object]] = []
    for scenario, decisions in cases:
        expected, predicted = set(scenario.expected_routes), route_set(decisions)
        exact += expected == predicted
        if scenario.clean:
            clean_total += 1; clean_ok += not predicted
        for column in scenario.expected_exclusions:
            protected_total += 1
            protected_ok += not any(c == column for c, _ in predicted)
        by_column: dict[str, set[str]] = defaultdict(set)
        for column, expert in predicted:
            by_column[column].add(expert)
        routed_columns += len(by_column)
        conflicts += sum(len(v) > 1 for v in by_column.values())
        for label in sorted(expected | predicted):
            column, expert = label
            is_expected, is_predicted = label in expected, label in predicted
            if expert in counts:
                counts[expert]["tp" if is_expected and is_predicted else "fp" if is_predicted else "fn"] += 1
            predictions.append({
                "schema_version": schema_version, "scenario_id": scenario.scenario_id,
                "seed": scenario.seed, "column": column, "expert": expert,
                "expected": is_expected, "predicted": is_predicted,
            })
    rows=[]
    for expert in EXPERTS:
        c=counts[expert]; p=_ratio(c["tp"],c["tp"]+c["fp"]); r=_ratio(c["tp"],c["tp"]+c["fn"]); f=_ratio(2*p*r,p+r)
        rows.append({"schema_version":schema_version,"specialist":expert,**c,"precision":p,"recall":r,"f1":f})
    tp=sum(c["tp"] for c in counts.values()); fp=sum(c["fp"] for c in counts.values()); fn=sum(c["fn"] for c in counts.values())
    mp=mean(float(r["precision"]) for r in rows); mr=mean(float(r["recall"]) for r in rows); mf=mean(float(r["f1"]) for r in rows)
    mip=_ratio(tp,tp+fp); mir=_ratio(tp,tp+fn)
    summary={
        "schema_version":schema_version,"evaluation_unit":"scenario-column-specialist assignment",
        "scenario_count":len(cases),"exact_routing_accuracy":_ratio(exact,len(cases)),
        "multi_label_exact_match_ratio":_ratio(exact,len(cases)),
        "macro_precision":mp,"macro_recall":mr,"macro_f1":mf,
        "micro_precision":mip,"micro_recall":mir,"micro_f1":_ratio(2*mip*mir,mip+mir),
        "false_routing_rate":_ratio(fp,tp+fp),"missed_routing_rate":_ratio(fn,tp+fn),
        "unnecessary_specialist_activation_rate":_ratio(fp,tp+fp),
        "protected_column_exclusion_accuracy":_ratio(protected_ok,protected_total),
        "clean_dataset_no_op_accuracy":_ratio(clean_ok,clean_total),
        "specialist_coverage":_ratio(sum(c["tp"]>0 for c in counts.values()),len(EXPERTS)),
        "routing_conflict_rate":_ratio(conflicts,routed_columns),
    }
    return predictions, {"by_specialist":rows,"summary":summary}


def descriptive(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n":0,"mean":0.0,"median":0.0,"std":0.0,"min":0.0,"max":0.0,"ci95_low":0.0,"ci95_high":0.0}
    avg=mean(values); sd=pstdev(values); half=1.96*sd/sqrt(len(values))
    return {"n":len(values),"mean":avg,"median":median(values),"std":sd,"min":min(values),"max":max(values),"ci95_low":avg-half,"ci95_high":avg+half}
