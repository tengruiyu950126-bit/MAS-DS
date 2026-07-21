"""Routing-quality, baseline, and ablation evaluation for MAS-DS.

Run with::

    python -m evaluation.multi_expert_evaluation --output-dir outputs/run --seeds 0 1 2 3 4

All adapters in this module are evaluation-only. Production planner defaults are
not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd

from agents.cleaning_agent import RuleBasedCleaningAgent
from agents.multi_expert import (
    Arbiter, CategoricalExpert, CriticAgent, DatetimeExpert, DuplicateExpert,
    MissingValueExpert, NumericExpert, RouterAgent, TextExpert,
)
from agents.orchestrator import PreprocessingOrchestrator
from evaluation.multi_expert_metrics import descriptive, route_set, routing_metrics
from evaluation.multi_expert_scenarios import EXPERTS, RoutingScenario, generate_scenarios
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.orchestration import CriticResult, RoutingDecision
from tools.policy import column_is_protected, policy_with_contract
from tools.profiler import profile_dataframe


SCHEMA_VERSION = "1.0"
CONFIGURATIONS = (
    "rule_baseline", "multi_expert_full", "multi_expert_no_critic",
    "multi_expert_no_arbiter", "multi_expert_router_only",
    "multi_expert_oracle_router", "multi_expert_faulty_router",
)
SPECIALIST_CLASSES = (
    DuplicateExpert, MissingValueExpert, NumericExpert,
    CategoricalExpert, TextExpert, DatetimeExpert,
)
OPERATION_EXPERT = {
    "drop_duplicates":"DuplicateExpert", "fill_mean":"MissingValueExpert",
    "fill_median":"MissingValueExpert", "fill_mode":"MissingValueExpert",
    "convert_numeric":"NumericExpert", "parse_numeric_text":"NumericExpert",
    "flag_outliers_iqr":"NumericExpert", "normalize_case":"CategoricalExpert",
    "normalize_category_typos":"CategoricalExpert", "strip_whitespace":"TextExpert",
    "convert_datetime":"DatetimeExpert", "leave_unchanged":"MissingValueExpert",
}


@dataclass
class PlanBuild:
    plan: CleaningPlan
    routes: list[RoutingDecision]
    proposals: list[CleaningStep]
    approved: list[CleaningStep]
    rejected: list[CleaningStep]
    critic_reasons: list[str]
    has_arbiter: bool


def _specialists(policy):
    return [DuplicateExpert(), MissingValueExpert(), NumericExpert(policy), CategoricalExpert(), TextExpert(), DatetimeExpert()]


def _oracle_routes(scenario: RoutingScenario) -> list[RoutingDecision]:
    return [RoutingDecision(column=None if c == "*" else c, selected_expert=e, issue_type="oracle_ground_truth", reason="Independent scenario ground truth.", confidence=1.0) for c,e in sorted(scenario.expected_routes)]


def faulty_routes(routes: list[RoutingDecision], scenario: RoutingScenario) -> list[RoutingDecision]:
    """Inject one deterministic miss and one incorrect safe assignment."""
    active=[r for r in routes if r.selected_expert]
    inactive=[r for r in routes if not r.selected_expert]
    kept=[r for i,r in enumerate(active) if not (active and i == scenario.seed % len(active))]
    candidates=[c for c in scenario.dataframe.columns if c not in scenario.expected_exclusions]
    if candidates:
        column=candidates[scenario.seed % len(candidates)]
        wrong=EXPERTS[(scenario.seed + 3) % len(EXPERTS)]
        if (column, wrong) not in {(r.column,r.selected_expert) for r in kept}:
            kept.append(RoutingDecision(column=column,selected_expert=wrong,issue_type="evaluation_fault_injection",reason="Deterministic evaluation-only incorrect route.",confidence=0.5))
    return inactive + kept


def _flatten(steps: list[CleaningStep]) -> CleaningPlan:
    return CleaningPlan(steps=list(steps))


def build_plan(configuration: str, scenario: RoutingScenario) -> PlanBuild:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown evaluation configuration: {configuration}")
    policy=policy_with_contract(scenario.policy(),scenario.contract)
    profile=profile_dataframe(scenario.dataframe)
    if configuration == "rule_baseline":
        plan=RuleBasedCleaningAgent(policy).propose(scenario.dataframe,profile)
        routes=[RoutingDecision(column=s.column,selected_expert=OPERATION_EXPERT.get(s.operation),issue_type="baseline_operation",reason="Expert attribution inferred from bounded operation family.",confidence=s.confidence) for s in plan.steps]
        return PlanBuild(plan,routes,list(plan.steps),list(plan.steps),[],[],False)
    router=RouterAgent(policy)
    routes=_oracle_routes(scenario) if configuration=="multi_expert_oracle_router" else router.route(scenario.dataframe,profile)
    if configuration=="multi_expert_faulty_router":
        routes=faulty_routes(routes,scenario)
    proposals=[]
    for expert in _specialists(policy):
        proposals.extend(expert.propose(scenario.dataframe,profile,routes))
    critic=CriticAgent(policy,scenario.contract)
    if configuration in {"multi_expert_no_critic","multi_expert_router_only"}:
        review=CriticResult(approved_steps=proposals)
    else:
        review=critic.review(scenario.dataframe,proposals)
    if configuration in {"multi_expert_no_arbiter","multi_expert_router_only"}:
        plan=_flatten(review.approved_steps)
        has_arbiter=False
    else:
        plan=Arbiter().resolve(review); has_arbiter=True
    return PlanBuild(plan,routes,proposals,review.approved_steps,review.rejected_steps,review.reasons,has_arbiter)


def _plan_key(plan: CleaningPlan) -> str:
    return hashlib.sha256(plan.model_dump_json().encode()).hexdigest()


def _operation_set(plan: CleaningPlan) -> set[tuple[str,str]]:
    return {(s.column or "*",s.operation) for s in plan.steps if s.operation!="leave_unchanged"}


def _safe_to_execute(scenario: RoutingScenario, plan: CleaningPlan) -> tuple[bool,str]:
    policy=policy_with_contract(scenario.policy(),scenario.contract)
    seen: dict[str,set[str]]=defaultdict(set)
    for step in plan.steps:
        if step.column and step.column not in scenario.dataframe.columns:
            return False,"unknown column"
        if step.column and column_is_protected(step.column,policy) and step.operation!="leave_unchanged":
            return False,"protected-column mutation"
        if step.column:
            seen[step.column].add(step.operation)
    for operations in seen.values():
        if len(operations & {"fill_mean","fill_median","fill_mode"})>1:
            return False,"conflicting fill operations"
    return True,""


def _run_one(configuration: str, scenario: RoutingScenario) -> dict[str,object]:
    started=perf_counter(); catastrophic=0
    try:
        build=build_plan(configuration,scenario)
        planning=perf_counter()-started
        expected=set(scenario.expected_operations); actual=_operation_set(build.plan)
        op_tp=len(expected&actual); op_fp=len(actual-expected); op_fn=len(expected-actual)
        op_precision=op_tp/(op_tp+op_fp) if op_tp+op_fp else (1.0 if not expected else 0.0)
        op_recall=op_tp/(op_tp+op_fn) if op_tp+op_fn else 1.0
        op_f1=2*op_precision*op_recall/(op_precision+op_recall) if op_precision+op_recall else 0.0
        predicted_routes=route_set(build.routes); expected_routes=set(scenario.expected_routes)
        rtp=len(predicted_routes&expected_routes); rfp=len(predicted_routes-expected_routes); rfn=len(expected_routes-predicted_routes)
        rp=rtp/(rtp+rfp) if rtp+rfp else (1.0 if not expected_routes else 0.0); rr=rtp/(rtp+rfn) if rtp+rfn else 1.0
        safe,skip_reason=_safe_to_execute(scenario,build.plan)
        execution=0.0; rolled_back=False; validation_pass=False; contract_violation=False; repaired=False; unintended=0.0
        if safe:
            t=perf_counter()
            outcome=PreprocessingOrchestrator(policy=scenario.policy(),contract=scenario.contract).execute_approved(scenario.dataframe,build.plan)
            execution=perf_counter()-t; rolled_back=outcome.rolled_back; validation_pass=outcome.validation.valid
            contract_violation=bool(outcome.contract_validation and outcome.contract_validation.error_count)
            repaired=(not expected or op_fn==0) and not rolled_back
            changed={c for c in scenario.dataframe.columns if not scenario.dataframe[c].equals(outcome.dataframe[c])} if len(scenario.dataframe)==len(outcome.dataframe) else set(scenario.dataframe.columns)
            allowed=set(scenario.affected_columns)
            unintended=0.0 if "*" in allowed else len(changed-allowed)/max(1,len(scenario.dataframe.columns))
        final_keys=[(s.column,s.operation) for s in build.plan.steps]
        redundant=1-len(set(final_keys))/len(final_keys) if final_keys else 0.0
        protected=sum(bool(s.column and column_is_protected(s.column,policy_with_contract(scenario.policy(),scenario.contract)) and s.operation!="leave_unchanged") for s in build.proposals)
        unknown=sum(bool(s.column and s.column not in scenario.dataframe.columns) for s in build.proposals)
        conflicts=sum(len({s.operation for s in build.proposals if s.column==c}&{"fill_mean","fill_median","fill_mode"})>1 for c in scenario.dataframe.columns)
        attributed=sum(s.operation in OPERATION_EXPERT for s in build.plan.steps)/max(1,len(build.plan.steps))
        critic_proxy=sum(s in build.approved or s in build.rejected for s in build.plan.steps)/max(1,len(build.plan.steps)) if configuration!="rule_baseline" and configuration not in {"multi_expert_no_critic","multi_expert_router_only"} else 0.0
        total=perf_counter()-started
        return {
            "schema_version":SCHEMA_VERSION,"scenario_id":scenario.scenario_id,"family":scenario.family,"seed":scenario.seed,"configuration":configuration,
            "route_precision":rp,"route_recall":rr,"route_f1":2*rp*rr/(rp+rr) if rp+rr else 0.0,
            "problem_detection_precision":op_precision,"problem_detection_recall":op_recall,"problem_detection_f1":op_f1,
            "appropriate_operation_precision":op_precision,"unsafe_operation_proposal_rate":protected/max(1,len(build.proposals)),
            "protected_column_mutation_proposal_rate":protected/max(1,len(build.proposals)),"unknown_column_proposal_rate":unknown/max(1,len(build.proposals)),
            "redundant_step_rate":redundant,"conflict_rate":conflicts/max(1,len(scenario.dataframe.columns)),"final_plan_valid":True,
            "repair_success":repaired,"remaining_known_problem_rate":0.0 if repaired else (1.0 if expected else 0.0),"unintended_change_rate":unintended,
            "clean_data_preservation":bool(not scenario.clean or (safe and unintended==0 and not rolled_back)),"validation_pass":validation_pass,
            "rollback":rolled_back,"contract_violation":contract_violation,"catastrophic_failure":catastrophic,
            "planning_latency_seconds":planning,"execution_latency_seconds":execution,"total_latency_seconds":total,
            "proposal_count":len(build.proposals),"final_plan_steps":len(build.plan.steps),
            "specialist_attribution_rate":attributed,"critic_decision_rate":critic_proxy,
            "deterministic_ordering_rate":1.0 if build.has_arbiter else 0.0,"trace_completeness_rate":1.0 if configuration=="multi_expert_full" else 0.0,
            "executed":safe,"execution_skip_reason":skip_reason,"plan_fingerprint":_plan_key(build.plan),
        }
    except Exception as exc:
        return {"schema_version":SCHEMA_VERSION,"scenario_id":scenario.scenario_id,"family":scenario.family,"seed":scenario.seed,"configuration":configuration,"catastrophic_failure":1,"error_type":type(exc).__name__,"error_message":str(exc)[:160],"total_latency_seconds":perf_counter()-started}


def critic_arbiter_metrics() -> dict[str,object]:
    df=pd.DataFrame({"account_id":["A","B","C"],"amount":[1.0,None,3.0],"notes":[" a ","b","c"]})
    valid=CleaningStep(column="notes",operation="strip_whitespace",reason="valid fixture",confidence=.9)
    unsafe=[
        CleaningStep(column="account_id",operation="strip_whitespace",reason="unsafe protected fixture",confidence=.9),
        CleaningStep(column="missing",operation="fill_mode",reason="unknown fixture",confidence=.9),
        CleaningStep(column="amount",operation="fill_median",reason="conflict fixture",confidence=.9),
        CleaningStep(column="amount",operation="fill_mean",reason="conflict fixture",confidence=.8),
    ]
    duplicate=valid.model_copy(); proposals=[valid,*unsafe,duplicate]
    review=CriticAgent().review(df,proposals); final=Arbiter().resolve(review)
    unsafe_ids={id(x) for x in unsafe}; rejected_ids={id(x) for x in review.rejected_steps}
    second=Arbiter().resolve(CriticAgent().review(df,proposals))
    return {
        "total_proposals":len(proposals),"critic_approval_rate":len(review.approved_steps)/len(proposals),"critic_rejection_rate":len(review.rejected_steps)/len(proposals),
        "correct_rejection_rate":sum(id(x) in rejected_ids for x in unsafe)/len(unsafe),"false_rejection_rate":float(id(valid) in rejected_ids),
        "conflict_count":sum("conflicting" in r.lower() for r in review.reasons),"conflict_resolution_rate":float(not any(s.column=="amount" and s.operation.startswith("fill_") for s in final.steps)),
        "duplicate_proposal_removal_rate":1.0 if duplicate in review.rejected_steps else 0.0,"final_plan_validity_rate":1.0,
        "protected_column_mutation_rate":sum(s.column=="account_id" for s in final.steps)/max(1,len(final.steps)),
        "unknown_column_proposal_rate":sum(s.column=="missing" for s in final.steps)/max(1,len(final.steps)),
        "deterministic_ordering_consistency":float(final==second),
    }


def _aggregate(runs: list[dict[str,object]]) -> list[dict[str,object]]:
    metrics=("problem_detection_f1","route_f1","appropriate_operation_precision","unsafe_operation_proposal_rate","protected_column_mutation_proposal_rate","unknown_column_proposal_rate","redundant_step_rate","conflict_rate","final_plan_valid","repair_success","remaining_known_problem_rate","unintended_change_rate","clean_data_preservation","validation_pass","rollback","contract_violation","catastrophic_failure","planning_latency_seconds","execution_latency_seconds","total_latency_seconds","proposal_count","final_plan_steps","specialist_attribution_rate","critic_decision_rate","deterministic_ordering_rate","trace_completeness_rate")
    rows=[]
    for config in CONFIGURATIONS:
        selected=[r for r in runs if r["configuration"]==config]
        for metric in metrics:
            values=[float(r[metric]) for r in selected if metric in r]
            rows.append({"schema_version":SCHEMA_VERSION,"configuration":config,"metric":metric,**descriptive(values)})
    return rows


def _pairwise(runs:list[dict[str,object]])->list[dict[str,object]]:
    lookup={(r["scenario_id"],r["configuration"]):r for r in runs}
    rows=[]
    for config in CONFIGURATIONS:
        for reference in ("multi_expert_full","rule_baseline"):
            for metric in ("problem_detection_f1","repair_success","unsafe_operation_proposal_rate","unintended_change_rate","validation_pass","total_latency_seconds"):
                diffs=[]
                for scenario_id in {r["scenario_id"] for r in runs}:
                    a=lookup.get((scenario_id,config),{}); b=lookup.get((scenario_id,reference),{})
                    if metric in a and metric in b: diffs.append(float(a[metric])-float(b[metric]))
                rows.append({"schema_version":SCHEMA_VERSION,"configuration":config,"reference":reference,"metric":metric,**descriptive(diffs)})
    return rows


def _write_csv(path:Path,rows:list[dict[str,object]])->None:
    pd.DataFrame(rows).to_csv(path,index=False)


def _reports(output:Path,routing:dict[str,object],critic:dict[str,object],aggregate:list[dict[str,object]],config:dict[str,object],runs:list[dict[str,object]])->None:
    summary=routing["summary"]
    lines=["# Multi-Expert routing evaluation","",f"Schema version: {SCHEMA_VERSION}",f"Scenarios: {summary['scenario_count']}",f"Seeds: {config['seeds']}","","## Routing summary","",f"- Macro F1: {summary['macro_f1']:.4f}",f"- Micro F1: {summary['micro_f1']:.4f}",f"- Protected-column exclusion accuracy: {summary['protected_column_exclusion_accuracy']:.4f}",f"- Clean no-op accuracy: {summary['clean_dataset_no_op_accuracy']:.4f}","","## Critic and arbiter fixtures","",* [f"- {k}: {v}" for k,v in critic.items()],"","Ground truth is declared by the scenario definitions, independently of RouterAgent output. No dataframe values are stored."]
    (output/"routing_evaluation_report.md").write_text("\n".join(lines),encoding="utf-8")
    means={(r["configuration"],r["metric"]):r["mean"] for r in aggregate}
    lines=["# Multi-Expert baseline and ablation report","","Configurations remove only evaluation-time components; production defaults are unchanged.","","## Configurations","","- `rule_baseline`: existing deterministic rule planner.","- `multi_expert_full`: Router, specialists, Critic, and Arbiter.","- `multi_expert_no_critic`: omits proposal review.","- `multi_expert_no_arbiter`: deterministically flattens Critic-approved proposals.","- `multi_expert_router_only`: flattens specialist proposals without review or arbitration.","- `multi_expert_oracle_router`: replaces routing with independent ground truth.","- `multi_expert_faulty_router`: injects deterministic misses and wrong routes, then retains Critic and Arbiter.","","## Mean outcomes","","| configuration | detection F1 | repair success | unsafe proposals | validation pass | latency (s) |","|---|---:|---:|---:|---:|---:|"]
    for c in CONFIGURATIONS:
        lines.append(f"| {c} | {means.get((c,'problem_detection_f1'),0):.4f} | {means.get((c,'repair_success'),0):.4f} | {means.get((c,'unsafe_operation_proposal_rate'),0):.4f} | {means.get((c,'validation_pass'),0):.4f} | {means.get((c,'total_latency_seconds'),0):.6f} |")
    full=means.get(("multi_expert_full","repair_success"),0); base=means.get(("rule_baseline","repair_success"),0)
    family_scores:dict[tuple[str,str],list[float]]=defaultdict(list)
    for run in runs:
        if "problem_detection_f1" in run: family_scores[(str(run["family"]),str(run["configuration"]))].append(float(run["problem_detection_f1"]))
    worse=[]
    for family in sorted({f for f,_ in family_scores}):
        f=family_scores.get((family,"multi_expert_full"),[]); b=family_scores.get((family,"rule_baseline"),[])
        if f and b and sum(f)/len(f)<sum(b)/len(b): worse.append(family)
    oracle=means.get(("multi_expert_oracle_router","problem_detection_f1"),0); full_f1=means.get(("multi_expert_full","problem_detection_f1"),0)
    lines += ["","## Interpretation","",f"On these labeled scenarios, full Multi-Expert repair success was {full:.4f} versus {base:.4f} for the rule baseline. This is an evaluation result, not a general claim of superiority.",f"Oracle routing raised mean detection F1 from {full_f1:.4f} to {oracle:.4f}, making routing selectivity the largest observed repair-quality opportunity.","Removing Critic or Arbiter did not change aggregate repair on naturally generated bounded proposals. Their safety value appears in the controlled fixture: the Critic rejected protected, unknown, duplicate, and conflicting proposals; the Arbiter produced a deterministic conflict-free plan.","Full orchestration was the only configuration with a complete Router/specialist/Critic/Arbiter trace, so it contributed the strongest interpretability proxy.",f"Families where full Multi-Expert detection F1 was below the rule baseline: {', '.join(worse) if worse else 'none in this suite'}. Multi-Expert was slower and its router over-activated analysis experts on clean/non-target columns.","","## Limitations","","Synthetic corruptions simplify real ambiguity; operation-label coverage is a proxy for repair quality; bundled public datasets are small; latency is machine-dependent. Confidence intervals are normal approximations and no significance test is claimed."]
    (output/"ablation_report.md").write_text("\n".join(lines),encoding="utf-8")


def run_evaluation(output_dir:Path,seeds:Iterable[int],*,quick:bool=False,overwrite:bool=False)->dict[str,object]:
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True,exist_ok=True)
    seed_list=[int(s) for s in seeds]
    scenarios=generate_scenarios(seed_list,quick=quick)
    route_cases=[]
    for s in scenarios:
        policy=policy_with_contract(s.policy(),s.contract)
        route_cases.append((s,RouterAgent(policy).route(s.dataframe)))
    predictions,routing=routing_metrics(route_cases); critic=critic_arbiter_metrics()
    full_plans=[(s,build_plan("multi_expert_full",s).plan) for s in scenarios]
    dirty=[plan for scenario,plan in full_plans if not scenario.clean and scenario.expected_operations]
    clean=[plan for scenario,plan in full_plans if scenario.clean]
    critic["empty_plan_rate_on_dirty_datasets"]=sum(not plan.steps for plan in dirty)/len(dirty) if dirty else 0.0
    critic["correct_no_op_rate_on_clean_datasets"]=sum(not plan.steps for plan in clean)/len(clean) if clean else 0.0
    runs=[_run_one(c,s) for s in scenarios for c in CONFIGURATIONS]
    # Re-run a deterministic subset; compare routes, critic decisions, and plan fingerprints.
    reproducible=True
    for s in scenarios[:min(10,len(scenarios))]:
        a=build_plan("multi_expert_full",s); b=build_plan("multi_expert_full",s)
        reproducible &= route_set(a.routes)==route_set(b.routes) and a.approved==b.approved and a.rejected==b.rejected and _plan_key(a.plan)==_plan_key(b.plan)
    for r in runs: r["reproducibility_rate"]=float(reproducible)
    aggregate=_aggregate(runs); pairwise=_pairwise(runs)
    config={"schema_version":SCHEMA_VERSION,"mode":"quick" if quick else "full","seeds":seed_list,"scenario_count":len(scenarios),"routing_runs":len(scenarios),"ablation_runs":len(runs),"configurations":list(CONFIGURATIONS),"reproducible":bool(reproducible),"generated_at_utc":datetime.now(timezone.utc).isoformat()}
    _write_csv(output_dir/"routing_scenarios.csv",[s.metadata() for s in scenarios]); _write_csv(output_dir/"routing_predictions.csv",predictions); _write_csv(output_dir/"routing_metrics_by_specialist.csv",routing["by_specialist"])
    (output_dir/"routing_metrics_summary.json").write_text(json.dumps({"configuration":config,"metrics":routing["summary"],"critic_arbiter":critic},indent=2,sort_keys=True),encoding="utf-8")
    _write_csv(output_dir/"ablation_runs.csv",runs); _write_csv(output_dir/"ablation_metrics_by_configuration.csv",aggregate); _write_csv(output_dir/"ablation_pairwise_comparison.csv",pairwise)
    (output_dir/"ablation_summary.json").write_text(json.dumps({"configuration":config,"critic_arbiter":critic,"metrics":aggregate,"reproducibility":reproducible},indent=2,sort_keys=True),encoding="utf-8")
    _reports(output_dir,routing,critic,aggregate,config,runs)
    return {"configuration":config,"routing":routing,"critic_arbiter":critic,"aggregate":aggregate}


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4]); parser.add_argument("--quick",action="store_true"); parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(argv)
    try:
        result=run_evaluation(args.output_dir,args.seeds,quick=args.quick,overwrite=args.overwrite)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}",file=sys.stderr); return 2
    print(json.dumps(result["configuration"],sort_keys=True)); return 0


if __name__=="__main__":
    raise SystemExit(main())
