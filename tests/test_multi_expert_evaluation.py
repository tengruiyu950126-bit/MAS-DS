import json

import pandas as pd
import pytest

from agents.multi_expert import RouterAgent
from evaluation.multi_expert_evaluation import (
    CONFIGURATIONS, SCHEMA_VERSION, _oracle_routes, build_plan, faulty_routes,
    run_evaluation,
)
from evaluation.multi_expert_metrics import routing_metrics
from evaluation.multi_expert_scenarios import RoutingScenario, generate_scenarios
from evaluation.streamlit_e2e import ROLLBACK_CONTRACT
from models.orchestration import RoutingDecision
from models.streamlit_e2e import StreamlitE2EResult


def _scenario(expected, *, clean=False, exclusions=()):
    return RoutingScenario(
        scenario_id="known", family="known", seed=0,
        dataframe=pd.DataFrame({"id":["A","B"],"x":[1,2]}),
        problems=(), affected_columns=(), expected_routes=frozenset(expected),
        expected_operations=frozenset(), clean=clean,
        expected_exclusions=tuple(exclusions), protected_columns=tuple(exclusions),
    )


def test_scenario_ground_truth_is_valid_and_independent():
    scenarios=generate_scenarios([0],quick=False)
    assert len(scenarios)>=20
    assert len({s.scenario_id for s in scenarios})==len(scenarios)
    assert all(expert.endswith("Expert") for s in scenarios for _,expert in s.expected_routes)
    assert any(s.multiple_experts for s in scenarios)
    assert any(s.conflicting_signals for s in scenarios)


def test_known_multilabel_metrics():
    scenario=_scenario({("x","NumericExpert"),("x","MissingValueExpert")})
    routes=[RoutingDecision(column="x",selected_expert="NumericExpert",issue_type="test",reason="fixture",confidence=1)]
    _,result=routing_metrics([(scenario,routes)])
    numeric=next(r for r in result["by_specialist"] if r["specialist"]=="NumericExpert")
    missing=next(r for r in result["by_specialist"] if r["specialist"]=="MissingValueExpert")
    assert numeric["precision"]==numeric["recall"]==1
    assert missing["recall"]==0
    assert result["summary"]["multi_label_exact_match_ratio"]==0


def test_routing_schema_parameter_changes_only_schema_fields():
    scenario=_scenario({("x","NumericExpert"),("x","MissingValueExpert")})
    routes=[RoutingDecision(column="x",selected_expert="NumericExpert",issue_type="test",reason="fixture",confidence=1)]
    current_predictions,current=routing_metrics([(scenario,routes)])
    legacy_predictions,legacy=routing_metrics([(scenario,routes)],schema_version="legacy")

    def without_schema(value):
        if isinstance(value,dict):
            return {key:without_schema(item) for key,item in value.items() if key!="schema_version"}
        if isinstance(value,list):
            return [without_schema(item) for item in value]
        return value

    assert without_schema(current_predictions)==without_schema(legacy_predictions)
    assert without_schema(current)==without_schema(legacy)
    assert {row["schema_version"] for row in current_predictions}=={SCHEMA_VERSION}
    assert {row["schema_version"] for row in current["by_specialist"]}=={SCHEMA_VERSION}
    assert current["summary"]["schema_version"]==SCHEMA_VERSION
    assert {row["schema_version"] for row in legacy_predictions}=={"legacy"}
    assert legacy["summary"]["schema_version"]=="legacy"


def test_protected_and_clean_metrics():
    scenario=_scenario(set(),clean=True,exclusions=("id",))
    routes=[RoutingDecision(column="id",selected_expert=None,issue_type="protected",reason="fixture",confidence=1)]
    _,result=routing_metrics([(scenario,routes)])
    assert result["summary"]["protected_column_exclusion_accuracy"]==1
    assert result["summary"]["clean_dataset_no_op_accuracy"]==1


def test_oracle_router_exactly_uses_ground_truth():
    scenario=generate_scenarios([0],quick=True)[0]
    assert {(r.column or "*",r.selected_expert) for r in _oracle_routes(scenario)}==set(scenario.expected_routes)


def test_faulty_router_is_deterministic():
    scenario=generate_scenarios([2],quick=True)[0]
    routes=RouterAgent(scenario.policy()).route(scenario.dataframe)
    assert faulty_routes(routes,scenario)==faulty_routes(routes,scenario)


@pytest.mark.parametrize("configuration",CONFIGURATIONS)
def test_ablation_configuration_isolation(configuration):
    scenario=generate_scenarios([0],quick=True)[0]
    result=build_plan(configuration,scenario)
    assert result.plan is not None
    if configuration=="multi_expert_full":
        assert result.has_arbiter
    if configuration in {"multi_expert_no_arbiter","multi_expert_router_only","rule_baseline"}:
        assert not result.has_arbiter


def test_quick_evaluation_outputs_and_overwrite_protection(tmp_path):
    output=tmp_path/"evaluation"
    first=run_evaluation(output,[0],quick=True)
    assert first["configuration"]["ablation_runs"]==56
    expected={"routing_scenarios.csv","routing_predictions.csv","routing_metrics_by_specialist.csv","routing_metrics_summary.json","routing_evaluation_report.md","ablation_runs.csv","ablation_metrics_by_configuration.csv","ablation_pairwise_comparison.csv","ablation_summary.json","ablation_report.md"}
    assert expected=={p.name for p in output.iterdir()}
    runs = pd.read_csv(output / "ablation_runs.csv")
    assert "expected_operation_coverage" in runs.columns
    assert "repair_success" not in runs.columns
    assert set(runs["schema_version"].astype(str)) == {"2.0"}
    for filename in (
        "routing_scenarios.csv",
        "routing_predictions.csv",
        "routing_metrics_by_specialist.csv",
        "ablation_runs.csv",
        "ablation_metrics_by_configuration.csv",
        "ablation_pairwise_comparison.csv",
    ):
        frame=pd.read_csv(output/filename,dtype={"schema_version":"string"})
        assert set(frame["schema_version"])=={SCHEMA_VERSION}
    routing_summary=json.loads((output/"routing_metrics_summary.json").read_text(encoding="utf-8"))
    ablation_summary=json.loads((output/"ablation_summary.json").read_text(encoding="utf-8"))
    assert routing_summary["configuration"]["schema_version"]==SCHEMA_VERSION
    assert routing_summary["metrics"]["schema_version"]==SCHEMA_VERSION
    assert ablation_summary["configuration"]["schema_version"]==SCHEMA_VERSION
    assert all(row["schema_version"]==SCHEMA_VERSION for row in ablation_summary["metrics"])
    assert scenario_metadata_versions(output)=={SCHEMA_VERSION}
    with pytest.raises(FileExistsError): run_evaluation(output,[0],quick=True)


def scenario_metadata_versions(output):
    """Collect every schema declaration in the current generated bundle."""
    versions=set()
    for path in output.iterdir():
        if path.suffix==".csv":
            frame=pd.read_csv(path,dtype={"schema_version":"string"})
            if "schema_version" in frame:
                versions.update(frame["schema_version"].dropna())
        elif path.suffix==".json":
            def collect(value):
                if isinstance(value,dict):
                    for key,item in value.items():
                        if key=="schema_version":
                            versions.add(str(item))
                        else:
                            collect(item)
                elif isinstance(value,list):
                    for item in value:
                        collect(item)
            collect(json.loads(path.read_text(encoding="utf-8")))
    return versions


def test_evaluation_reproducibility(tmp_path):
    a=run_evaluation(tmp_path/"a",[0],quick=True)
    b=run_evaluation(tmp_path/"b",[0],quick=True)
    assert a["configuration"]["reproducible"] and b["configuration"]["reproducible"]
    pa=pd.read_csv(tmp_path/"a"/"routing_predictions.csv")
    pb=pd.read_csv(tmp_path/"b"/"routing_predictions.csv")
    pd.testing.assert_frame_equal(pa,pb)


def test_outputs_are_value_free(tmp_path):
    output=tmp_path/"out"; run_evaluation(output,[0],quick=True)
    combined="\n".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    assert "short note" not in combined
    assert "R0000" not in combined


def test_streamlit_result_serialization_is_value_free():
    result=StreamlitE2EResult(
        tested_at_utc="2026-01-01T00:00:00Z",python_version="3.13",streamlit_version="1.58",
        rollback_verification_result="passed",passed_scenarios=["application load"],
    )
    payload=json.loads(result.model_dump_json())
    assert payload["passed_scenarios"]==["application load"]
    assert "dataframe" not in payload
    assert json.loads(ROLLBACK_CONTRACT)["columns"]["age"]["numeric_max"]==10
