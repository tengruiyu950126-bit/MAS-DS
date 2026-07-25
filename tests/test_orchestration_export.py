import json

import pandas as pd

from agents.multi_expert import MultiExpertCleaningAgent
from models.policy import PreprocessingPolicy
from tools.orchestration_export import (
    orchestration_trace_to_json,
    orchestration_trace_to_markdown,
    safe_orchestration_trace_filename,
)
from tools.provenance import build_audit_provenance
from tools.ui_tables import (
    arbiter_trace_frame,
    critic_trace_frame,
    router_trace_frame,
    specialist_trace_frame,
)


def build_trace():
    dataframe = pd.DataFrame(
        {
            "row_id": [1, 2, 2, 4],
            "amount": ["10", None, None, "30"],
            "city": ["Singapore", "singapore", "singapore", None],
        }
    )
    agent = MultiExpertCleaningAgent()
    agent.propose(dataframe)
    assert agent.last_trace is not None
    return agent.last_trace


def test_json_export_contains_all_orchestration_sections() -> None:
    trace = build_trace()
    dataframe = pd.DataFrame({"value": [1, 2]})
    provenance = build_audit_provenance(
        dataframe=dataframe,
        plan=trace.final_plan,
        policy=PreprocessingPolicy(),
        trace=trace,
        source_label="sample:demo:dirty",
        planner_name="Multi-Expert",
        planner_mode="Multi-Expert",
    )
    payload = json.loads(orchestration_trace_to_json(trace, provenance))

    assert payload["schema_version"] == 1
    assert payload["available"] is True
    assert payload["router_decisions"]
    assert "NumericExpert" in payload["specialist_proposals"]
    assert set(payload["critic"]) == {
        "approved_steps",
        "rejected_steps",
        "warnings",
        "reasons",
    }
    assert payload["arbiter"]["final_plan"]["steps"]
    assert payload["summary"]["final_steps"] == len(
        payload["arbiter"]["final_plan"]["steps"]
    )
    assert payload["provenance"]["dataset_fingerprint"]
    assert payload["provenance"]["cleaning_plan_fingerprint"]
    assert payload["provenance"]["policy_fingerprint"]
    assert payload["provenance"]["orchestration_trace_fingerprint"]


def test_markdown_export_contains_clear_audit_headings() -> None:
    markdown = orchestration_trace_to_markdown(build_trace())

    assert "# MAS-DS Deterministic Routed-Planner Trace" in markdown
    assert "## Summary" in markdown
    assert "## Audit Provenance" in markdown
    assert "## Router Decisions" in markdown
    assert "## Specialist Proposals" in markdown
    assert "## Critic Review" in markdown
    assert "### Approved Steps" in markdown
    assert "### Rejected Steps" in markdown
    assert "### Warnings" in markdown
    assert "## Arbiter Final Plan" in markdown
    assert "### Selected Steps" in markdown


def test_missing_trace_exports_are_safe_and_explicit() -> None:
    payload = json.loads(orchestration_trace_to_json(None))
    markdown = orchestration_trace_to_markdown(None)

    assert payload["available"] is False
    assert payload["router_decisions"] == []
    assert payload["specialist_proposals"] == {}
    assert payload["critic"]["approved_steps"] == []
    assert payload["arbiter"]["final_plan"]["steps"] == []
    assert "No orchestration trace was available" in markdown


def test_safe_filename_removes_path_and_unsafe_characters() -> None:
    filename = safe_orchestration_trace_filename(
        "../../upload:risky data?.csv",
        ".JSON",
    )

    assert filename == "mas_ds_orchestration_trace_upload_risky_data_csv.json"
    assert "/" not in filename
    assert "\\" not in filename
    assert ".." not in filename


def test_existing_trace_table_helpers_accept_exported_trace_source() -> None:
    trace = build_trace()

    assert not router_trace_frame(trace).empty
    assert not specialist_trace_frame(trace).empty
    assert not critic_trace_frame(trace).empty
    assert not arbiter_trace_frame(trace).empty
