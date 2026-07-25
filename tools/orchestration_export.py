"""Local JSON and Markdown exports for multi-expert planning traces."""

from __future__ import annotations

import json
import re
from typing import Any

from models.cleaning_plan import CleaningStep
from models.orchestration import OrchestrationTrace
from models.provenance import AuditProvenance


def _provenance_payload(
    provenance: AuditProvenance | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if provenance is None:
        return None
    if isinstance(provenance, AuditProvenance):
        return provenance.model_dump(mode="json")
    return dict(provenance)


def _trace_payload(trace: OrchestrationTrace | None) -> dict[str, Any]:
    """Build a stable, versioned audit envelope for machine consumption."""
    if trace is None:
        return {
            "schema_version": 1,
            "available": False,
            "summary": {
                "router_decisions": 0,
                "specialists": 0,
                "specialist_proposals": 0,
                "critic_approved": 0,
                "critic_rejected": 0,
                "critic_warnings": 0,
                "final_steps": 0,
            },
            "router_decisions": [],
            "specialist_proposals": {},
            "critic": {
                "approved_steps": [],
                "rejected_steps": [],
                "warnings": [],
                "reasons": [],
            },
            "arbiter": {"final_plan": {"steps": []}},
        }

    proposals = {
        expert: [step.model_dump(mode="json") for step in steps]
        for expert, steps in trace.specialist_proposals.items()
    }
    return {
        "schema_version": 1,
        "available": True,
        "summary": {
            "router_decisions": len(trace.routes),
            "specialists": len(proposals),
            "specialist_proposals": sum(len(steps) for steps in proposals.values()),
            "critic_approved": len(trace.critique.approved_steps),
            "critic_rejected": len(trace.critique.rejected_steps),
            "critic_warnings": len(trace.critique.warnings),
            "final_steps": len(trace.final_plan.steps),
        },
        "router_decisions": [
            route.model_dump(mode="json") for route in trace.routes
        ],
        "specialist_proposals": proposals,
        "critic": trace.critique.model_dump(mode="json"),
        "arbiter": {
            "final_plan": trace.final_plan.model_dump(mode="json"),
        },
    }


def orchestration_trace_to_json(
    trace: OrchestrationTrace | None,
    provenance: AuditProvenance | dict[str, Any] | None = None,
) -> str:
    """Serialize a trace as structured, machine-readable JSON."""
    payload = _trace_payload(trace)
    payload["provenance"] = _provenance_payload(provenance)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No entries._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_markdown_value(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _steps_table(steps: list[CleaningStep]) -> str:
    return _markdown_table(
        ["Operation", "Column", "Confidence", "Reason"],
        [
            [
                step.operation,
                step.column or "*",
                f"{step.confidence:.3f}",
                step.reason,
            ]
            for step in steps
        ],
    )


def _provenance_markdown(
    provenance: AuditProvenance | dict[str, Any] | None,
) -> str:
    payload = _provenance_payload(provenance)
    if payload is None:
        return "_Audit provenance was not provided._"
    ordered_fields = [
        "schema_version",
        "created_at_utc",
        "source_label",
        "planner_name",
        "planner_mode",
        "dataset_rows",
        "dataset_columns",
        "dataset_fingerprint",
        "cleaning_plan_fingerprint",
        "policy_fingerprint",
        "orchestration_trace_fingerprint",
        "contract_fingerprint",
        "final_plan_steps",
        "notes",
    ]
    rows = []
    for field in ordered_fields:
        value = payload.get(field)
        if isinstance(value, list):
            value = "<br>".join(str(item) for item in value) or "none"
        rows.append([field, value])
    return _markdown_table(["Field", "Value"], rows)


def orchestration_trace_to_markdown(
    trace: OrchestrationTrace | None,
    provenance: AuditProvenance | dict[str, Any] | None = None,
) -> str:
    """Render a self-contained, human-readable orchestration audit."""
    title = "# MAS-DS Deterministic Routed-Planner Trace"
    if trace is None:
        return "\n".join(
            [
                title,
                "",
                "## Summary",
                "",
                "No orchestration trace was available for this proposal.",
                "",
                "## Audit Provenance",
                "",
                _provenance_markdown(provenance),
                "",
                "## Router Decisions",
                "",
                "_No entries._",
                "",
                "## Specialist Proposals",
                "",
                "_No entries._",
                "",
                "## Critic Review",
                "",
                "_No entries._",
                "",
                "## Arbiter Final Plan",
                "",
                "_No entries._",
                "",
            ]
        )

    proposal_count = sum(
        len(steps) for steps in trace.specialist_proposals.values()
    )
    lines = [
        title,
        "",
        "This read-only audit was generated locally by MAS-DS. It is planning "
        "evidence and is not an execution input.",
        "",
        "## Summary",
        "",
        _markdown_table(
            [
                "Router decisions",
                "Specialists",
                "Proposals",
                "Critic approved",
                "Critic rejected",
                "Warnings",
                "Final steps",
            ],
            [[
                len(trace.routes),
                len(trace.specialist_proposals),
                proposal_count,
                len(trace.critique.approved_steps),
                len(trace.critique.rejected_steps),
                len(trace.critique.warnings),
                len(trace.final_plan.steps),
            ]],
        ),
        "",
        "## Audit Provenance",
        "",
        _provenance_markdown(provenance),
        "",
        "## Router Decisions",
        "",
        _markdown_table(
            ["Column", "Selected expert", "Issue type", "Confidence", "Reason"],
            [
                [
                    route.column or "*",
                    route.selected_expert or "Not routed",
                    route.issue_type,
                    f"{route.confidence:.3f}",
                    route.reason,
                ]
                for route in trace.routes
            ],
        ),
        "",
        "## Specialist Proposals",
        "",
    ]
    if not trace.specialist_proposals:
        lines.extend(["_No entries._", ""])
    else:
        for expert, steps in trace.specialist_proposals.items():
            lines.extend([f"### {expert}", "", _steps_table(steps), ""])

    lines.extend(
        [
            "## Critic Review",
            "",
            "### Approved Steps",
            "",
            _steps_table(trace.critique.approved_steps),
            "",
            "### Rejected Steps",
            "",
            _steps_table(trace.critique.rejected_steps),
            "",
            "### Warnings",
            "",
            "\n".join(f"- {_markdown_value(item)}" for item in trace.critique.warnings)
            or "_No warnings._",
            "",
            "### Reasons",
            "",
            "\n".join(f"- {_markdown_value(item)}" for item in trace.critique.reasons)
            or "_No additional reasons._",
            "",
            "## Arbiter Final Plan",
            "",
            "### Selected Steps",
            "",
            _steps_table(trace.final_plan.steps),
            "",
        ]
    )
    return "\n".join(lines)


def safe_orchestration_trace_filename(source_label: str, extension: str) -> str:
    """Return a path-free, length-bounded download filename."""
    source = re.sub(r"[^A-Za-z0-9_-]+", "_", str(source_label)).strip("_-")
    source = source[:80].rstrip("_-") or "run"
    safe_extension = re.sub(r"[^A-Za-z0-9]+", "", str(extension)).lower() or "txt"
    safe_extension = safe_extension[:10]
    return f"mas_ds_orchestration_trace_{source}.{safe_extension}"
