"""Run value-free Streamlit AppTest checks with the current interpreter."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from models.streamlit_e2e import StreamlitE2EResult
from tools.orchestration_export import orchestration_trace_to_json, orchestration_trace_to_markdown


ROLLBACK_CONTRACT = json.dumps({
    "name":"E2E rollback contract","schema_version":"1.0",
    "columns":{"age":{"numeric_max":10,"severity":"error"}},
})


def _labels(elements) -> set[str]:
    return {str(getattr(element,"label","")) for element in elements}


def _state(app, key: str):
    try:
        return app.session_state[key]
    except KeyError:
        return None


def run_apptest(app_path: str | Path = "app.py") -> StreamlitE2EResult:
    import streamlit
    from streamlit.testing.v1 import AppTest

    attempted=["application load","select Multi-Expert","select built-in contract","render pre-cleaning contract result","render orchestration trace stages","approve and execute successful cleaning","render validation result","expose trace/report/processed downloads","error-level contract rollback","preserve original dataframe after rollback"]
    passed=[]; failed=[]; skipped={"CSV file upload":"AppTest file-uploader payload injection is not supported by the installed framework; the built-in deterministic sample was exercised instead.","download button click/transfer":"AppTest exposes download buttons but does not return their byte payload; underlying generators were invoked directly."}
    warnings=[]
    try:
        success=AppTest.from_file(str(app_path)).run(timeout=45)
        if not success.exception: passed.append("application load")
        success.radio[0].set_value("Multi-Expert"); success.radio[1].set_value("Built-in example"); success.run(timeout=45)
        if success.exception: failed.append("Multi-Expert successful workflow")
        else:
            passed += ["select Multi-Expert","select built-in contract","render pre-cleaning contract result"]
            expander_labels=_labels(success.expander)
            markdown="\n".join(str(x.value) for x in success.markdown)
            if "Multi-Expert Orchestration Trace" in expander_labels and all(term in markdown for term in ("Router decisions","Specialist proposal summary","Critic review","Arbiter final selected steps")):
                passed.append("render orchestration trace stages")
            else: failed.append("render orchestration trace stages")
            success.button[0].click(); success.run(timeout=45)
            if not success.exception and any("Validation passed" in str(x.value) for x in success.success):
                passed += ["approve and execute successful cleaning","render validation result"]
            else: failed.append("approve and execute successful cleaning")
            download_labels=_labels(success.get("download_button"))
            needed={"Download orchestration trace JSON","Download orchestration trace Markdown","Download processed CSV","Download cleaning report Markdown"}
            if needed <= download_labels: passed.append("expose trace/report/processed downloads")
            else: failed.append("expose trace/report/processed downloads")
        trace=_state(success,"orchestration_trace")
        json_bytes=orchestration_trace_to_json(trace).encode(); md_bytes=orchestration_trace_to_markdown(trace).encode()
        outcome=_state(success,"outcome")
        processed_bytes=outcome.dataframe.to_csv(index=False).encode() if outcome is not None else b""
        downloads={"trace_json_nonempty":bool(json_bytes),"trace_markdown_nonempty":bool(md_bytes),"processed_csv_nonempty":bool(processed_bytes),"cleaning_report_button_exposed":"Download cleaning report Markdown" in _labels(success.get("download_button"))}
    except Exception as exc:
        failed.append("successful workflow runtime"); warnings.append(f"{type(exc).__name__}: {str(exc)[:160]}"); downloads={}

    rollback_result="failed"
    try:
        rollback=AppTest.from_file(str(app_path)).run(timeout=45)
        rollback.radio[0].set_value("Multi-Expert"); rollback.radio[1].set_value("Paste JSON"); rollback.run(timeout=45)
        contract_area=next(area for area in rollback.text_area if area.label=="Contract JSON")
        contract_area.set_value(ROLLBACK_CONTRACT); rollback.run(timeout=45)
        rollback.button[0].click(); rollback.run(timeout=45)
        outcome=_state(rollback,"outcome")
        if not rollback.exception and outcome is not None and outcome.rolled_back and outcome.contract_caused_rollback:
            from evaluation.datasets import load_dataset, make_dirty_copy
            original=make_dirty_copy(load_dataset("demo",rows=60).dataframe)
            if outcome.dataframe.equals(original):
                rollback_result="passed: original dataframe preserved"; passed += ["error-level contract rollback","preserve original dataframe after rollback"]
            else:
                rollback_result="failed: rollback dataframe differed"; failed.append("preserve original dataframe after rollback")
        else:
            failed.append("error-level contract rollback")
    except Exception as exc:
        warnings.append(f"Rollback {type(exc).__name__}: {str(exc)[:160]}"); failed.append("error-level contract rollback")
    return StreamlitE2EResult(
        tested_at_utc=datetime.now(timezone.utc),python_version=platform.python_version(),streamlit_version=streamlit.__version__,
        automated_ui_scenarios_attempted=attempted,passed_scenarios=passed,failed_scenarios=failed,skipped_scenarios=skipped,
        rollback_verification_result=rollback_result,download_generation_verification=downloads,warnings=warnings,
    )


def write_result(path:str|Path,result:StreamlitE2EResult)->None:
    Path(path).write_text(result.model_dump_json(indent=2),encoding="utf-8")
