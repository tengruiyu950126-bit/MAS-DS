"""Streamlit entry point for MAS-DS."""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.hybrid_cleaning_agent import HybridCleaningAgent
from agents.local_llm_cleaning_agent import (
    LocalLLMCleaningAgent,
    LocalLLMPlanningError,
)
from agents.multi_expert import MultiExpertCleaningAgent
from agents.planner_factory import build_offline_planner
from evaluation.datasets import available_dataset_names, load_dataset, make_dirty_copy
from models.data_contract import BUILTIN_EXAMPLE_CONTRACT, DataContract
from models.policy import PreprocessingPolicy
from providers.ollama import OllamaClient, OllamaError
from tools.policy import (
    PolicyLoadError,
    normalize_column_list,
    policy_from_json,
    policy_to_json,
    policy_with_contract,
)
from tools.data_contract import ContractLoadError, contract_from_json, contract_to_json
from tools.provenance import build_audit_provenance
from tools.diff import diff_summary_frame, plan_diff_frame
from tools.orchestration_export import (
    orchestration_trace_to_json,
    orchestration_trace_to_markdown,
    safe_orchestration_trace_filename,
)
from tools.reporting import build_cleaning_report, build_report_filename
from tools.ingestion import (
    CSVIngestionError,
    DEFAULT_CSV_LIMITS,
    load_csv_bytes,
)
from tools.export_safety import neutralize_spreadsheet_formulas
from tools.ui_tables import (
    arbiter_trace_frame,
    changed_columns_frame,
    contract_findings_frame,
    contract_summary_frame,
    contract_validation_summary_frame,
    critic_trace_frame,
    dataset_metrics_frame,
    execution_records_frame,
    operation_summary_frame,
    plan_frame,
    plan_summary_frame,
    policy_summary_frame,
    profile_frame,
    router_trace_frame,
    specialist_trace_frame,
    validation_issues_frame,
    validation_summary_frame,
)
from agents.orchestrator import PreprocessingOrchestrator


POLICY_OPERATIONS = [
    "drop_duplicates",
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
]


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_OVERVIEW_PATH = PROJECT_ROOT / "outputs" / "experiment_summary_overview.csv"
EXPERIMENT_REPORT_PATH = PROJECT_ROOT / "outputs" / "experiment_summary_report.md"


def _dataframe_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .mas-hero {
            padding: 1.1rem 1.25rem;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            background: linear-gradient(135deg, #F8FBFF 0%, #F4F7FB 100%);
            margin-bottom: 1rem;
        }
        .mas-hero h1 {
            margin: 0 0 0.25rem 0;
            color: #0B2545;
        }
        .mas-hero p {
            margin: 0.2rem 0 0 0;
            color: #475569;
        }
        .mas-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid #D5E3F7;
            background: #EEF6FF;
            color: #1F4D78;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-top: 0.55rem;
        }
        .mas-callout {
            padding: 0.85rem 1rem;
            border-left: 4px solid #2E74B5;
            border-radius: 10px;
            background: #F8FAFC;
            color: #334155;
            margin: 0.5rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_header() -> None:
    st.markdown(
        """
        <div class="mas-hero">
            <h1>MAS-DS Data Preprocessing</h1>
            <p>
                A local policy-constrained data-cleaning system with deterministic
                execution, validation, and rollback.
            </p>
            <span class="mas-badge">$0 local rule mode</span>
            <span class="mas-badge">Human approval</span>
            <span class="mas-badge">Validation + rollback</span>
            <span class="mas-badge">Cell-level change audit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_sidebar_guide() -> None:
    with st.sidebar.expander("How to use this app", expanded=False):
        st.markdown(
            """
            1. Choose a sample dataset or upload CSV.
            2. Select the cleaning expert.
            3. Review the generated cleaning plan.
            4. Approve only if the plan looks safe.
            5. Inspect validation and change audit.
            6. Download cleaned CSV, audit CSV, and run report.
            """
        )


def _metric_cards(frame: pd.DataFrame, label_map: dict[str, str]) -> None:
    values = frame.iloc[0].to_dict()
    columns = st.columns(len(values))
    for column, (key, value) in zip(columns, values.items(), strict=True):
        label = label_map.get(key, key.replace("_", " ").title())
        column.metric(label, value)


def _show_policy_summary(policy: PreprocessingPolicy) -> None:
    with st.sidebar.expander("Active policy summary", expanded=False):
        st.dataframe(
            policy_summary_frame(policy),
            use_container_width=True,
            hide_index=True,
        )


def _build_policy() -> PreprocessingPolicy:
    with st.sidebar.expander("Safety policy", expanded=False):
        uploaded_policy = st.file_uploader(
            "Upload policy JSON",
            type=["json"],
            key="policy_json_upload",
            help="Optional: load a saved PreprocessingPolicy JSON file.",
        )
        base_policy = PreprocessingPolicy()
        if uploaded_policy is not None:
            try:
                base_policy = policy_from_json(uploaded_policy.getvalue())
                st.success("Policy JSON loaded.")
            except PolicyLoadError as exc:
                st.error(str(exc))

        protect_identifier_columns = st.checkbox(
            "Auto-protect ID columns",
            value=base_policy.protect_identifier_columns,
            help="Protects columns such as id, customer_id, order_id, uuid, and guid.",
        )
        protected_text = st.text_area(
            "Additional protected columns",
            value=", ".join(base_policy.protected_columns),
            placeholder="Example: email, phone_number",
            help="Comma-separated or one column per line.",
        )
        flag_outliers = st.checkbox(
            "Flag numeric outliers",
            value=base_policy.outlier_action == "flag",
            help="Advisory only: outlier values are not changed automatically.",
        )
        denied_operations = st.multiselect(
            "Disable operations",
            POLICY_OPERATIONS,
            default=[
                operation
                for operation in base_policy.denied_operations
                if operation in POLICY_OPERATIONS
            ],
            help="Use this to make the planner more conservative.",
        )
        if not flag_outliers and "flag_outliers_iqr" not in denied_operations:
            denied_operations = [*denied_operations, "flag_outliers_iqr"]

        policy = base_policy.model_copy(
            update={
                "protect_identifier_columns": protect_identifier_columns,
                "protected_columns": normalize_column_list(protected_text),
                "denied_operations": denied_operations,
                "outlier_action": "flag" if flag_outliers else "ignore",
            }
        )
        st.download_button(
            "Download current policy JSON",
            data=policy_to_json(policy),
            file_name="mas_ds_policy.json",
            mime="application/json",
        )
    return policy


def _build_contract() -> DataContract | None:
    with st.sidebar.expander("Data contract", expanded=False):
        mode = st.radio(
            "Contract source",
            ["No contract", "Built-in example", "Paste JSON", "Upload JSON"],
            key="contract_source",
        )
        if mode == "No contract":
            st.caption("No semantic contract is active; existing behavior is unchanged.")
            return None
        if mode == "Built-in example":
            st.caption("Warning-only example with common protected identifier columns.")
            st.code(contract_to_json(BUILTIN_EXAMPLE_CONTRACT), language="json")
            return BUILTIN_EXAMPLE_CONTRACT.model_copy(deep=True)
        raw: str | bytes | None = None
        if mode == "Paste JSON":
            raw = st.text_area(
                "Contract JSON",
                value="",
                height=180,
                placeholder='{"name": "Customer contract", "required_columns": ["customer_id"]}',
            )
            if not raw.strip():
                st.info("Paste a contract JSON object to activate validation.")
                return None
        else:
            uploaded = st.file_uploader(
                "Upload contract JSON", type=["json"], key="contract_json_upload"
            )
            if uploaded is None:
                st.info("Upload a local JSON contract to activate validation.")
                return None
            raw = uploaded.getvalue()
        try:
            contract = contract_from_json(raw)
        except ContractLoadError as exc:
            st.error(str(exc))
            return None
        st.success(f"Contract loaded: {contract.name} (schema {contract.schema_version})")
        return contract


def _build_orchestrator(
    planner_mode: str,
    policy: PreprocessingPolicy,
    contract: DataContract | None = None,
) -> tuple[
    PreprocessingOrchestrator,
    str,
    MultiExpertCleaningAgent | None,
]:
    if planner_mode in {"Rule-based baseline", "Deterministic routed planner"}:
        planner = build_offline_planner(planner_mode, policy, contract)
        multi_expert_agent = (
            planner if isinstance(planner, MultiExpertCleaningAgent) else None
        )
        return (
            PreprocessingOrchestrator(
                cleaning_agent=planner,
                policy=policy,
                contract=contract,
            ),
            planner_mode,
            multi_expert_agent,
        )

    planner_name = "Rule-based baseline"

    ollama_model = st.sidebar.text_input(
        "Ollama model",
        value=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        placeholder="Example: your-installed-model:tag",
    )
    ollama_url = st.sidebar.text_input(
        "Ollama URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    if not ollama_model.strip():
        st.sidebar.warning("Enter an installed Ollama model; using rule baseline.")
        return PreprocessingOrchestrator(policy=policy, contract=contract), planner_name, None

    remote_opt_in = st.sidebar.checkbox(
        "Allow a non-local model endpoint",
        value=False,
        help=(
            "Remote endpoints receive dataset metadata and any explicitly "
            "enabled sample rows. Leave disabled for local-only processing."
        ),
    )
    include_sample_rows = st.sidebar.checkbox(
        "Send up to 3 sample rows to the model",
        value=False,
        help=(
            "Off by default. When off, model planning sends profile metadata "
            "and column names but no dataframe cell values."
        ),
    )
    llm_agent = LocalLLMCleaningAgent(
        OllamaClient(
            model=ollama_model.strip(),
            base_url=ollama_url.strip(),
            allow_remote=remote_opt_in,
        ),
        sample_rows=3 if include_sample_rows else 0,
        policy=policy,
    )
    if planner_mode == "Hybrid rule + Ollama":
        return (
            PreprocessingOrchestrator(
                cleaning_agent=HybridCleaningAgent(llm_agent, policy=policy),
                policy=policy,
                contract=contract,
            ),
            f"Hybrid rule + Ollama: {ollama_model.strip()}",
            None,
        )
    return (
        PreprocessingOrchestrator(cleaning_agent=llm_agent, policy=policy, contract=contract),
        f"Local Ollama: {ollama_model.strip()}",
        None,
    )


def _load_data_source() -> tuple[pd.DataFrame | None, bytes | None, str]:
    source_mode = st.sidebar.radio("Data source", ["Sample dataset", "Upload CSV"])
    if source_mode == "Sample dataset":
        dataset_name = st.sidebar.selectbox(
            "Sample dataset",
            available_dataset_names(),
            index=0,
        )
        rows = st.sidebar.slider("Sample rows", min_value=20, max_value=200, value=60)
        inject_demo_issues = st.sidebar.checkbox(
            "Inject demo issues",
            value=True,
            help="Adds deterministic missing values and duplicate rows for demos.",
        )
        dataset = load_dataset(dataset_name, rows=rows)
        dataframe = (
            make_dirty_copy(dataset.dataframe)
            if inject_demo_issues
            else dataset.dataframe
        )
        issue_label = "dirty" if inject_demo_issues else "clean"
        return (
            dataframe,
            _dataframe_bytes(dataframe),
            f"sample:{dataset.name}:{issue_label}",
        )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        return None, None, "upload:empty"
    file_bytes = uploaded_file.getvalue()
    try:
        dataframe = load_csv_bytes(file_bytes)
    except CSVIngestionError as exc:
        st.error(str(exc))
        st.info(
            "For larger files, use `python -m scripts.run_chunked_preprocessing`."
        )
        return None, None, "upload:rejected"
    return dataframe, file_bytes, f"upload:{uploaded_file.name}"


def _show_workflow(planner_name: str) -> None:
    st.subheader("Preprocessing workflow")
    columns = st.columns(4)
    steps = [
        ("1. Profile", "Scans schema, missing values, duplicates, dtypes."),
        ("2. Cleaning planner", planner_name),
        ("3. Human approval", "You review the proposed actions before execution."),
        ("4. Validation & rollback", "Commits safe results or rolls back risky changes."),
    ]
    for column, (title, body) in zip(columns, steps, strict=True):
        with column:
            st.markdown(f"**{title}**")
            st.caption(body)


def _show_readiness(
    *,
    source_label: str,
    planner_name: str,
    policy: PreprocessingPolicy,
) -> None:
    protected = "on" if policy.protect_identifier_columns else "off"
    denied = len(policy.denied_operations)
    st.markdown(
        f"""
        <div class="mas-callout">
        <strong>Current run:</strong> {html.escape(source_label)} ·
        <strong>Planner:</strong> {html.escape(planner_name)} ·
        <strong>ID protection:</strong> {protected} ·
        <strong>Disabled ops:</strong> {denied}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_experiment_summary() -> None:
    st.subheader("Experiment summary")
    if not EXPERIMENT_OVERVIEW_PATH.exists():
        st.info(
            "No experiment summary found yet. Run "
            "`python -m evaluation.summarize_outputs` to generate it."
        )
        return

    overview = pd.read_csv(EXPERIMENT_OVERVIEW_PATH)
    st.caption(
        "Aggregated from local CSV summaries in outputs/. This does not call any API."
    )
    st.dataframe(overview, use_container_width=True)
    st.download_button(
        "Download experiment overview CSV",
        data=EXPERIMENT_OVERVIEW_PATH.read_bytes(),
        file_name="experiment_summary_overview.csv",
        mime="text/csv",
    )

    if EXPERIMENT_REPORT_PATH.exists():
        with st.expander("Read generated Markdown report", expanded=False):
            st.markdown(EXPERIMENT_REPORT_PATH.read_text(encoding="utf-8"))
        st.download_button(
            "Download experiment summary report",
            data=EXPERIMENT_REPORT_PATH.read_bytes(),
            file_name="experiment_summary_report.md",
            mime="text/markdown",
        )


def _show_multi_expert_trace(
    trace: object | None,
    source_label: str,
    provenance: object | None,
) -> None:
    """Render a read-only orchestration audit; never alter the plan."""
    with st.expander("Deterministic Planner Trace", expanded=False):
        if trace is None:
            st.info(
                "No orchestration trace is available for this proposal. "
                "Generate a plan with Deterministic routed planner mode to view it."
            )
            return

        st.caption(
            "Read-only planning evidence. These tables do not change the "
            "approved plan or execution path."
        )
        json_column, markdown_column = st.columns(2)
        with json_column:
            st.download_button(
                "Download orchestration trace JSON",
                data=orchestration_trace_to_json(trace, provenance),
                file_name=safe_orchestration_trace_filename(
                    source_label,
                    "json",
                ),
                mime="application/json",
            )
        with markdown_column:
            st.download_button(
                "Download orchestration trace Markdown",
                data=orchestration_trace_to_markdown(trace, provenance),
                file_name=safe_orchestration_trace_filename(
                    source_label,
                    "md",
                ),
                mime="text/markdown",
            )
        st.markdown("#### Router decisions")
        router = router_trace_frame(trace)
        if router.empty:
            st.info("The router produced no decisions for this dataset.")
        else:
            st.dataframe(router, use_container_width=True, hide_index=True)

        st.markdown("#### Specialist proposal summary")
        specialists = specialist_trace_frame(trace)
        if specialists.empty:
            st.info("No specialist proposal information is available.")
        else:
            st.dataframe(specialists, use_container_width=True, hide_index=True)

        st.markdown("#### Critic review")
        critic = critic_trace_frame(trace)
        if critic.empty:
            st.info("The critic had no proposals or warnings to report.")
        else:
            st.dataframe(critic, use_container_width=True, hide_index=True)

        st.markdown("#### Arbiter final selected steps")
        arbiter = arbiter_trace_frame(trace)
        if arbiter.empty:
            st.info("The arbiter selected no cleaning steps.")
        else:
            st.dataframe(arbiter, use_container_width=True, hide_index=True)


st.set_page_config(page_title="MAS-DS", page_icon="M", layout="wide")
_inject_styles()
_show_header()

st.sidebar.header("Configuration")
_show_sidebar_guide()
planner_mode = st.sidebar.radio(
    "Cleaning expert",
    [
        "Rule-based baseline",
        "Local Ollama model",
        "Hybrid rule + Ollama",
        "Deterministic routed planner",
    ],
)
st.sidebar.info(
    "Rule-based and deterministic routed modes do not call a model. "
    "Ollama-compatible modes send metadata to the configured endpoint; "
    "sample cells are off by default."
)

policy = _build_policy()
contract = _build_contract()
policy = policy_with_contract(policy, contract)
_show_policy_summary(policy)
orchestrator, planner_name, multi_expert_agent = _build_orchestrator(
    planner_mode,
    policy,
    contract,
)
dataframe, file_bytes, source_label = _load_data_source()

if dataframe is None or file_bytes is None:
    st.info("Choose a sample dataset or upload a CSV file to begin.")
    st.stop()

proposal_key = (
    hashlib.sha256(file_bytes).hexdigest(),
    source_label,
    planner_mode,
    planner_name,
    policy.model_dump_json(),
    contract.model_dump_json() if contract is not None else "no-contract",
)

try:
    if st.session_state.get("proposal_key") != proposal_key:
        with st.spinner(f"Generating plan with {planner_name}..."):
            proposal = orchestrator.propose(dataframe)
        st.session_state["proposal_key"] = proposal_key
        st.session_state["proposal"] = proposal
        st.session_state["orchestration_trace"] = (
            multi_expert_agent.last_trace
            if multi_expert_agent is not None
            else None
        )
        st.session_state["audit_provenance"] = build_audit_provenance(
            dataframe=dataframe,
            plan=proposal.plan,
            policy=policy,
            trace=st.session_state["orchestration_trace"],
            source_label=source_label,
            planner_name=planner_name,
            planner_mode=planner_mode,
            contract=contract,
        )
        st.session_state.pop("outcome_key", None)
        st.session_state.pop("outcome", None)
    else:
        proposal = st.session_state["proposal"]
    orchestration_trace = st.session_state.get("orchestration_trace")
    audit_provenance = st.session_state.get("audit_provenance")
except (OllamaError, LocalLLMPlanningError, ValueError) as exc:
    st.error(f"Local planner failed: {exc}")
    st.info("Falling back to the free rule-based cleaning expert.")
    orchestrator = PreprocessingOrchestrator(policy=policy, contract=contract)
    planner_name = "Rule-based fallback"
    proposal = orchestrator.propose(dataframe)
    st.session_state["proposal_key"] = proposal_key
    st.session_state["proposal"] = proposal
    st.session_state["orchestration_trace"] = None
    st.session_state["audit_provenance"] = build_audit_provenance(
        dataframe=dataframe,
        plan=proposal.plan,
        policy=policy,
        trace=None,
        source_label=source_label,
        planner_name=planner_name,
        planner_mode=planner_mode,
        contract=contract,
    )
    orchestration_trace = None
    audit_provenance = st.session_state["audit_provenance"]

_show_readiness(source_label=source_label, planner_name=planner_name, policy=policy)
_show_workflow(planner_name)

st.subheader("Dataset overview")
_metric_cards(dataset_metrics_frame(dataframe), {})

preview_tab, profile_tab, contract_tab, plan_tab, execute_tab, experiment_tab = st.tabs(
    [
        "Preview",
        "Profile",
        "Data contract",
        "Cleaning plan",
        "Execute & validate",
        "Experiment summary",
    ]
)

with preview_tab:
    st.dataframe(
        dataframe.head(DEFAULT_CSV_LIMITS.preview_rows),
        use_container_width=True,
    )

with profile_tab:
    st.caption(
        f"Rows: {proposal.profile.rows} | Columns: {proposal.profile.columns} | "
        f"Duplicate rows: {proposal.profile.duplicate_rows}"
    )
    st.dataframe(profile_frame(proposal.profile), use_container_width=True)

with contract_tab:
    st.dataframe(contract_summary_frame(contract), use_container_width=True, hide_index=True)
    if contract is None:
        st.info("No data contract is active. Select one in the sidebar to enable semantic validation.")
    else:
        pre_contract_validation = proposal.contract_validation
        st.markdown("#### Pre-cleaning validation")
        st.dataframe(contract_validation_summary_frame(pre_contract_validation), use_container_width=True, hide_index=True)
        pre_findings = contract_findings_frame(pre_contract_validation)
        if pre_findings.empty:
            st.success("The input dataframe satisfies the active contract.")
        else:
            st.dataframe(pre_findings, use_container_width=True, hide_index=True)
        if pre_contract_validation and pre_contract_validation.block_execution:
            st.error("Structural contract errors block execution. Repair the input schema or update the contract first.")
        elif pre_contract_validation and pre_contract_validation.error_count:
            st.warning("Error-level semantic violations exist, but cleaning may attempt repair. Remaining post-cleaning errors will trigger rollback.")

with plan_tab:
    st.caption(f"Planner used: {planner_name}")
    if planner_mode == "Deterministic routed planner":
        _show_multi_expert_trace(
            orchestration_trace,
            source_label,
            audit_provenance,
        )
    if not proposal.plan.steps:
        st.success("No high-confidence cleaning operation is required.")
    else:
        st.markdown("#### Plan summary")
        _metric_cards(
            plan_summary_frame(proposal.plan),
            {
                "planned_steps": "Planned steps",
                "mutating_steps": "Mutating steps",
                "advisory_steps": "Advisory steps",
                "targeted_columns": "Targeted columns",
                "avg_confidence": "Avg confidence",
            },
        )
        st.markdown("#### Operation mix")
        st.dataframe(
            operation_summary_frame(proposal.plan),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Detailed plan")
        st.dataframe(plan_frame(proposal.plan), use_container_width=True)
        st.warning(
            "Review the plan before applying it. Execution uses only whitelisted "
            "operations and never mutates the original file."
        )

with execute_tab:
    if not proposal.plan.steps:
        st.success("Nothing to apply for this dataset.")
    else:
        execution_blocked = bool(
            proposal.contract_validation is not None
            and proposal.contract_validation.block_execution
        )
        if execution_blocked:
            st.error("Execution is blocked by structural data-contract errors.")
        if st.button("Approve and apply plan", type="primary", disabled=execution_blocked):
            with st.spinner("Executing plan and validating result..."):
                st.session_state["outcome"] = orchestrator.execute_approved(
                    dataframe,
                    proposal.plan,
                    run_id=proposal.run_id,
                )
                st.session_state["outcome_key"] = proposal_key

        outcome = (
            st.session_state.get("outcome")
            if st.session_state.get("outcome_key") == proposal_key
            else None
        )
        if outcome is not None:
            if outcome.rolled_back:
                st.error("Validation failed. Changes were rolled back.")
            else:
                st.success("Validation passed. Changes were committed.")
            if contract is not None:
                st.markdown("#### Post-cleaning contract validation")
                st.dataframe(contract_validation_summary_frame(outcome.contract_validation), use_container_width=True, hide_index=True)
                post_findings = contract_findings_frame(outcome.contract_validation)
                if post_findings.empty:
                    st.success("The processed dataframe satisfies the active contract.")
                else:
                    st.dataframe(post_findings, use_container_width=True, hide_index=True)
                if outcome.contract_caused_rollback:
                    st.error("Rollback was caused by an error-level data-contract violation.")

            diff = plan_diff_frame(
                dataframe,
                proposal.plan,
                max_total_changes=DEFAULT_CSV_LIMITS.max_diff_rows,
            )
            result_frame = pd.DataFrame(
                [
                    {
                        "committed": not outcome.rolled_back,
                        "rolled_back": outcome.rolled_back,
                        "validation_issues": len(outcome.validation.issues),
                        "audit_rows": len(diff),
                    }
                ]
            )
            st.markdown("#### Result summary")
            _metric_cards(
                result_frame,
                {
                    "committed": "Committed",
                    "rolled_back": "Rolled back",
                    "validation_issues": "Validation issues",
                    "audit_rows": "Audit rows",
                },
            )

            st.markdown("#### Validation summary")
            st.dataframe(
                validation_summary_frame(outcome.validation),
                use_container_width=True,
            )

            issues = validation_issues_frame(outcome.validation)
            if not issues.empty:
                st.markdown("#### Validation issues")
                st.dataframe(issues, use_container_width=True)

            st.markdown("#### Execution audit")
            st.dataframe(
                execution_records_frame(outcome.execution_records),
                use_container_width=True,
            )

            st.markdown("#### Before / after change audit")
            if outcome.rolled_back:
                st.caption(
                    "Validation rolled the result back. The table below shows "
                    "the candidate changes that would have been applied before "
                    "rollback, so you can inspect what the plan attempted."
                )
            else:
                st.caption(
                    "Cell-level and row-level changes attributed to each approved "
                    "cleaning step."
                )
            if diff.empty:
                st.info("No cell-level, row-level, or dtype changes were detected.")
            else:
                st.markdown("##### Change summary")
                st.dataframe(diff_summary_frame(diff), use_container_width=True)
                st.markdown("##### Detailed changes")
                st.dataframe(diff, use_container_width=True, height=320)
                st.download_button(
                    "Download change audit CSV",
                    data=neutralize_spreadsheet_formulas(diff)
                    .to_csv(index=False)
                    .encode("utf-8"),
                    file_name="change_audit.csv",
                    mime="text/csv",
                )

            st.markdown("#### Before / after changed columns")
            changed = changed_columns_frame(dataframe, outcome.dataframe)
            if changed.empty:
                st.info("No column-level value or dtype changes were detected.")
            else:
                st.dataframe(changed, use_container_width=True)

            before_col, after_col = st.columns(2)
            with before_col:
                st.markdown("#### Before")
                st.dataframe(dataframe.head(30), use_container_width=True)
            with after_col:
                st.markdown("#### After")
                st.dataframe(outcome.dataframe.head(30), use_container_width=True)

            st.download_button(
                "Download processed CSV",
                data=_dataframe_bytes(outcome.dataframe),
                file_name="processed_data.csv",
                mime="text/csv",
            )
            st.caption(
                "Processed CSV values are preserved exactly and may be interpreted "
                "as formulas by spreadsheet software. Audit CSV exports neutralize "
                "formula-like text."
            )
            report_markdown = build_cleaning_report(
                before=dataframe,
                after=outcome.dataframe,
                plan=proposal.plan,
                validation=outcome.validation,
                execution_records=outcome.execution_records,
                policy=policy,
                planner_name=planner_name,
                source_label=source_label,
                rolled_back=outcome.rolled_back,
                provenance=audit_provenance,
                contract=contract,
                pre_contract_validation=proposal.contract_validation,
                post_contract_validation=outcome.contract_validation,
                contract_caused_rollback=outcome.contract_caused_rollback,
            )
            st.download_button(
                "Download cleaning report Markdown",
                data=report_markdown.encode("utf-8"),
                file_name=build_report_filename(source_label),
                mime="text/markdown",
            )

with experiment_tab:
    _show_experiment_summary()
