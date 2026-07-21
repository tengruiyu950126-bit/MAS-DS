import pandas as pd

from evaluation.summarize_outputs import (
    build_corruption_breakdown,
    build_experiment_overview,
    build_markdown_report,
    build_policy_summary,
    discover_summary_files,
    load_summary_frames,
    write_summary_artifacts,
)


def test_discover_summary_files_only_returns_summary_csvs(tmp_path) -> None:
    (tmp_path / "rule_summary.csv").write_text("method\nrule\n", encoding="utf-8")
    (tmp_path / "raw_results.csv").write_text("method\nrule\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore", encoding="utf-8")

    files = discover_summary_files(tmp_path)

    assert [path.name for path in files] == ["rule_summary.csv"]


def test_overview_computes_balanced_score_and_method_label(tmp_path) -> None:
    path = tmp_path / "evaluation_results_summary.csv"
    pd.DataFrame(
        [
            {
                "method": "rule",
                "corruption": "missing_value",
                "detection_f1": 1.0,
                "repair_success_rate": 1.0,
                "data_preservation_rate": 1.0,
                "latency_seconds": 0.1,
                "plan_steps": 2,
            },
            {
                "method": "rule",
                "corruption": "datetime_type",
                "detection_f1": 0.0,
                "repair_success_rate": 0.0,
                "data_preservation_rate": 1.0,
                "latency_seconds": 0.2,
                "plan_steps": 0,
            },
        ]
    ).to_csv(path, index=False)

    summary = load_summary_frames([path])
    overview = build_experiment_overview(summary)

    assert len(overview) == 1
    row = overview.iloc[0]
    assert row["experiment"] == "evaluation_results"
    assert row["label"] == "rule"
    assert row["runs_or_rows"] == 2
    assert row["balanced_score"] == 0.6667
    assert row["latency_seconds"] == 0.15


def test_policy_summary_preserves_scenario_pass_status(tmp_path) -> None:
    path = tmp_path / "policy_experiment_results_summary.csv"
    pd.DataFrame(
        [
            {"scenario": "default", "passed": True, "plan_steps": 5},
            {"scenario": "strict", "passed": False, "plan_steps": 2},
        ]
    ).to_csv(path, index=False)

    summary = load_summary_frames([path])
    policy = build_policy_summary(summary)
    overview = build_experiment_overview(summary)

    assert list(policy["scenario"]) == ["default", "strict"]
    assert overview.loc[overview["label"] == "default", "pass_rate"].iloc[0] == 1.0
    assert overview.loc[overview["label"] == "strict", "pass_rate"].iloc[0] == 0.0


def test_policy_and_corruption_sections_filter_mixed_summary_rows(tmp_path) -> None:
    metric_path = tmp_path / "evaluation_results_summary.csv"
    pd.DataFrame(
        [
            {
                "method": "rule",
                "corruption": "missing_value",
                "detection_f1": 1.0,
                "repair_success_rate": 1.0,
                "data_preservation_rate": 1.0,
            }
        ]
    ).to_csv(metric_path, index=False)
    policy_path = tmp_path / "policy_experiment_results_summary.csv"
    pd.DataFrame(
        [{"scenario": "default", "passed": True, "plan_steps": 5}]
    ).to_csv(policy_path, index=False)

    summary = load_summary_frames([metric_path, policy_path])
    policy = build_policy_summary(summary)
    corruption = build_corruption_breakdown(summary)

    assert list(policy["scenario"]) == ["default"]
    assert list(corruption["corruption"]) == ["missing_value"]


def test_write_summary_artifacts_creates_csv_and_report(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "method": "rule",
                "corruption": "missing_value",
                "detection_f1": 1.0,
                "repair_success_rate": 1.0,
                "data_preservation_rate": 1.0,
            }
        ]
    ).to_csv(tmp_path / "evaluation_results_summary.csv", index=False)

    artifacts = write_summary_artifacts(output_dir=tmp_path)

    assert artifacts.overview_csv.exists()
    assert artifacts.report_md.exists()
    assert artifacts.scanned_files[0].name == "evaluation_results_summary.csv"
    report = artifacts.report_md.read_text(encoding="utf-8")
    assert "MAS-DS Experiment Summary" in report
    assert "evaluation_results_summary.csv" in report


def test_markdown_report_handles_empty_inputs() -> None:
    report = build_markdown_report(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        [],
    )

    assert "No `*_summary.csv` files found" in report
    assert "No summary rows were available" in report
