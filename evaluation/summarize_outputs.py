"""Aggregate MAS-DS experiment outputs into one readable summary report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "repair_success_rate",
    "data_preservation_rate",
    "invalid_plan",
    "rolled_back",
    "latency_seconds",
    "plan_steps",
]

CORE_SCORE_COLUMNS = [
    "detection_f1",
    "repair_success_rate",
    "data_preservation_rate",
]


@dataclass(frozen=True)
class SummaryArtifacts:
    overview_csv: Path
    report_md: Path
    scanned_files: list[Path]


def _experiment_name(path: Path) -> str:
    stem = path.stem
    return stem[:-8] if stem.endswith("_summary") else stem


def discover_summary_files(output_dir: Path) -> list[Path]:
    """Return summary CSV files in a stable order."""
    return sorted(
        path
        for path in output_dir.glob("*_summary.csv")
        if path.is_file()
    )


def load_summary_frames(paths: list[Path]) -> pd.DataFrame:
    """Load summary files and annotate each row with its source experiment."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.insert(0, "source_file", path.name)
        frame.insert(0, "experiment", _experiment_name(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return None
    return float(values.mean())


def _format_optional(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _method_label(row: pd.Series) -> str:
    for column in ("method", "mode", "scenario"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return "overall"


def build_experiment_overview(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Build one comparable row per experiment/method-like label."""
    if summary_frame.empty:
        return pd.DataFrame(
            columns=[
                "experiment",
                "label",
                "runs_or_rows",
                "balanced_score",
                *METRIC_COLUMNS,
                "pass_rate",
            ]
        )

    working = summary_frame.copy()
    working["label"] = working.apply(_method_label, axis=1)

    rows = []
    for (experiment, label), group in working.groupby(
        ["experiment", "label"],
        dropna=False,
        sort=True,
    ):
        core_values = [
            _numeric_mean(group, column)
            for column in CORE_SCORE_COLUMNS
            if _numeric_mean(group, column) is not None
        ]
        balanced_score = (
            sum(core_values) / len(core_values)
            if core_values
            else None
        )
        pass_rate = _numeric_mean(group, "passed")
        row = {
            "experiment": experiment,
            "label": label,
            "runs_or_rows": len(group),
            "balanced_score": _format_optional(balanced_score),
            "pass_rate": _format_optional(pass_rate),
        }
        for column in METRIC_COLUMNS:
            row[column] = _format_optional(_numeric_mean(group, column))
        rows.append(row)

    ordered_columns = [
        "experiment",
        "label",
        "runs_or_rows",
        "balanced_score",
        "detection_f1",
        "repair_success_rate",
        "data_preservation_rate",
        "pass_rate",
        "invalid_plan",
        "rolled_back",
        "latency_seconds",
        "plan_steps",
        "detection_precision",
        "detection_recall",
    ]
    return pd.DataFrame(rows)[ordered_columns]


def build_corruption_breakdown(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Build method/corruption detail when corruption columns are available."""
    if summary_frame.empty or "corruption" not in summary_frame.columns:
        return pd.DataFrame()

    working = summary_frame.copy()
    working = working[working["corruption"].notna()]
    if working.empty:
        return pd.DataFrame()
    working["label"] = working.apply(_method_label, axis=1)
    rows = []
    for (experiment, label, corruption), group in working.groupby(
        ["experiment", "label", "corruption"],
        dropna=False,
        sort=True,
    ):
        rows.append(
            {
                "experiment": experiment,
                "label": label,
                "corruption": corruption,
                "detection_f1": _format_optional(
                    _numeric_mean(group, "detection_f1")
                ),
                "repair_success_rate": _format_optional(
                    _numeric_mean(group, "repair_success_rate")
                ),
                "data_preservation_rate": _format_optional(
                    _numeric_mean(group, "data_preservation_rate")
                ),
                "rolled_back": _format_optional(_numeric_mean(group, "rolled_back")),
                "plan_steps": _format_optional(_numeric_mean(group, "plan_steps")),
            }
        )
    return pd.DataFrame(rows)


def build_policy_summary(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Return policy scenario pass/fail rows when policy results exist."""
    if summary_frame.empty or "passed" not in summary_frame.columns:
        return pd.DataFrame()
    working = summary_frame[summary_frame["passed"].notna()].copy()
    if "scenario" in working.columns:
        working = working[working["scenario"].notna()]
    if working.empty:
        return pd.DataFrame()
    columns = [
        column
        for column in ["experiment", "scenario", "passed", "plan_steps"]
        if column in working.columns
    ]
    return working[columns].copy()


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No data available._"
    preview = frame.head(max_rows).copy()
    preview = preview.fillna("")
    columns = [str(column) for column in preview.columns]
    rows = [
        [str(value) for value in record]
        for record in preview.to_numpy().tolist()
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _top_takeaways(overview: pd.DataFrame, policy: pd.DataFrame) -> list[str]:
    takeaways: list[str] = []
    if not overview.empty and "balanced_score" in overview.columns:
        ranked = overview.dropna(subset=["balanced_score"]).sort_values(
            "balanced_score",
            ascending=False,
        )
        if not ranked.empty:
            best = ranked.iloc[0]
            takeaways.append(
                "Best balanced result in the scanned summaries: "
                f"`{best['label']}` from `{best['experiment']}` "
                f"(balanced_score={best['balanced_score']})."
            )
    if not overview.empty and "data_preservation_rate" in overview.columns:
        preservation = pd.to_numeric(
            overview["data_preservation_rate"],
            errors="coerce",
        )
        if not preservation.dropna().empty:
            takeaways.append(
                "Average preservation across comparable summaries: "
                f"{preservation.mean():.3f}."
            )
    if not policy.empty and "passed" in policy.columns:
        pass_rate = pd.to_numeric(policy["passed"], errors="coerce").mean()
        takeaways.append(f"Policy scenario pass rate: {pass_rate:.3f}.")
    if not takeaways:
        takeaways.append("No comparable metrics were found in the scanned outputs.")
    return takeaways


def build_markdown_report(
    summary_frame: pd.DataFrame,
    overview: pd.DataFrame,
    corruption_breakdown: pd.DataFrame,
    policy_summary: pd.DataFrame,
    scanned_files: list[Path],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# MAS-DS Experiment Summary",
        "",
        f"Generated at: {generated_at}",
        "",
        "This report aggregates existing CSV summaries in `outputs/`. It does not "
        "call paid APIs and does not rerun expensive local-LLM experiments.",
        "",
        "## Files scanned",
        "",
    ]
    if scanned_files:
        lines.extend(f"- `{path.name}`" for path in scanned_files)
    else:
        lines.append("- No `*_summary.csv` files found.")

    lines.extend(
        [
            "",
            "## Key takeaways",
            "",
        ]
    )
    lines.extend(f"- {takeaway}" for takeaway in _top_takeaways(overview, policy_summary))

    lines.extend(
        [
            "",
            "## Experiment overview",
            "",
            "`balanced_score` is the average of detection F1, repair success rate, "
            "and data preservation rate when those fields are available. Higher is "
            "better, but latency, rollback behavior, and policy compliance should "
            "still be inspected separately.",
            "",
            _markdown_table(overview),
            "",
            "## Corruption-level breakdown",
            "",
            _markdown_table(corruption_breakdown, max_rows=40),
            "",
            "## Policy scenario summary",
            "",
            _markdown_table(policy_summary, max_rows=40),
            "",
            "## How to interpret the metrics",
            "",
            "- `detection_f1`: whether the planner identified the intended problem.",
            "- `repair_success_rate`: whether known corrupted cells/rows were repaired.",
            "- `data_preservation_rate`: whether unaffected clean cells were preserved.",
            "- `invalid_plan`: rate of planner failures or rejected plans.",
            "- `rolled_back`: rate of validation-triggered rollback.",
            "- `latency_seconds`: average runtime for the summarized runs.",
            "",
            "For this project, a strong method is not simply the one that changes the "
            "most cells. A strong method repairs dirty values while preserving clean "
            "data and respecting policy constraints.",
        ]
    )
    if summary_frame.empty:
        lines.extend(
            [
                "",
                "## Note",
                "",
                "No summary rows were available. Run one of the evaluation scripts "
                "first, for example `python -m evaluation.run_experiment --methods rule`.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_summary_artifacts(
    *,
    output_dir: Path,
    overview_name: str = "experiment_summary_overview.csv",
    report_name: str = "experiment_summary_report.md",
) -> SummaryArtifacts:
    paths = discover_summary_files(output_dir)
    summary_frame = load_summary_frames(paths)
    overview = build_experiment_overview(summary_frame)
    corruption_breakdown = build_corruption_breakdown(summary_frame)
    policy_summary = build_policy_summary(summary_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / overview_name
    report_path = output_dir / report_name

    overview.to_csv(overview_path, index=False)
    report_path.write_text(
        build_markdown_report(
            summary_frame,
            overview,
            corruption_breakdown,
            policy_summary,
            paths,
        ),
        encoding="utf-8",
    )
    return SummaryArtifacts(
        overview_csv=overview_path,
        report_md=report_path,
        scanned_files=paths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--overview-name", default="experiment_summary_overview.csv")
    parser.add_argument("--report-name", default="experiment_summary_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = write_summary_artifacts(
        output_dir=Path(args.output_dir),
        overview_name=args.overview_name,
        report_name=args.report_name,
    )
    print(f"Scanned {len(artifacts.scanned_files)} summary files.")
    print(f"Wrote overview CSV to {artifacts.overview_csv}")
    print(f"Wrote Markdown report to {artifacts.report_md}")


if __name__ == "__main__":
    main()
