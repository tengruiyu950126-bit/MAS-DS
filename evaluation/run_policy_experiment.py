"""Evaluate whether preprocessing policies actually control planner behavior."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from agents.cleaning_agent import RuleBasedCleaningAgent
from models.policy import PreprocessingPolicy


OperationKey = tuple[str, str | None]


@dataclass(frozen=True)
class PolicyScenario:
    name: str
    description: str
    policy: PreprocessingPolicy
    required_operations: set[OperationKey]
    forbidden_operations: set[OperationKey]


@dataclass(frozen=True)
class PolicyExperimentRecord:
    scenario: str
    passed: bool
    required_satisfied: bool
    forbidden_satisfied: bool
    plan_steps: int
    operations: str
    missing_required: str
    forbidden_present: str
    description: str


def make_policy_stress_dataset() -> pd.DataFrame:
    """Create a table that exercises policy-controlled decisions."""
    return pd.DataFrame(
        {
            "customer_id": [
                " C-001 ",
                "C-002",
                "C-003",
                "C-004",
                "C-005",
                "C-006",
                "C-007",
                "C-008",
                "C-009",
            ],
            "city": [
                "Singapore",
                " Singapore ",
                "SINGAPORE",
                "Singaproe",
                "Bangkok",
                "Bangkok",
                "Jakarta",
                "Jakarta",
                None,
            ],
            "amount": [10, 11, 9, 10, 12, 11, 10, 9, 500],
        }
    )


def policy_scenarios() -> list[PolicyScenario]:
    return [
        PolicyScenario(
            name="default",
            description=(
                "Default safe policy protects identifier columns, cleans city "
                "quality issues, fills city missing values, and flags outliers."
            ),
            policy=PreprocessingPolicy(),
            required_operations={
                ("strip_whitespace", "city"),
                ("normalize_case", "city"),
                ("normalize_category_typos", "city"),
                ("fill_mode", "city"),
                ("flag_outliers_iqr", "amount"),
            },
            forbidden_operations={
                ("strip_whitespace", "customer_id"),
                ("fill_mode", "customer_id"),
            },
        ),
        PolicyScenario(
            name="strict_city_protected",
            description=(
                "User protects city, so no automatic city cleanup or fill should "
                "be proposed."
            ),
            policy=PreprocessingPolicy(protected_columns=["city"]),
            required_operations={("leave_unchanged", "city")},
            forbidden_operations={
                ("strip_whitespace", "city"),
                ("normalize_case", "city"),
                ("normalize_category_typos", "city"),
                ("fill_mode", "city"),
            },
        ),
        PolicyScenario(
            name="no_outlier_flagging",
            description="User disables advisory outlier flagging.",
            policy=PreprocessingPolicy(outlier_action="ignore"),
            required_operations={
                ("strip_whitespace", "city"),
                ("normalize_case", "city"),
                ("normalize_category_typos", "city"),
                ("fill_mode", "city"),
            },
            forbidden_operations={("flag_outliers_iqr", "amount")},
        ),
        PolicyScenario(
            name="relaxed_identifier_cleaning",
            description=(
                "User disables automatic ID protection, so customer_id whitespace "
                "cleanup becomes allowed."
            ),
            policy=PreprocessingPolicy(protect_identifier_columns=False),
            required_operations={("strip_whitespace", "customer_id")},
            forbidden_operations=set(),
        ),
        PolicyScenario(
            name="disable_text_normalization",
            description=(
                "User disables text normalization operations while still allowing "
                "missing-value repair."
            ),
            policy=PreprocessingPolicy(
                denied_operations=[
                    "strip_whitespace",
                    "normalize_case",
                    "normalize_category_typos",
                ]
            ),
            required_operations={("fill_mode", "city")},
            forbidden_operations={
                ("strip_whitespace", "city"),
                ("normalize_case", "city"),
                ("normalize_category_typos", "city"),
            },
        ),
    ]


def _format_keys(keys: set[OperationKey]) -> str:
    return ";".join(
        f"{operation}:{column or '*'}"
        for operation, column in sorted(keys)
    )


def evaluate_scenario(
    dataframe: pd.DataFrame,
    scenario: PolicyScenario,
) -> PolicyExperimentRecord:
    plan = RuleBasedCleaningAgent(policy=scenario.policy).propose(dataframe)
    operations = {(step.operation, step.column) for step in plan.steps}
    missing_required = scenario.required_operations - operations
    forbidden_present = scenario.forbidden_operations & operations
    required_satisfied = not missing_required
    forbidden_satisfied = not forbidden_present
    return PolicyExperimentRecord(
        scenario=scenario.name,
        passed=required_satisfied and forbidden_satisfied,
        required_satisfied=required_satisfied,
        forbidden_satisfied=forbidden_satisfied,
        plan_steps=len(plan.steps),
        operations=_format_keys(operations),
        missing_required=_format_keys(missing_required),
        forbidden_present=_format_keys(forbidden_present),
        description=scenario.description,
    )


def write_results(records: list[PolicyExperimentRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/policy_experiment_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = make_policy_stress_dataset()
    records = [
        evaluate_scenario(dataframe, scenario)
        for scenario in policy_scenarios()
    ]
    output_path = Path(args.output)
    write_results(records, output_path)
    frame = pd.DataFrame(asdict(record) for record in records)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    frame[["scenario", "passed", "plan_steps"]].to_csv(summary_path, index=False)
    print(frame[["scenario", "passed", "operations"]].to_string(index=False))
    print(f"\nWrote {len(records)} policy scenarios to {output_path}")
    print(f"Wrote summary to {summary_path}")
    if not frame["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
