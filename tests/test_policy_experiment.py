from evaluation.run_policy_experiment import (
    evaluate_scenario,
    make_policy_stress_dataset,
    policy_scenarios,
)


def test_policy_experiment_scenarios_pass() -> None:
    dataframe = make_policy_stress_dataset()

    records = [
        evaluate_scenario(dataframe, scenario)
        for scenario in policy_scenarios()
    ]

    assert records
    assert all(record.passed for record in records)


def test_policy_experiment_demonstrates_policy_changes_operations() -> None:
    dataframe = make_policy_stress_dataset()
    records = {
        scenario.name: evaluate_scenario(dataframe, scenario)
        for scenario in policy_scenarios()
    }

    assert "flag_outliers_iqr:amount" in records["default"].operations
    assert "flag_outliers_iqr:amount" not in records["no_outlier_flagging"].operations
    assert "strip_whitespace:city" in records["default"].operations
    assert "strip_whitespace:city" not in records["strict_city_protected"].operations
    assert (
        "strip_whitespace:customer_id"
        in records["relaxed_identifier_cleaning"].operations
    )
