import pytest

from evaluation.run_ablation import evaluate_once
from evaluation.run_experiment import make_demo_dataset


@pytest.mark.parametrize("mode", ["with_validation_rollback", "without_validation"])
def test_validation_ablation_detects_unsafe_candidate(mode: str) -> None:
    record = evaluate_once(
        make_demo_dataset(),
        mode=mode,
        seed=0,
        fraction=0.10,
    )

    assert not record.validation_valid
    assert "new_missing_values" in record.validation_issues
    assert record.detection_recall == 1.0


def test_validation_rollback_preserves_data_at_cost_of_repair() -> None:
    record = evaluate_once(
        make_demo_dataset(),
        mode="with_validation_rollback",
        seed=0,
        fraction=0.10,
    )

    assert record.rolled_back
    assert record.repair_success_rate == 0.0
    assert record.data_preservation_rate == 1.0
    assert record.missing_after_committed == record.missing_before


def test_without_validation_commits_unsafe_candidate() -> None:
    record = evaluate_once(
        make_demo_dataset(),
        mode="without_validation",
        seed=0,
        fraction=0.10,
    )

    assert not record.rolled_back
    assert record.repair_success_rate == 0.0
    assert record.data_preservation_rate < 1.0
    assert record.data_preservation_rate < 1.0
    assert record.missing_after_committed > record.missing_before
