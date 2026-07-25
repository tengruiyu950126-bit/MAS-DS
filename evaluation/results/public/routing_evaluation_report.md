# Multi-Expert routing evaluation

Schema version: 2.0
Scenarios: 100
Seeds: [0, 1, 2, 3, 4]

## Routing summary

- Macro F1: 0.7117
- Micro F1: 0.4375
- Protected-column exclusion accuracy: 1.0000
- Clean no-op accuracy: 0.0000

## Critic and arbiter fixtures

- total_proposals: 6
- critic_approval_rate: 0.16666666666666666
- critic_rejection_rate: 0.8333333333333334
- correct_rejection_rate: 1.0
- false_rejection_rate: 0.0
- conflict_count: 1
- conflict_resolution_rate: 1.0
- duplicate_proposal_removal_rate: 1.0
- final_plan_validity_rate: 1.0
- protected_column_mutation_rate: 0.0
- unknown_column_proposal_rate: 0.0
- deterministic_ordering_consistency: 1.0
- empty_plan_rate_on_dirty_datasets: 0.0
- correct_no_op_rate_on_clean_datasets: 0.2

Ground truth is declared by the scenario definitions, independently of RouterAgent output. No dataframe values are stored.