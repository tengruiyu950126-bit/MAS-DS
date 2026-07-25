# Deterministic routed-planner baseline and ablation report

Configurations remove only evaluation-time components; production defaults are unchanged.

## Configurations

- `rule_baseline`: existing deterministic rule planner.
- `multi_expert_full`: deterministic Router, rule specialists, Critic, and Arbiter.
- `multi_expert_no_critic`: omits proposal review.
- `multi_expert_no_arbiter`: deterministically flattens Critic-approved proposals.
- `multi_expert_router_only`: flattens specialist proposals without review or arbitration.
- `multi_expert_oracle_router`: replaces routing with independent labels.
- `multi_expert_faulty_router`: injects deterministic misses and wrong routes, then retains Critic and Arbiter.

## Mean outcomes

| configuration | detection F1 | expected-operation coverage | unsafe proposals | validation pass | latency (s) |
|---|---:|---:|---:|---:|---:|
| rule_baseline | 0.5103 | 0.8500 | 0.0000 | 1.0000 | 0.043347 |
| multi_expert_full | 0.5707 | 1.0000 | 0.0000 | 1.0000 | 0.050919 |
| multi_expert_no_critic | 0.5707 | 1.0000 | 0.0000 | 1.0000 | 0.044017 |
| multi_expert_no_arbiter | 0.5707 | 1.0000 | 0.0000 | 1.0000 | 0.050569 |
| multi_expert_router_only | 0.5707 | 1.0000 | 0.0000 | 1.0000 | 0.043464 |
| multi_expert_oracle_router | 0.9787 | 1.0000 | 0.0000 | 1.0000 | 0.028051 |
| multi_expert_faulty_router | 0.5017 | 0.7400 | 0.0100 | 1.0000 | 0.047204 |

## Interpretation

On these labeled scenarios, the deterministic routed planner's expected-operation coverage was 1.0000 versus 0.8500 for the rule baseline. This does not measure repaired cell values and is not a claim of superiority.
Oracle routing raised mean detection F1 from 0.5707 to 0.9787, identifying routing selectivity as the largest measured planning opportunity.
Removing Critic or Arbiter did not change aggregate expected-operation coverage on naturally generated bounded proposals. Their safety value appears only in the controlled fixture.
The current implementation is a deterministic routed planner, not a genuine multi-agent system.
Families where routed-planner detection F1 was below the rule baseline: none in this suite. It was slower and over-activated analysis rules on clean/non-target columns.

## Limitations

This suite measures routing and operation labels, not repaired-cell correctness. Use the ground-truth corruption evaluation for value recovery. Bundled datasets are small; latency is machine-dependent; no significance test is claimed.