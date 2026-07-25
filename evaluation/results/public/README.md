# Reviewed public evaluation results

This directory contains the reviewed aggregate-only subset of the MAS-DS
multi-expert evaluation generated on 2026-07-25.

- Evaluation schema: **2.0**
- Seeds: **0, 1, 2, 3, 4**
- Routing scenarios/runs: **100**
- Ablation runs: **700**
- Configurations: **7**

## Public artifacts

- [Routing metrics by specialist](routing_metrics_by_specialist.csv)
- [Routing metrics summary](routing_metrics_summary.json)
- [Routing evaluation report](routing_evaluation_report.md)
- [Ablation metrics by configuration](ablation_metrics_by_configuration.csv)
- [Ablation pairwise comparison](ablation_pairwise_comparison.csv)
- [Ablation summary](ablation_summary.json)
- [Ablation report](ablation_report.md)

Only aggregate artifacts are public. Scenario rows, routing predictions, and
individual ablation-run rows remain ignored local review evidence under
`outputs/`; they are not part of this public subset.

## Interpretation

`expected_operation_coverage` measures whether a plan contains independently
expected operation labels. It is **not repaired-cell correctness**. Use the
ground-truth corruption evaluation for value-level recovery and preservation
claims.

The routed planner's routing false/unnecessary-activation rate is approximately
`0.716216`, and clean-dataset no-op accuracy is `0.0`. These weak results mean
the current router over-activates specialists and does not reliably stay idle on
clean controls. They identify a routing-selectivity limitation; they must not be
hidden behind operation-coverage results.

Timestamps, durations, and latency-derived statistics are environment-specific
and nondeterministic. The deterministic decisions and non-timing metrics were
reproduced in a second five-seed run before promotion.

These controlled synthetic results do not prove general repair correctness,
privacy, model quality, production readiness, or suitability for public or
multi-user deployment. The implementation is a deterministic routed planner,
not a genuine Multi-Agent System. No production-readiness or genuine-MAS claim
is made.

Artifacts contain aggregate labels, counts, and metrics only. They contain no
scenario-level rows, routing predictions, source dataframe cells, uploaded
filenames, local paths, process IDs, or runtime logs. Human privacy, copyright,
and provenance review remains necessary for future result updates.

See [the evaluation methodology](../../../docs/MULTI_EXPERT_EVALUATION.md) for
definitions and threats to validity.
