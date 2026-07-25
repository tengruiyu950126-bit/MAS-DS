# Multi-Expert evaluation

MAS-DS includes a local, deterministic evaluation of routing quality and of the
Router → specialists → Critic → Arbiter architecture. It uses synthetic labeled
corruptions and small public datasets bundled with scikit-learn; it never
downloads data or calls a model/API.

## Reproduce an evaluation

```powershell
python -m evaluation.multi_expert_evaluation `
  --output-dir outputs/multi_expert_evaluation_YYYYMMDD_HHMMSS `
  --seeds 0 1 2 3 4
```

Use `--quick` for a small test run. A non-empty output directory is rejected;
`--overwrite` is the explicit opt-in for disposable evaluation directories.
Production planner settings are never changed by an ablation.

## Ground truth and scenarios

Ground truth is declared in `evaluation/multi_expert_scenarios.py`, independently
of RouterAgent output. The unit is a `(scenario, column, specialist)` assignment;
`*` represents a dataset-level duplicate-row assignment. Five seeded repetitions
cover duplicates, missing numeric/categorical values, outliers, numeric text,
categories, high-cardinality data, text whitespace/length, datetimes, mixed/all
families, protected identifiers, contract protection, ambiguity, clean controls,
unsupported objects, and bundled Iris/Wine controls. Artifacts contain dimensions,
labels, predictions, and metrics—not dataframe cell values.

## Metrics

Routing reports exact multi-label match, per-specialist precision/recall/F1,
macro and micro averages, false/missed routing, unnecessary activation,
protected exclusion, clean no-op, coverage, and multiple-expert conflict rate.
Zero denominators are defined as zero, except configuration-level precision on a
correct empty prediction, which is one.

Controlled proposal fixtures measure Critic approvals/rejections, correct unsafe
rejection, false rejection, conflict handling, duplicate removal, final validity,
protected/unknown-column leakage, and ordering consistency. Ablation rows add
repair-label coverage, validation/rollback, preservation, latency, proposal/plan
size, and interpretability proxies. Confidence intervals are normal 95% intervals;
no statistical significance is claimed. Peak memory is omitted because no reliable
cross-platform existing measurement mechanism is available.

## Configurations

- `rule_baseline`: existing single rule planner.
- `multi_expert_full`: all four orchestration stages.
- `multi_expert_no_critic`: proposals go directly to the Arbiter.
- `multi_expert_no_arbiter`: Critic-approved proposals are flattened in stable
  specialist order.
- `multi_expert_router_only`: raw proposals are flattened; unsafe plans are not
  executed.
- `multi_expert_oracle_router`: independent scenario labels replace Router output.
- `multi_expert_faulty_router`: a seeded miss and wrong route are injected before
  normal Critic/Arbiter processing.

## Artifacts

`routing_scenarios.csv`, `routing_predictions.csv`,
`routing_metrics_by_specialist.csv`, `routing_metrics_summary.json`, and
`routing_evaluation_report.md` describe routing. `ablation_runs.csv`,
`ablation_metrics_by_configuration.csv`, `ablation_pairwise_comparison.csv`,
`ablation_summary.json`, and `ablation_report.md` describe configurations.
Every current table has schema version 2.0. JSON records the seeds, counts, mode,
and reproducibility result.

The reviewed public subset in `evaluation/results/public/` contains only the
seven aggregate metric, summary, and report artifacts. It covers seeds 0–4,
100 routing scenarios, 700 ablation runs, and seven configurations. The
scenario table, routing predictions, and individual ablation runs remain
ignored review evidence and are not public.

`expected_operation_coverage` is an operation-label proxy, not repaired-cell
correctness. Timings and latency-derived statistics are environment-specific
and nondeterministic. Current routing false/unnecessary activation is about
0.716216 and clean-data no-op accuracy is 0.0, showing that routing selectivity
remains weak even where expected-operation coverage is high.

## Interpretation and threats to validity

Macro F1 should be read beside specialist rows; exact match is deliberately strict.
Routing an expert does not imply a mutation because bounded experts may return no
proposal. Synthetic labels are necessarily simplified, small bundled datasets do
not represent every domain, operation coverage is only a repair proxy, and timing
is machine-dependent. Inspect family-level rows for regressions rather than using
one aggregate score as a product claim. The deterministic routed planner is not a
genuine Multi-Agent System, and these results do not establish production
readiness.
