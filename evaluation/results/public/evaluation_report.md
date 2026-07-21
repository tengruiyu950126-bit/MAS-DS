# Reviewed Multi-Expert evaluation summary

This value-free report summarizes the deterministic five-seed evaluation
(schema 1.0): 110 scenarios, 110 routing runs, and 770 runs across seven
configurations. Ground truth was declared independently from Router output.

## Routing quality

- Macro F1: 0.6818
- Micro F1: 0.3802
- False-routing rate: 0.7629
- Missed-routing rate: 0.0417
- Protected-column exclusion accuracy: 1.0000
- Clean-dataset routing no-op accuracy: 0.0000
- NumericExpert precision: 0.1282
- CategoricalExpert precision: 0.0732

The Router is not optimal. It activates numeric and categorical analysis too
broadly, preserving high recall but causing low precision and unnecessary expert
activation on clean or non-target columns.

## Baseline and ablations

| Configuration | Detection F1 | Repair success | Unsafe proposals | Mean latency (s) |
|---|---:|---:|---:|---:|
| rule baseline | 0.5165 | 0.8636 | 0.0000 | 0.058539 |
| full Multi-Expert | 0.5714 | 1.0000 | 0.0000 | 0.069837 |
| no Critic | 0.5714 | 1.0000 | 0.0000 | 0.060034 |
| no Arbiter | 0.5714 | 1.0000 | 0.0000 | 0.069257 |
| router/specialists only | 0.5714 | 1.0000 | 0.0000 | 0.060482 |
| oracle Router | 0.9806 | 1.0000 | 0.0000 | 0.037212 |
| faulty Router | 0.4955 | 0.7455 | 0.0091 | 0.063972 |

Full Multi-Expert improved labeled detection and repair coverage over the rule
baseline in this suite, but was slower. Oracle routing indicates that Router
selectivity is the largest measured opportunity. Removing Critic or Arbiter did
not change aggregate natural-scenario repair because bounded specialists did not
emit unsafe conflicts there. Controlled fixtures showed correct unsafe rejection,
conflict resolution, duplicate removal, and deterministic ordering.

## Limitations

Synthetic labels simplify real ambiguity; bundled public datasets are small;
operation-label coverage is a repair proxy; latency is machine-dependent; and
normal confidence intervals are descriptive. No statistical-significance or
general production-performance claim is made. No dataframe values are included.
