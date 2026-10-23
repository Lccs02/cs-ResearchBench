# PACSR audit contract (`pacsr-audit/1.0`)

## Status and scope

Real-cache/paper reproduction: **Unverified**, blocked on a matching cache and
original results. The synthetic reference is an independent NumPy formula fixture,
not a PACSR paper result. Original private evaluators were inspected, not executed:
some entry points load weights and perform inference. This CLI does neither.

`pacsr-inspect FILE --format npz|csv|json --output NEW.json` reports keys/selectors,
shape, dtype and SHA-256. It cannot infer axes, units, mask semantics, model identity,
normalization or test split. All these remain Unverified until evidence is inspected.
No source values or weights are copied into the public repository.

## Configuration

Run `python examples/generate_pacsr_synthetic.py --output examples/generated/pacsr`
for a complete executable configuration. **Its axes, channels, keys and shapes are
synthetic fixtures, not a production PACSR mapping.** Paths resolve relative to the
configuration. Unknown fields/versions are rejected. Required fields:

| Field | Meaning |
|---|---|
| `schema_version` | Literal `pacsr-audit/1.0` |
| `runs` | Nonempty list, unique `run_id` |
| Run identity | `run_id,dataset,model,variant,seed,split,evaluation_id,data_kind,model_config` |
| `input.file` | `path,sha256,format,mapping`; map `target,prediction,mask,test_ids` explicitly |
| `input.axes` | A permutation of `sample,horizon,node,channel`; no inferred reshape |
| `input.expected_shape` | Exact shape in stored axis order, required |
| `test_scope.expected_ids` | Independent full-test IDs source with `path,sha256,format,mapping: {test_ids: SELECTOR}` |
| `test_scope.id_axes` | `[sample]` or `[sample,horizon]`; rows must identify distinct windows |
| `test_scope.description` | Actual test boundaries, stride, overlap, ordering and sampling evidence |
| `evaluation.total_channel/component_channels` | Distinct explicit channel indices; no default |
| `evaluation.horizons` | Unique 1-based single-step horizons; `all` additionally pools every horizon |
| `evaluation.metrics` | Map `all` and each selected `hN` to a nonempty list of required metric names |
| `evaluation.fine_reduction` | `macro` or `pooled`; cannot silently interchange |
| `evaluation.normalization` | `identity` + description, or the explicit affine contract below |
| `evaluation.aggregation` | `kind: sum|mean|weighted_sum`, explicit `dtype: float32|float64`; weighted_sum also requires ordered weights |
| `evaluation.ncv_epsilon` | Explicit positive denominator epsilon; never optimized to match results |
| `evaluation.units` | Evaluated measurement unit; ncv is dimensionless |
| `evaluation.missing` | Literal `finite_target_and_mask_reject_nonfinite_prediction` |
| `evidence` | Local files with `role,path,sha256`; must include evaluator, model_config, split, normalization |
| `golden` | Original JSON report, selectors, identities, source bindings and frozen tolerances |

CSV uses literal column names/lists, JSON literal keys/list traversal paths, NPZ
exact keys with `allow_pickle=False`. A 4D forecast represented as nested JSON or
NPZ is supported; a flat CSV is not automatically reshaped. Such projects need a
separately reviewed adapter. No timestamp or row sorting, truncation, test-set
filtering, checkpoint selection or post-hoc forecast reconciliation is performed.

Affine normalization requires `kind: affine`, `description`, `dtype`, `axes`, and
`parameters` (file descriptor mapping `scale` and `offset`). Both parameter shapes
must exactly match the declared axes, in that order; scalar parameters use `axes: []`.
The only broadcast is the one explicitly authorized by these named axes. Both target
and prediction are transformed as `x * scale + offset`. Scale must be positive and
finite. Caches already saved in raw units must use identity; do not inverse twice.
Different target/prediction transforms, nonlinear scaling and unsupported aggregation
operators require a reviewed extension, never guessed conversions.

All sources require pre-recorded lowercase SHA-256. Files are checked again before
export commits its new directory. IDs must match the independently supplied full
test manifest in shape, values and order. Matching IDs cannot prove provenance of
that manifest: source authenticity always remains Unverified.

## Metrics and statistical units

Generic `regression_metrics` is unchanged. PACSR first rejects all nonfinite
predictions (including masked positions), then intersects the Boolean/0/1 mask with
finite targets. Each channel reduces **observed scalar elements** uniformly over
samples/nodes and, for `all`, all stored horizons. No sample weights are supported
in this PACSR contract. Missing and masked elements are excluded and `n_valid` /
`n_total` are recorded. Completely unobserved channels produce missing values,
exported as Unverified, not zero.

- `coarse_mae`, `coarse_rmse`: total channel, observation weighted.
- `component_N_mae/rmse`: the explicit channel N, observation weighted.
- `fine_mae_macro`, `fine_rmse_macro`: equal mean of active component metrics;
  specifically, mean of component RMSEs, not the root of a pooled mean square.
- `fine_pooled_mae/rmse`: all selected observed component scalars pooled. Available
  only with `fine_reduction: pooled`; these are different metrics from macro.
- Let `g = abs(total_prediction - A(component_predictions))`, where A is the
  declared operator in the declared dtype. `cv = mean(g)` on joint observed support
  (total and every selected component have finite targets and true masks).
- `max_cv = max(g)` on that same support.
- `ncv = sum(g) / (sum(abs(target_total)) + ncv_epsilon)` on that same support.
- `cv_all_forecasts = mean(g)` on every forecast, regardless of target masks.

Consistency's observation unit is sample/horizon/node, not a component. These
names define ResearchBench's exact configured calculation. They must not be equated
with a paper's similarly named metric without verifying that paper's evaluator.
The inspected PACSR-Simple source uses active-component macro errors and computes
the sum constraint before promoting errors to double precision. NumPy and PyTorch
reductions can still differ in rounding; choose justified tolerances beforehand.
The framework does not copy a PyTorch evaluator or add a torch dependency.

Summaries group dataset, model, variant, split, evaluation ID, exact protocol,
horizon, metric, units and data kind. One seed is one independent unit. Seed means
are equally weighted, sample std uses ddof=1, a singleton's std is Unverified.
Any missing seed metric makes the group's mean/std Unverified; it is not dropped
silently. Repeated seed/protocol pairs and varying model config under the same
variant are rejected. Horizon samples are never counted as independent seeds.

## Golden Reference and tolerances

`golden` requires `path,sha256,format: json,identity,bindings,metrics,atol,rtol`.
`identity` maps at least `dataset` and `seed` to selectors in the **original** report;
optional `model,variant,split` selectors check those as well. Unavailable identity
fields are listed as Unverified in `comparison.csv`. The real acceptance test
requires all five, so an incomplete old report needs additional original identity
evidence and a reviewed adapter, not invented values in an original JSON file.

`bindings` maps `cache,evaluator,test_scope,model_config,normalization,split` to lists
of corresponding SHA-256 values. These investigator-declared associations prevent
accidentally mixing configured sources but are not independent proof that the
original evaluator consumed them. Preserve original logs/dependency versions and
all imported evaluation source files as additional evaluator evidence entries.

`metrics` maps every planned `horizon/metric` to its original JSON selector, e.g.
`all/coarse_mae: [methods, model, coarse_mae]`. Selectors are explicit; this example
is not confirmation for a particular cache. Every planned output must have exactly
one mapping. Extra original report fields may remain unused (e.g. training time).
Choose scope and metric coverage before inspection of numerical differences.

Pass means `abs(recomputed - original) <= atol + rtol * abs(original)`.
Both tolerances must be finite, nonnegative, and declared before this invocation;
the configuration hash is recorded. The software cannot prove when a human first
chose a tolerance. Neither automatic tolerance adjustment nor best-match searching
exists. Missing values are Unverified and prevent an all-pass result. Relative
error for a zero reference is Unverified, while absolute tolerance comparison
still works. Comparison pass is numerical agreement, never scientific verification.

`pacsr-run CONFIG --output NEW_DIRECTORY` exits 0 on all-pass, 3 on mismatch/missing
reference metrics (retaining a complete report), and 2 on invalid config or changed
sources. No script is executed. No network request is made.

## Outputs and acceptance

`metrics.csv` retains per-run seed/horizon/counts/unit/protocol/source IDs.
`summary.csv` retains group identities, seeds and run IDs. `comparison.csv` records
original value, absolute/relative errors, tolerances, selectors and reference hash.
`summary.tex` uses booktabs; all strings are escaped. Figures are grouped by
protocol/units and labeled synthetic/real/unknown plus Unverified. Bars are seed
sample std, not confidence intervals. `audit.json` retains configuration, original
source paths/hashes, protocols, mappings and comparisons; `manifest.json` records
artifact hashes and package versions. Reports can contain private paths/configs:
keep them local. The exporter never copies private source files into reports.

The full real-data criterion is not met by the shipped synthetic fixture. Run the
opt-in `tests/test_pacsr_real.py` only with genuine locally inspected evidence.
Do not commit the environment configuration, caches, original private scripts,
weights, or private exported reports. CI tests only synthetic data.
