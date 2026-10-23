# Phase 2 interfaces and acceptance boundary

The PACSR audit uses a separate `pacsr-audit/1.0` contract; core experiment
schema 1.0 and its MAE/RMSE behavior remain unchanged.

| Layer | Interface | Contract |
|---|---|---|
| adapters | `inspect_cache(path, format)` | Read-only inventory, shapes, dtypes, SHA-256; semantics Unverified |
| adapters | `PACSRAdapter.evaluate(run, base_dir)` | Explicit axes, channels, normalization, masks, full test identifiers → metric rows + sources |
| metrics | `pacsr_metrics(...)` | Existing regression function per channel; explicit fine macro/pooling; configurable consistency operator |
| analysis | `summarize_pacsr`, `compare_golden` | Equal training-seed mean/sample std; strict identity and predeclared tolerance comparison |
| export | `export_pacsr` | Atomic new directory, CSV/LaTeX/SVG/PDF, configuration and source/artifact hashes |
| CLI | `pacsr-inspect`, `pacsr-run` | No training, checkpoint loading, source execution or AI |

Inspection of privately supplied evaluator source found a PACSR-Simple evaluator
that saves forecasts, targets, masks and timestamps, calculates coarse errors and
active-component macro errors, and computes consistency in prediction dtype.
Other evaluator versions delegate to unavailable modules. This does **not** prove
which evaluator produced any paper result. No private source is included here.
The only available NPZ candidate has `pred`, `names`, `weights`; it lacks matching
PACSR targets/masks and cannot establish a real PACSR reproduction. Real cache,
matching original metrics, complete evaluation dependencies, model config and
test-range evidence remain **Unverified**. No inferred production mapping is shipped.

Numerical agreement and scientific provenance are separate. Synthetic integration
fixtures verify implementation only, never the real-cache acceptance criterion.
A draft PR must remain blocked on at least one real dataset/model/seed audit.
