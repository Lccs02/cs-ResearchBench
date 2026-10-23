"""Explicit PACSR reductions; generic regression semantics are unchanged."""
import numpy as np
from researchbench.metrics.regression import regression_metrics


def aggregate_components(values, operator):
    """Reduce final component axis in an explicitly declared floating dtype."""
    if set(operator) - {"kind", "dtype", "weights"}:
        raise ValueError("Unknown aggregation option")
    if operator.get("dtype") not in {"float32", "float64"}:
        raise ValueError("aggregation.dtype must explicitly be float32 or float64")
    x = np.asarray(values, dtype=operator["dtype"])
    kind = operator["kind"]
    if kind in {"sum", "mean"}:
        if "weights" in operator:
            raise ValueError("weights only belong to weighted_sum")
        return x.sum(-1) if kind == "sum" else x.mean(-1)
    if kind == "weighted_sum":
        w = np.asarray(operator["weights"], dtype=x.dtype)
        if w.shape != (x.shape[-1],) or not np.isfinite(w).all():
            raise ValueError("Aggregation weights must match selected components and be finite")
        return (x * w).sum(-1)
    raise ValueError("Unsupported aggregation operator; implement and test the dataset definition")


def pacsr_metrics(target, prediction, mask, *, total, components, fine_reduction,
                  aggregation, ncv_epsilon):
    """Inputs have canonical [sample,horizon,node,channel] axes.

    Reject every nonfinite prediction, omit nonfinite targets, and intersect
    targets with the explicit Boolean observation mask. Macro excludes channels
    with zero observations; never substitute zero errors. Consistency uses joint
    observation support; all-forecasts consistency is separately named.
    """
    t, p, m = np.asarray(target), np.asarray(prediction), np.asarray(mask)
    if t.shape != p.shape or m.shape != t.shape or t.ndim != 4 or not t.size:
        raise ValueError("PACSR target, prediction and mask must have the same nonempty 4D shape")
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions")
    if not np.isin(m, [0, 1]).all():
        raise ValueError("mask must contain only Boolean/0/1 values")
    if fine_reduction not in {"macro", "pooled"}:
        raise ValueError("fine_reduction must be macro or pooled")
    if not isinstance(ncv_epsilon, (int, float)) or not np.isfinite(ncv_epsilon) or ncv_epsilon <= 0:
        raise ValueError("ncv_epsilon must be finite and positive")
    indices = [total, *components]
    if not components or any(type(i) is not int or not 0 <= i < t.shape[-1] for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Explicit distinct in-range total and component indices required")
    valid = m.astype(bool) & np.isfinite(t)
    out = {}

    def errors(tt, pp, vv, prefix):
        count = int(vv.sum())
        values = regression_metrics(tt[vv], pp[vv]) if count else None
        for name in ("MAE", "RMSE"):
            out[f"{prefix}_{name.lower()}"] = {
                "value": values[name]["value"] if values else None,
                "n_valid": count, "n_total": int(tt.size),
                "statistical_unit": "observed scalar", "weighting": "uniform observations"}

    errors(t[..., total], p[..., total], valid[..., total], "coarse")
    for c in components:
        errors(t[..., c], p[..., c], valid[..., c], f"component_{c}")
    if fine_reduction == "pooled":
        errors(t[..., components], p[..., components], valid[..., components], "fine_pooled")
    else:
        for name in ("mae", "rmse"):
            members = [out[f"component_{c}_{name}"] for c in components]
            values = [x["value"] for x in members if x["value"] is not None]
            out[f"fine_{name}_macro"] = {
                "value": float(np.mean(values)) if values else None,
                "n_valid": sum(x["n_valid"] for x in members),
                "n_total": sum(x["n_total"] for x in members),
                "active_components": len(values), "statistical_unit": "active component",
                "weighting": "equal active components after observation-weighted channel error"}
    aggregate = aggregate_components(p[..., components], aggregation)
    total_prediction = p[..., total].astype(aggregation["dtype"])
    with np.errstate(over="raise", invalid="raise"):
        try:
            gap = np.abs(total_prediction - aggregate).astype(np.float64)
        except FloatingPointError as exc:
            raise ValueError("Nonfinite aggregation gap") from exc
    if not np.isfinite(gap).all():
        raise ValueError("Nonfinite aggregation result")
    joint = valid[..., total] & valid[..., components].all(-1)
    count = int(joint.sum())
    selected = gap[joint]
    for name, value in {
        "cv": float(selected.mean()) if count else None,
        "ncv": float(selected.sum() / (np.abs(t[..., total][joint].astype(float)).sum() + ncv_epsilon)) if count else None,
        "max_cv": float(selected.max()) if count else None,
        "cv_all_forecasts": float(gap.mean()),
    }.items():
        out[name] = {"value": value, "n_valid": int(gap.size) if name == "cv_all_forecasts" else count,
                     "n_total": int(gap.size), "statistical_unit": "forecast sample/horizon/node",
                     "weighting": "uniform forecasts" if name != "ncv" else "sum gap / (sum absolute observed total + epsilon)"}
    return out
