"""Element-weighted metrics with explicit finite-value and alignment contracts."""
import numpy as np


def regression_metrics(y_true, y_pred, *, weights=None, missing="error") -> dict:
    """Reduce all selected elements; never broadcast or infer axes.

    With valid positive weights w: MAE=sum(w*abs(e))/sum(w),
    RMSE=sqrt(sum(w*e**2)/sum(w)). NaN/Inf are missing. Invalid
    weights always fail, even at missing targets. Zero weights are excluded.
    """
    if missing not in {"error", "omit"}:
        raise ValueError("missing must be error or omit")
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Prediction/target shapes must match exactly and be nonempty")
    w = np.ones_like(true) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != true.shape:
        raise ValueError("Weight shape must match targets exactly; broadcasting is disabled")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    finite = np.isfinite(true) & np.isfinite(pred)
    if missing == "error" and not finite.all():
        raise ValueError("Nonfinite prediction/target encountered; explicitly choose missing=omit")
    valid = finite & (w > 0)
    if not valid.any():
        raise ValueError("No valid positive-weight observations")
    # Normalization before multiplication avoids unnecessary large-weight overflow.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            total_weight = float(w[valid].sum())
            norm = w[valid] / total_weight
            error = np.abs(pred[valid] - true[valid])
            scale = float(error.max())
            mae = float(np.sum(norm * error))
            rmse = 0.0 if scale == 0 else float(scale * np.sqrt(np.sum(norm * (error / scale) ** 2)))
        except FloatingPointError as exc:
            raise ValueError("Numeric overflow in metrics; check data scale") from exc
    counts = {"n_total": int(true.size), "n_valid": int(valid.sum()),
              "n_missing": int((~finite).sum()), "n_zero_weight": int((finite & (w == 0)).sum()),
              "weight_sum": total_weight, "direction": "min"}
    return {"MAE": {"value": mae, **counts}, "RMSE": {"value": rmse, **counts}}
