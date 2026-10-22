"""Equal-seed summaries and strict paired ablations (no significance claims)."""
import math
import numpy as np
import pandas as pd
from researchbench.core.schema import canonical, digest, validate_records

CONTEXT = ["project", "dataset", "split", "evaluation_id", "protocol_id", "data_kind", "verification", "metric", "direction", "unit"]


def metric_rows(records):
    validate_records(records)
    rows = []
    for r in records:
        for name, metric in r.metrics.items():
            rows.append({**{k: getattr(r, k) for k in ("project", "dataset", "model", "seed", "split", "evaluation_id", "data_kind", "verification", "run_id")},
                         "protocol_id": digest(r.protocol), "config_id": digest(r.config),
                         "metric": name, "unit": r.protocol["metric_units"][name],
                         "source_ids": canonical([s["sha256"] for s in r.sources]), **metric})
    frame = pd.DataFrame(rows)
    keys = CONTEXT + ["model"]
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate seed in an analysis group; do not count repeated runs as independent seeds")
    for _, group in frame.groupby(keys, dropna=False, sort=True):
        if group.config_id.nunique() != 1:
            raise ValueError("Different model configs across seeds; assign distinct model/variant names")
    return frame


def stats(values):
    a = np.asarray(values, dtype=float)
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if len(a) > 1 else None
    if not math.isfinite(mean) or (std is not None and not math.isfinite(std)):
        raise ValueError("Numeric overflow in seed statistics")
    return {"mean": mean, "std": std, "n_seeds": len(a), "ddof": 1,
            "std_status": "Defined" if len(a) > 1 else "Unverified: fewer than 2 seeds"}


def summarize(records):
    frame = metric_rows(records)
    results = []
    keys = CONTEXT + ["model", "config_id"]
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        results.append({**dict(zip(keys, values)), **stats(group.value), "statistical_unit": "seed",
                        "seed_weighting": "equal", "seeds": canonical(sorted(group.seed.tolist())),
                        "run_ids": canonical(sorted(group.run_id.tolist()))})
    return pd.DataFrame(results)


def ablations(records, baseline):
    """Compare variant - baseline on exactly matching seeds and protocols.

    Improvement: (baseline-variant)/abs(baseline)*100 for min metrics;
    reverse numerator for max metrics. Undefined at baseline mean == 0.
    """
    frame = metric_rows(records)
    if baseline not in set(frame.model):
        raise ValueError(f"Baseline model not found: {baseline}")
    results = []
    for values, group in frame.groupby(CONTEXT, sort=True, dropna=False):
        base = group[group.model == baseline].set_index("seed")
        variants = group[group.model != baseline]
        if len(variants) and base.empty:
            raise ValueError("Variant has no baseline with matching evaluation/protocol/metric")
        for model, variant in variants.groupby("model", sort=True):
            variant = variant.set_index("seed")
            if set(base.index) != set(variant.index):
                raise ValueError(f"Seed sets differ for {baseline} vs {model}; paired analysis requires identical seeds")
            seeds = sorted(base.index)
            b, v = base.loc[seeds, "value"].to_numpy(), variant.loc[seeds, "value"].to_numpy()
            s = stats(v - b)
            bm, vm = float(b.mean()), float(v.mean())
            direction = group.direction.iloc[0]
            improvement = None if bm == 0 else (bm - vm) / abs(bm) * 100 * (1 if direction == "min" else -1)
            if improvement is not None and not math.isfinite(improvement):
                raise ValueError("Numeric overflow in relative improvement; check baseline scale")
            results.append({**dict(zip(CONTEXT, values)), "baseline": baseline, "variant": model,
                            "baseline_mean": bm, "variant_mean": vm,
                            "delta_mean": s["mean"], "delta_std": s["std"], "ddof": 1,
                            "n_pairs": s["n_seeds"], "std_status": s["std_status"],
                            "improvement_pct": improvement,
                            "improvement_status": "Defined" if bm != 0 else "Unverified: zero baseline mean",
                            "statistical_unit": "paired_seed", "seeds": canonical(seeds),
                            "run_ids": canonical(sorted(base.run_id.tolist() + variant.run_id.tolist()))})
    if not results:
        raise ValueError("No comparable variant models for ablation")
    return pd.DataFrame(results)
