"""Seed statistics and predeclared Golden Reference checks, no fitting."""
import json
import math
import numpy as np
import pandas as pd
from researchbench.adapters.files import select_json
from researchbench.adapters.pacsr import require_keys, source_file

GROUP = ["dataset", "model", "variant", "split", "evaluation_id", "data_kind", "protocol_id", "horizon", "metric", "unit", "verification"]


def summarize_pacsr(rows):
    frame = pd.DataFrame(rows)
    if frame.duplicated(GROUP + ["seed"]).any():
        raise ValueError("Duplicate seed for the same protocol/model/variant/metric")
    results = []
    for identity, group in frame.groupby(GROUP, dropna=False, sort=True):
        if group.model_config_id.nunique() != 1:
            raise ValueError("Model configuration differs across seeds; use a different variant")
        values = group.value.to_numpy(dtype=float)
        complete = bool(np.isfinite(values).all())
        results.append({**dict(zip(GROUP, identity)), "mean": float(values.mean()) if complete else None,
                        "std": float(values.std(ddof=1)) if complete and len(values) > 1 else None,
                        "n_seeds": len(values), "n_valid_seeds": int(np.isfinite(values).sum()),
                        "ddof": 1, "statistical_unit": "training seed", "weighting": "equal seeds",
                        "seeds": json.dumps(sorted(group.seed.tolist())),
                        "run_ids": json.dumps(sorted(group.run_id.unique().tolist()))})
    return results


def compare_golden(rows, spec, *, base_dir, run, sources):
    """Map original JSON scalars explicitly; compare *all* planned outputs.

    The reference is read, never executed or rewritten. Identity checks and
    declared source hashes bind a reference to the audited scope. They are
    assertions supplied by the investigator, not proof of authenticity.
    """
    require_keys(spec, {"path", "sha256", "format", "identity", "bindings", "metrics", "atol", "rtol"})
    for name in ("atol", "rtol"):
        value = spec[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Golden tolerances must be predeclared finite nonnegative numbers")
    if spec["format"] != "json":
        raise ValueError("Golden Reference must be an original JSON report with explicit selectors")
    raw, source = source_file(base_dir / spec["path"], "golden", spec["sha256"])
    original = json.loads(raw)
    require_keys(spec["identity"], {"dataset", "seed"}, {"model", "variant", "split"})
    for name, selector in spec["identity"].items():
        if select_json(original, selector) != run[name]:
            raise ValueError(f"Golden identity mismatch: {name}")
    require_keys(spec["bindings"], {"cache", "evaluator", "test_scope", "model_config", "normalization", "split"})
    for role, hashes in spec["bindings"].items():
        actual = sorted({s["sha256"] for s in sources if s["role"] == role})
        if not isinstance(hashes, list) or sorted(set(hashes)) != actual:
            raise ValueError(f"Golden source binding mismatch: {role}")
    mapping = spec["metrics"]
    expected = {f"{r['horizon']}/{r['metric']}" for r in rows}
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError("Golden mapping must cover exactly all planned horizon/metric rows")
    comparisons = []
    for row in rows:
        key = f"{row['horizon']}/{row['metric']}"
        ref = select_json(original, mapping[key])
        if ref is not None and (type(ref) not in (int, float) or not math.isfinite(ref)):
            raise ValueError(f"Nonfinite/nonnumeric Golden value: {key}")
        value = row["value"]
        available = value is not None and ref is not None
        error = abs(value - ref) if available else None
        tolerance = spec["atol"] + spec["rtol"] * abs(ref) if available else None
        comparisons.append({**row, "reference": ref, "abs_error": error,
            "relative_error": error / abs(ref) if available and ref != 0 else None,
            "atol": spec["atol"], "rtol": spec["rtol"], "allowed_error": tolerance,
            "comparison": ("pass" if error <= tolerance else "fail") if available else "Unverified",
            "identity_fields_checked": ";".join(sorted(spec["identity"])),
            "identity_fields_unverified": ";".join(sorted({"dataset", "model", "variant", "seed", "split"} - set(spec["identity"]))),
            "reference_sha256": source["sha256"], "reference_selector": json.dumps(mapping[key])})
    return comparisons, source
