"""One source file per run; no project-specific key or dimension assumptions."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from researchbench.core.schema import ExperimentRecord, SCHEMA_VERSION, strict_keys, validate_records
from researchbench.metrics.regression import regression_metrics


def select_json(obj, selector):
    """A string selects a literal key; a list traverses dict keys/list indices."""
    parts = selector if isinstance(selector, list) else [selector]
    for part in parts:
        obj = obj[part]
    return obj


def read_arrays(raw: bytes, fmt: str, mappings: dict) -> dict:
    if fmt == "csv":
        frame = pd.read_csv(io.BytesIO(raw))
        # A column list preserves its 2D shape, including a single-column list.
        return {key: frame[selector].to_numpy() for key, selector in mappings.items()}
    if fmt == "json":
        obj = json.loads(raw)
        return {key: np.asarray(select_json(obj, selector)) for key, selector in mappings.items()}
    if fmt == "npz":
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            return {key: np.array(archive[selector], copy=True) for key, selector in mappings.items()}
    raise ValueError("format must be csv, json, or npz")


class ConfigurableFileAdapter:
    def read(self, run: dict, *, base_dir: Path, config_source: dict) -> ExperimentRecord:
        strict_keys(run, {"run_id", "project", "dataset", "model", "seed", "split", "evaluation_id", "config", "data_kind", "input", "evaluation"}, "run")
        inp, ev = run["input"], run["evaluation"]
        strict_keys(inp, {"path", "format", "mapping", "expected_shape"}, "input")
        strict_keys(ev, {"mode", "missing", "statistical_unit", "units", "directions"}, "evaluation")
        mode = ev["mode"]
        path = (base_dir / inp["path"]).resolve()
        raw = path.read_bytes()
        mappings = inp["mapping"]
        if not isinstance(mappings, dict) or not mappings:
            raise ValueError("input.mapping must be nonempty")
        arrays = read_arrays(raw, inp["format"], mappings)
        if mode == "predictions":
            if set(arrays) not in ({"target", "prediction"}, {"target", "prediction", "weights"}):
                raise ValueError("prediction mapping requires target, prediction, optional weights")
            shape = list(arrays["target"].shape)
            if "expected_shape" in inp and inp["expected_shape"] != shape:
                raise ValueError(f"expected_shape mismatch: got {shape}")
            if "directions" in ev:
                raise ValueError("MAE/RMSE directions are fixed to min")
            metrics = regression_metrics(arrays["target"], arrays["prediction"], weights=arrays.get("weights"), missing=ev.get("missing", "error"))
            protocol = {"mode": mode, "missing": ev.get("missing", "error"),
                        "weighting": "explicit" if "weights" in arrays else "uniform",
                        "statistical_unit": ev["statistical_unit"], "reduction": "all_selected_elements",
                        "metric_units": {name: ev["units"] for name in metrics}}
        elif mode == "precomputed":
            if "missing" in ev or "expected_shape" in inp:
                raise ValueError("precomputed mode does not accept missing or expected_shape")
            metrics = {}
            for name, array in arrays.items():
                if array.size != 1:
                    raise ValueError("Precomputed metrics must be scalar; do not silently average episodes/steps")
                metrics[name] = {"value": float(array.item()), "direction": ev["directions"][name]}
            if set(ev["directions"]) != set(metrics):
                raise ValueError("directions must match metric names")
            protocol = {"mode": mode, "missing": "not_applicable", "weighting": "reported",
                        "statistical_unit": ev["statistical_unit"], "reduction": "reported_scalar",
                        "metric_units": ev["units"]}
        else:
            raise ValueError("evaluation.mode must be predictions or precomputed")
        source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "role": "data",
                  "format": inp["format"], "mapping": mappings,
                  "shape": {k: list(v.shape) for k, v in arrays.items()}}
        record = ExperimentRecord(schema_version=SCHEMA_VERSION,
            **{key: run[key] for key in ("run_id", "project", "dataset", "model", "seed", "split", "evaluation_id", "config")},
            metrics=metrics, protocol=protocol, sources=[source, config_source],
            data_kind=run.get("data_kind", "unknown"))
        record.validate()
        return record


def ingest_config(path: Path, adapter=None) -> list[ExperimentRecord]:
    path = path.resolve()
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    strict_keys(config, {"schema_version", "runs"}, "configuration")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported configuration schema_version; use quoted '1.0'")
    if not isinstance(config.get("runs"), list):
        raise ValueError("runs must be a list")
    source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "role": "config", "format": "yaml"}
    adapter = adapter or ConfigurableFileAdapter()
    records = [adapter.read(run, base_dir=path.parent, config_source=source) for run in config["runs"]]
    validate_records(records)
    return records
