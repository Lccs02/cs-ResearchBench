"""Read-only, explicit PACSR cache contract. Never import evaluation/training code."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from researchbench.adapters.files import read_arrays
from researchbench.metrics.pacsr import pacsr_metrics

VERSION = "pacsr-audit/1.0"
AXES = ["sample", "horizon", "node", "channel"]
_UNSET = object()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"Required keys: {sorted(required)}; optional: {sorted(optional)}")


def source_file(path, role, expected=_UNSET):
    if expected is not _UNSET and (not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("Explicit lowercase SHA-256 required")
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not _UNSET and digest != expected:
        raise ValueError(f"Source hash mismatch ({role})")
    return raw, {"path": str(path), "sha256": digest, "role": role, "bytes": len(raw)}


def inspect_cache(path, fmt):
    raw, source = source_file(path, "cache")
    if fmt == "npz":
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            arrays = {k: np.asarray(data[k]) for k in data.files}
    elif fmt == "csv":
        data = pd.read_csv(io.BytesIO(raw))
        arrays = {k: data[k].to_numpy() for k in data.columns}
    elif fmt == "json":
        obj = json.loads(raw)
        arrays = {}
        def walk(item, parts):
            if isinstance(item, dict):
                for key, value in item.items():
                    walk(value, [*parts, key])
            else:
                try:
                    arrays[json.dumps(parts)] = np.asarray(item)
                except ValueError:
                    arrays[json.dumps(parts)] = np.asarray(item, dtype=object)
        walk(obj, [])
    else:
        raise ValueError("format must be csv, json or npz")
    return {"schema_version": VERSION, "source": source,
            "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()},
            "axes": "Unverified", "normalization": "Unverified", "mask": "Unverified",
            "test_scope": "Unverified", "model_config": "Unverified"}


class PACSRAdapter:
    """Dedicated audit API; read() remains compatible with the original adapter seam."""
    project_name = "PACSR"

    def read(self, run, *, base_dir, config_source):
        if not run:
            raise NotImplementedError("PACSR real format Unverified; use explicit pacsr-run configuration")
        raise ValueError("PACSR audit returns multiple horizon/channel results; use evaluate() or pacsr-run")

    def evaluate(self, run, *, base_dir):
        require_keys(run, {"run_id", "dataset", "model", "variant", "seed", "split", "evaluation_id",
                           "data_kind", "model_config", "input", "evaluation", "test_scope", "evidence", "golden"})
        for key in ("run_id", "dataset", "model", "variant", "split", "evaluation_id"):
            if not isinstance(run[key], str) or not run[key].strip():
                raise ValueError(f"{key} must be nonempty text")
        if type(run["seed"]) is not int or run["seed"] < 0 or run["split"] != "test":
            raise ValueError("Explicit nonnegative seed and test split required")
        if run["data_kind"] not in {"real", "synthetic", "unknown"} or not isinstance(run["model_config"], dict):
            raise ValueError("Invalid data_kind/model_config")
        sources = []

        def load(spec, role):
            require_keys(spec, {"path", "sha256", "format", "mapping"})
            raw, source = source_file(base_dir / spec["path"], role, spec["sha256"])
            source.update(format=spec["format"], mapping=spec["mapping"])
            arrays = read_arrays(raw, spec["format"], spec["mapping"])
            source["shapes"] = {k: list(v.shape) for k, v in arrays.items()}
            sources.append(source)
            return arrays

        inp = run["input"]
        require_keys(inp, {"file", "axes", "expected_shape"})
        axes = inp["axes"]
        if len(axes) != 4 or set(axes) != set(AXES):
            raise ValueError("Explicit axes must be a permutation of sample,horizon,node,channel")
        arrays = load(inp["file"], "cache")
        if set(arrays) != {"target", "prediction", "mask", "test_ids"}:
            raise ValueError("Map target,prediction,mask,test_ids explicitly")
        shape = inp["expected_shape"]
        if any(list(arrays[key].shape) != shape for key in ("target", "prediction", "mask")):
            raise ValueError("Cache shape differs from expected_shape")
        order = [axes.index(name) for name in AXES]
        target, prediction, mask = (arrays[key].transpose(order) for key in ("target", "prediction", "mask"))
        scope = run["test_scope"]
        require_keys(scope, {"expected_ids", "id_axes", "description"})
        if not isinstance(scope["description"], str) or not scope["description"]:
            raise ValueError("Describe full test range, window overlap and sampling")
        expected = load(scope["expected_ids"], "test_scope")
        if set(expected) != {"test_ids"} or scope["id_axes"] not in (["sample"], ["sample", "horizon"]):
            raise ValueError("Test identifiers require explicit sample or sample,horizon axes")
        ids = arrays["test_ids"]
        expected_shape = target.shape[:len(scope["id_axes"])]
        if ids.shape != expected_shape or not np.array_equal(ids, expected["test_ids"]):
            raise ValueError("Full test identifiers differ (shape, order or values); no subset selection allowed")
        if len(np.unique(ids, axis=0)) != len(ids):
            raise ValueError("Duplicate test window identifiers")
        ev = run["evaluation"]
        require_keys(ev, {"total_channel", "component_channels", "fine_reduction", "horizons",
                          "normalization", "aggregation", "ncv_epsilon", "units", "missing", "metrics"})
        if ev["missing"] != "finite_target_and_mask_reject_nonfinite_prediction":
            raise ValueError("Unsupported PACSR missing policy")
        if not isinstance(ev["units"], str) or not ev["units"]:
            raise ValueError("Explicit metric units required")
        norm = ev["normalization"]
        if not isinstance(norm, dict):
            raise ValueError("Normalization must be an explicit mapping")
        if norm.get("kind") == "identity":
            require_keys(norm, {"kind", "description"})
        elif norm.get("kind") == "affine":
            require_keys(norm, {"kind", "description", "parameters", "axes", "dtype"})
            if norm["dtype"] not in {"float32", "float64"}:
                raise ValueError("Explicit affine dtype required")
            params = load(norm["parameters"], "normalization")
            if set(params) != {"scale", "offset"}:
                raise ValueError("Affine requires scale and offset arrays")
            dims = norm["axes"]
            if len(set(dims)) != len(dims) or any(x not in AXES for x in dims):
                raise ValueError("Invalid affine parameter axes")
            expected_shape = tuple(target.shape[AXES.index(x)] for x in dims)
            expanded = {}
            for k, value in params.items():
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise ValueError("Affine parameter shape/finite mismatch; no implicit broadcasting")
                sorted_dims = sorted(range(len(dims)), key=lambda i: AXES.index(dims[i]))
                ordered = value.transpose(sorted_dims) if dims else value
                expanded[k] = ordered.reshape([target.shape[i] if name in dims else 1 for i, name in enumerate(AXES)]).astype(norm["dtype"])
            if (expanded["scale"] <= 0).any():
                raise ValueError("Affine scale must be positive")
            target = target.astype(norm["dtype"]) * expanded["scale"] + expanded["offset"]
            prediction = prediction.astype(norm["dtype"]) * expanded["scale"] + expanded["offset"]
        else:
            raise ValueError("Normalization must be explicitly identity or affine")
        if not isinstance(norm["description"], str) or not norm["description"]:
            raise ValueError("Normalization evidence description required")
        evidence = run["evidence"]
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        roles = set()
        for entry in evidence:
            require_keys(entry, {"role", "path", "sha256"})
            _, source = source_file(base_dir / entry["path"], entry["role"], entry["sha256"])
            sources.append(source)
            roles.add(entry["role"])
        if not {"evaluator", "model_config", "split", "normalization"} <= roles:
            raise ValueError("Evidence requires evaluator,model_config,split,normalization files")
        horizons = ev["horizons"]
        if not isinstance(horizons, list) or not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or not 1 <= h <= target.shape[1] for h in horizons):
            raise ValueError("horizons must be unique explicit 1-based indices in range")
        # A protocol includes shape, exact test IDs, semantics, and evaluator identity.
        protocol = {"evaluation": ev, "canonical_shape": list(target.shape),
                    "test_ids": fingerprint(ids.tolist()), "test_scope": scope["description"],
                    "evaluator": sorted(s["sha256"] for s in sources if s["role"] == "evaluator")}
        # Parameter paths are evidence, not semantic differences between equivalent runs.
        if norm["kind"] == "affine":
            protocol = json.loads(json.dumps(protocol))
            protocol["evaluation"]["normalization"]["parameters"].pop("path")
        planned = ev["metrics"]
        if not isinstance(planned, dict) or set(planned) != {"all", *[f"h{h}" for h in horizons]}:
            raise ValueError("Planned metrics must cover all and each selected horizon")
        rows = []
        for horizon in ["all", *[f"h{h}" for h in horizons]]:
            selected = slice(None) if horizon == "all" else slice(int(horizon[1:]) - 1, int(horizon[1:]))
            metrics = pacsr_metrics(target[:, selected], prediction[:, selected], mask[:, selected],
                total=ev["total_channel"], components=ev["component_channels"], fine_reduction=ev["fine_reduction"],
                aggregation=ev["aggregation"], ncv_epsilon=ev["ncv_epsilon"])
            selected_metrics = planned[horizon]
            if not isinstance(selected_metrics, list) or not selected_metrics or any(not isinstance(m, str) for m in selected_metrics) or len(set(selected_metrics)) != len(selected_metrics) or set(selected_metrics) - set(metrics):
                raise ValueError("Unknown, duplicate or empty planned metric selection")
            for metric in selected_metrics:
                values = metrics[metric]
                rows.append({**{k: run[k] for k in ("run_id", "dataset", "model", "variant", "seed", "split", "evaluation_id", "data_kind")},
                    "horizon": horizon, "metric": metric, **values,
                    "unit": "dimensionless" if metric == "ncv" else ev["units"],
                    "protocol_id": fingerprint(protocol), "model_config_id": fingerprint(run["model_config"]),
                    "verification": "Unverified"})
        return {"rows": rows, "sources": sources, "protocol": protocol}


def load_audit(path):
    path = Path(path).resolve()
    raw, config_source = source_file(path, "audit_config")
    config = yaml.safe_load(raw)
    require_keys(config, {"schema_version", "runs"})
    if config["schema_version"] != VERSION or not isinstance(config["runs"], list) or not config["runs"]:
        raise ValueError("Expected nonempty pacsr-audit/1.0 configuration")
    if len({run["run_id"] for run in config["runs"]}) != len(config["runs"]):
        raise ValueError("Duplicate run_id")
    adapter = PACSRAdapter()
    audits = [adapter.evaluate(run, base_dir=path.parent) for run in config["runs"]]
    return config, config_source, audits
