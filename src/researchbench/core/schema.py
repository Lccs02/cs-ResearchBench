"""Versioned, JSON-serializable records. Validation is not scientific verification."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

SCHEMA_VERSION = "1.0"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def strict_keys(obj: dict, allowed: set[str], name: str) -> None:
    if not isinstance(obj, dict) or set(obj) - allowed:
        raise ValueError(f"{name}: unknown fields or not a mapping: {obj}")


@dataclass
class ExperimentRecord:
    schema_version: str
    run_id: str
    project: str
    dataset: str
    model: str
    seed: int
    split: str
    evaluation_id: str
    config: dict
    metrics: dict
    protocol: dict
    sources: list[dict]
    data_kind: str = "unknown"
    verification: str = "Unverified"

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        for key in ("run_id", "project", "dataset", "model", "split", "evaluation_id"):
            require_text(getattr(self, key), key)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.verification != "Unverified":
            raise ValueError("MVP cannot verify scientific authenticity: use Unverified")
        if self.data_kind not in {"synthetic", "real", "unknown"}:
            raise ValueError("data_kind must be synthetic, real, or unknown")
        if not isinstance(self.config, dict):
            raise ValueError("config must be a mapping")
        canonical(self.config)
        strict_keys(self.protocol, {"mode", "missing", "weighting", "statistical_unit", "reduction", "metric_units"}, "protocol")
        if self.protocol.get("mode") not in {"predictions", "precomputed"}:
            raise ValueError("Invalid protocol mode")
        for key in ("missing", "weighting", "statistical_unit", "reduction"):
            require_text(self.protocol.get(key), f"protocol.{key}")
        if self.protocol["missing"] not in {"error", "omit", "not_applicable"}:
            raise ValueError("Invalid missing policy")
        if self.protocol["mode"] == "predictions":
            if self.protocol["missing"] not in {"error", "omit"} or self.protocol["weighting"] not in {"uniform", "explicit"} or self.protocol["reduction"] != "all_selected_elements":
                raise ValueError("Invalid prediction protocol")
        elif any(self.protocol[k] != v for k, v in {"missing": "not_applicable", "weighting": "reported", "reduction": "reported_scalar"}.items()):
            raise ValueError("Invalid precomputed protocol")
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise ValueError("metrics must be nonempty")
        units = self.protocol.get("metric_units")
        if not isinstance(units, dict) or set(units) != set(self.metrics):
            raise ValueError("metric_units must match metric names")
        if self.protocol["mode"] == "predictions" and set(self.metrics) != {"MAE", "RMSE"}:
            raise ValueError("Prediction records must contain MAE and RMSE")
        for name, metric in self.metrics.items():
            require_text(name, "metric name")
            require_text(units[name], f"metric_units.{name}")
            strict_keys(metric, {"value", "direction", "n_total", "n_valid", "n_missing", "n_zero_weight", "weight_sum"}, "metric")
            value = metric.get("value")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}: value must be finite")
            if metric.get("direction") not in {"min", "max"}:
                raise ValueError(f"{name}: direction must be min or max")
            if self.protocol["mode"] == "predictions":
                if value < 0 or metric["direction"] != "min":
                    raise ValueError(f"{name}: regression errors must be nonnegative and minimized")
                for key in ("n_total", "n_valid", "n_missing", "n_zero_weight"):
                    if type(metric.get(key)) is not int or metric[key] < 0:
                        raise ValueError(f"{name}: invalid {key}")
                if metric["n_valid"] < 1 or metric["n_total"] != metric["n_valid"] + metric["n_missing"] + metric["n_zero_weight"]:
                    raise ValueError(f"{name}: inconsistent counts")
                w = metric.get("weight_sum")
                if type(w) not in (int, float) or not math.isfinite(w) or w <= 0:
                    raise ValueError(f"{name}: invalid weight_sum")
        if not isinstance(self.sources, list) or not self.sources:
            raise ValueError("sources must be nonempty")
        for source in self.sources:
            strict_keys(source, {"path", "sha256", "role", "format", "mapping", "shape"}, "source")
            for key in ("path", "role", "format"):
                require_text(source.get(key), f"source.{key}")
            if not re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")):
                raise ValueError("source.sha256 must contain 64 hexadecimal characters")
            if source["role"] not in {"data", "config"} or source["format"] not in {"csv", "json", "npz", "yaml"}:
                raise ValueError("Invalid source role or format")
        if not any(s["role"] == "data" for s in self.sources):
            raise ValueError("At least one data source is required")
        canonical(asdict(self))

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRecord":
        try:
            record = cls(**data)
        except TypeError as exc:
            raise ValueError(f"Invalid record fields: {exc}") from exc
        record.validate()
        return record


def validate_records(records: list[ExperimentRecord]) -> None:
    if not records:
        raise ValueError("At least one experiment record is required")
    seen = set()
    for record in records:
        record.validate()
        if record.run_id in seen:
            raise ValueError(f"Duplicate run_id: {record.run_id}")
        seen.add(record.run_id)


def write_records(records: list[ExperimentRecord], path: Path) -> None:
    validate_records(records)
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "records": [r.to_dict() for r in records]}, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_records(path: Path) -> list[ExperimentRecord]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    strict_keys(obj, {"schema_version", "records"}, "record document")
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported document schema_version")
    records = [ExperimentRecord.from_dict(x) for x in obj["records"]]
    validate_records(records)
    return records
