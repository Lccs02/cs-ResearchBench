"""Fixed-seed software fixture only. NEVER a real PACSR cache or paper result.

The independent reference below deliberately uses sum-of-errors formulas rather
than ResearchBench functions. It is not an original PACSR Golden Reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_metrics(target, prediction, mask):
    valid = mask.astype(bool) & np.isfinite(target)
    out = {}
    maes, rmses = [], []
    for c in range(target.shape[-1]):
        errors = (prediction[..., c].astype(float) - target[..., c].astype(float))[valid[..., c]]
        mae = float(np.abs(errors).sum() / len(errors)) if len(errors) else None
        rmse = float(np.sqrt(np.square(errors).sum() / len(errors))) if len(errors) else None
        prefix = "coarse" if c == 0 else f"component_{c}"
        out.update({f"{prefix}_mae": mae, f"{prefix}_rmse": rmse})
        if c and len(errors):
            maes.append(mae)
            rmses.append(rmse)
    out["fine_mae_macro"] = float(np.mean(maes)) if maes else None
    out["fine_rmse_macro"] = float(np.mean(rmses)) if rmses else None
    gap = np.abs(prediction[..., 0] - prediction[..., 1:].sum(-1)).astype(float)
    joint = valid.all(-1)
    out.update(cv=float(gap[joint].mean()), max_cv=float(gap[joint].max()),
               ncv=float(gap[joint].sum() / (np.abs(target[..., 0][joint].astype(float)).sum() + 1e-8)),
               cv_all_forecasts=float(gap.mean()))
    return out


def generate(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    ids = np.arange(24).reshape(8, 3) + 1000
    np.savez(destination / "scope.npz", ids=ids)
    metadata = {"dataset": "SYNTHETIC_ONLY", "model": "synthetic-model", "variant": "synthetic-variant", "split": "test"}
    (destination / "evidence.json").write_text(json.dumps({"status": "Unverified", "synthetic": True,
        "model_config": {"fixture": True}, "test_range": [1000, 1024], "normalization": "identity"}), encoding="utf-8")
    evidence_path = destination / "evidence.json"
    evaluator = Path(__file__).resolve()
    runs = []
    for seed in (17, 29):
        rng = np.random.default_rng(seed)
        target = rng.uniform(1, 10, (8, 3, 2, 3)).astype(np.float32)
        prediction = target + rng.normal(0, .5, target.shape).astype(np.float32)
        mask = rng.random(target.shape) > .15
        mask[0, 0] = True
        target[1, 1, 0, 2] = np.nan
        cache = destination / f"synthetic-{seed}.npz"
        np.savez(cache, observed=target, forecast=prediction, observed_mask=mask, ids=ids)
        gold = {**metadata, "seed": seed, "metrics": {"all": reference_metrics(target, prediction, mask)}}
        for h in (1, 3):
            gold["metrics"][f"h{h}"] = reference_metrics(target[:, h-1:h], prediction[:, h-1:h], mask[:, h-1:h])
        golden = destination / f"synthetic-reference-{seed}.json"
        golden.write_text(json.dumps(gold, allow_nan=False), encoding="utf-8")
        descriptor = lambda path, mapping: {"path": str(path), "sha256": digest(path), "format": "npz", "mapping": mapping}
        bindings = {"cache": [digest(cache)], "evaluator": [digest(evaluator)], "test_scope": [digest(destination / "scope.npz")]}
        bindings.update({key: [digest(evidence_path)] for key in ("model_config", "normalization", "split")})
        runs.append({**metadata, "run_id": f"synthetic-{seed}", "seed": seed, "evaluation_id": "synthetic-fixed-full-test",
            "model_config": {"fixture": True}, "data_kind": "synthetic",
            "input": {"file": descriptor(cache, {"target": "observed", "prediction": "forecast", "mask": "observed_mask", "test_ids": "ids"}),
                      "axes": ["sample", "horizon", "node", "channel"], "expected_shape": list(target.shape)},
            "evaluation": {"total_channel": 0, "component_channels": [1, 2], "fine_reduction": "macro", "horizons": [1, 3],
                "normalization": {"kind": "identity", "description": "Synthetic arrays already in evaluated units"},
                "aggregation": {"kind": "sum", "dtype": "float32"}, "ncv_epsilon": 1e-8, "units": "synthetic units",
                "missing": "finite_target_and_mask_reject_nonfinite_prediction",
                "metrics": {h: list(metrics) for h, metrics in gold["metrics"].items()}},
            "test_scope": {"expected_ids": descriptor(destination / "scope.npz", {"test_ids": "ids"}),
                           "id_axes": ["sample", "horizon"], "description": "Eight synthetic windows; all 3 horizons, no subset or truncation"},
            "evidence": [{"role": "evaluator", "path": str(evaluator), "sha256": digest(evaluator)},
                         *[{"role": role, "path": str(evidence_path), "sha256": digest(evidence_path)} for role in ("model_config", "split", "normalization")]],
            "golden": {"path": str(golden), "sha256": digest(golden), "format": "json", "identity": {k: k for k in gold if k != "metrics"},
                "bindings": bindings, "atol": 1e-10, "rtol": 1e-10,
                "metrics": {f"{h}/{m}": ["metrics", h, m] for h, metrics in gold["metrics"].items() for m in metrics}}})
    config = destination / "audit.yaml"
    config.write_text(yaml.safe_dump({"schema_version": "pacsr-audit/1.0", "runs": runs}, sort_keys=False), encoding="utf-8")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/generated/pacsr"))
    print(generate(parser.parse_args().output))
