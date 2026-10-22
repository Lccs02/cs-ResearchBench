"""Fixed-seed software fixtures ONLY. These are not research/paper results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def generate(directory: Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for model, scale in (("baseline", 0.4), ("variant", 0.25)):
        for seed, fmt in zip((42, 716, 2026), ("csv", "json", "npz")):
            # Same targets and standardized errors across paired models.
            rng = np.random.default_rng(seed)
            target = rng.normal(size=(8, 3))
            pred = target + rng.normal(size=target.shape) * scale
            weights = np.ones_like(target)
            filename = f"{model}_{seed}.{fmt}"
            if fmt == "csv":
                pd.DataFrame({"observed": target.ravel(), "estimated": pred.ravel(), "importance": weights.ravel()}).to_csv(directory / filename, index=False)
                mapping = {"target": "observed", "prediction": "estimated", "weights": "importance"}
                shape = [24]
            elif fmt == "json":
                (directory / filename).write_text(json.dumps({"payload": {"truth": target.tolist(), "forecast": pred.tolist(), "weights": weights.tolist()}}), encoding="utf-8")
                mapping = {"target": ["payload", "truth"], "prediction": ["payload", "forecast"], "weights": ["payload", "weights"]}
                shape = [8, 3]
            else:
                np.savez(directory / filename, actual=target, estimate=pred, importance=weights)
                mapping = {"target": "actual", "prediction": "estimate", "weights": "importance"}
                shape = [8, 3]
            runs.append({"run_id": f"synthetic-{model}-{seed}", "project": "synthetic-demo", "dataset": "synthetic-normal-v1", "model": model, "seed": seed,
                         "split": "test", "evaluation_id": "synthetic-elements-v1", "config": {"noise_scale": scale}, "data_kind": "synthetic",
                         "input": {"path": filename, "format": fmt, "mapping": mapping, "expected_shape": shape},
                         "evaluation": {"mode": "predictions", "missing": "error", "statistical_unit": "selected scalar element", "units": "synthetic units"}})
    config = directory / "synthetic.yaml"
    config.write_text(yaml.safe_dump({"schema_version": "1.0", "runs": runs}, sort_keys=False), encoding="utf-8")
    (directory / "NOTICE.txt").write_text("SYNTHETIC SOFTWARE FIXTURES ONLY. Scientific status: Unverified. Never use as paper results.\n", encoding="utf-8")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    print(generate(args.output).resolve())
