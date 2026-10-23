"""Synthetic implementation tests; no real-cache reproduction claim."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml
from researchbench.adapters.pacsr import PACSRAdapter, inspect_cache, load_audit
from researchbench.analysis.pacsr import compare_golden, summarize_pacsr
from researchbench.metrics.pacsr import aggregate_components, pacsr_metrics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixture_generator", ROOT / "examples/generate_pacsr_synthetic.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture
def fixture(tmp_path):
    path = generator.generate(tmp_path / "synthetic")
    config, source, audits = load_audit(path)
    return path, config, audits


def test_independent_reference_parity_and_seed_summary(fixture):
    path, config, audits = fixture
    for run, audit in zip(config["runs"], audits):
        compared, _ = compare_golden(audit["rows"], run["golden"], base_dir=path.parent, run=run, sources=audit["sources"])
        assert all(row["comparison"] == "pass" for row in compared)
        assert {row["horizon"] for row in compared} == {"all", "h1", "h3"}
    summary = summarize_pacsr([r for a in audits for r in a["rows"]])
    assert all(r["n_seeds"] == 2 and r["ddof"] == 1 for r in summary)
    first = next(r for r in summary if r["horizon"] == "all" and r["metric"] == "coarse_mae")
    values = [next(r["value"] for r in a["rows"] if r["horizon"] == "all" and r["metric"] == "coarse_mae") for a in audits]
    assert first["std"] == pytest.approx(np.std(values, ddof=1))


@pytest.mark.parametrize("kind,weights,expected", [("sum", None, 6), ("mean", None, 3), ("weighted_sum", [2, -1], 0)])
def test_aggregation_operator(kind, weights, expected):
    op = {"kind": kind, "dtype": "float64"}
    if weights is not None:
        op["weights"] = weights
    assert aggregate_components(np.array([[2, 4]]), op).item() == expected


def calculate(t, p, m, reduction="macro"):
    return pacsr_metrics(t, p, m, total=0, components=[1, 2], fine_reduction=reduction,
                         aggregation={"kind": "sum", "dtype": "float32"}, ncv_epsilon=1e-8)


def test_macro_is_not_pooled_rmse():
    t = np.zeros((2, 1, 1, 3))
    p = np.array([[[[1, 1, 10]]], [[[1, 3, 10]]]], dtype=float)
    m = np.ones_like(t, bool)
    m[1, 0, 0, 2] = False
    macro, pooled = calculate(t, p, m), calculate(t, p, m, "pooled")
    assert macro["fine_rmse_macro"]["value"] == pytest.approx((np.sqrt(5) + 10) / 2)
    assert pooled["fine_pooled_rmse"]["value"] == pytest.approx(np.sqrt(110 / 3))
    assert macro["cv"]["n_valid"] == 1
    assert macro["cv_all_forecasts"]["n_valid"] == 2


def test_empty_channels_are_unverified_not_zero():
    x = np.ones((2, 1, 1, 3))
    result = calculate(x, x, np.zeros_like(x))
    assert result["coarse_mae"]["value"] is None
    assert result["fine_rmse_macro"]["value"] is None
    assert result["cv"]["value"] is None
    assert result["cv_all_forecasts"]["value"] == 1


def test_nonfinite_predictions_rejected_even_masked():
    t = np.ones((2, 1, 1, 3))
    p = t.copy(); p[0, 0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite predictions"):
        calculate(t, p, np.zeros_like(t))


def test_nonfinite_target_intersects_mask():
    t = np.ones((2, 1, 1, 3)); t[0, 0, 0, 1] = np.nan
    result = calculate(t, np.ones_like(t), np.ones_like(t))
    assert result["component_1_mae"]["n_valid"] == 1
    assert result["cv"]["n_valid"] == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["input"].update(axes=["x"] * 4), "axes"),
    (lambda r: r["input"].update(expected_shape=[1, 2, 3, 4]), "shape"),
    (lambda r: r["evaluation"].update(horizons=[0]), "horizons"),
    (lambda r: r["evaluation"].update(component_channels=[0, 1]), "indices"),
    (lambda r: r["evaluation"]["normalization"].update(kind="guess"), "Normalization"),
    (lambda r: r.update(evidence=[]), "Evidence"),
    (lambda r: r["input"]["file"].update(sha256="0" * 64), "hash mismatch"),
    (lambda r: r["evaluation"].update(missing="omit"), "missing policy"),
])
def test_invalid_mapping_rejected(fixture, mutation, match):
    path, config, _ = fixture
    run = copy.deepcopy(config["runs"][0]); mutation(run)
    with pytest.raises(ValueError, match=match):
        PACSRAdapter().evaluate(run, base_dir=path.parent)


def test_test_scope_reorder_rejected(fixture):
    path, config, _ = fixture
    run = config["runs"][0]
    scope = Path(run["test_scope"]["expected_ids"]["path"])
    with np.load(scope) as z:
        ids = z["ids"][::-1]
    np.savez(scope, ids=ids)
    run["test_scope"]["expected_ids"]["sha256"] = generator.digest(scope)
    with pytest.raises(ValueError, match="Full test identifiers"):
        PACSRAdapter().evaluate(run, base_dir=path.parent)


@pytest.mark.parametrize("mutation,match", [
    (lambda g: g.update(atol=-1), "tolerances"),
    (lambda g: g.update(rtol=float("nan")), "tolerances"),
    (lambda g: g["metrics"].pop("all/cv"), "cover exactly"),
    (lambda g: g["bindings"].update(cache=["bad"]), "binding mismatch"),
    (lambda g: g["identity"].update(seed="dataset"), "identity mismatch"),
])
def test_golden_validation(fixture, mutation, match):
    path, config, audits = fixture
    run, audit = config["runs"][0], audits[0]
    mutation(run["golden"])
    with pytest.raises(ValueError, match=match):
        compare_golden(audit["rows"], run["golden"], base_dir=path.parent, run=run, sources=audit["sources"])


def test_comparison_failure_and_zero_reference(fixture):
    path, config, audits = fixture
    run, audit = config["runs"][0], audits[0]
    golden = Path(run["golden"]["path"])
    data = json.loads(golden.read_text()); data["metrics"]["all"]["cv"] = 0
    golden.write_text(json.dumps(data)); run["golden"]["sha256"] = generator.digest(golden)
    results, _ = compare_golden(audit["rows"], run["golden"], base_dir=path.parent, run=run, sources=audit["sources"])
    item = next(r for r in results if r["horizon"] == "all" and r["metric"] == "cv")
    assert item["comparison"] == "fail" and item["relative_error"] is None


def test_seed_duplicates_and_missing_std(fixture):
    _, _, audits = fixture
    rows = audits[0]["rows"]
    assert all(r["std"] is None for r in summarize_pacsr(rows))
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize_pacsr(rows + rows)


def test_inspection_read_only(fixture):
    _, config, _ = fixture
    path = Path(config["runs"][0]["input"]["file"]["path"])
    before = path.read_bytes()
    report = inspect_cache(path, "npz")
    assert report["arrays"]["forecast"]["shape"] == [8, 3, 2, 3]
    assert report["axes"] == "Unverified"
    assert path.read_bytes() == before


def test_cli_end_to_end(fixture, tmp_path):
    config_path, config, _ = fixture
    output = tmp_path / "report"
    before = {r["input"]["file"]["path"]: generator.digest(r["input"]["file"]["path"]) for r in config["runs"]}
    cmd = [sys.executable, "-m", "researchbench", "pacsr-run", str(config_path), "--output", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Unverified" in result.stdout
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["numerical_match"] is True and manifest["real_cache_acceptance"] == "Unverified"
    for name, sha in manifest["artifacts"].items():
        assert generator.digest(output / name) == sha
    assert list(output.glob("*.svg")) and list(output.glob("*.pdf"))
    assert "synthetic / Unverified" in (output / "summary.tex").read_text()
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    assert all(generator.digest(p) == sha for p, sha in before.items())


def test_explicit_axis_permutation(fixture):
    path, config, audits = fixture
    run = config["runs"][0]
    cache = Path(run["input"]["file"]["path"])
    with np.load(cache) as z:
        arrays = {key: z[key].copy() for key in z.files}
    for key in ("observed", "forecast", "observed_mask"):
        arrays[key] = arrays[key].transpose(3, 2, 0, 1)
    np.savez(cache, **arrays)
    run["input"].update(axes=["channel", "node", "sample", "horizon"], expected_shape=[3, 2, 8, 3])
    run["input"]["file"]["sha256"] = generator.digest(cache)
    result = PACSRAdapter().evaluate(run, base_dir=path.parent)
    assert result["rows"] == audits[0]["rows"]


def test_named_axis_affine_scaling(fixture):
    path, config, audits = fixture
    run = config["runs"][0]
    scale = path.parent / "scale.npz"
    np.savez(scale, s=np.array([2., 2.]), b=np.array([0., 0.]))
    run["evaluation"]["normalization"] = {"kind": "affine", "description": "Synthetic factor two", "axes": ["node"], "dtype": "float32",
        "parameters": {"path": str(scale), "sha256": generator.digest(scale), "format": "npz", "mapping": {"scale": "s", "offset": "b"}}}
    result = PACSRAdapter().evaluate(run, base_dir=path.parent)
    for original, transformed in zip(audits[0]["rows"], result["rows"]):
        if original["metric"] != "ncv":
            assert transformed["value"] == pytest.approx(2 * original["value"])
    assert any(s["role"] == "normalization" and s["path"] == str(scale) for s in result["sources"])


def test_predeclared_original_report_metric_subset(fixture):
    path, config, _ = fixture
    run = config["runs"][0]
    selected = {"all": ["coarse_mae", "fine_rmse_macro", "cv"], "h1": ["coarse_rmse"], "h3": ["fine_mae_macro"]}
    run["evaluation"]["metrics"] = selected
    run["golden"]["metrics"] = {key: value for key, value in run["golden"]["metrics"].items() if key.split("/")[1] in selected[key.split("/")[0]]}
    run["golden"]["identity"] = {"dataset": "dataset", "seed": "seed"}
    audit = PACSRAdapter().evaluate(run, base_dir=path.parent)
    results, _ = compare_golden(audit["rows"], run["golden"], base_dir=path.parent, run=run, sources=audit["sources"])
    assert len(results) == 5 and all(r["comparison"] == "pass" for r in results)
    assert all(r["identity_fields_unverified"] == "model;split;variant" for r in results)


def test_unknown_mask_and_operator_rejected():
    a = np.ones((2, 1, 1, 3))
    with pytest.raises(ValueError, match="mask"):
        calculate(a, a, a * .5)
    with pytest.raises(ValueError, match="Unsupported"):
        aggregate_components(a, {"kind": "guess", "dtype": "float64"})


def test_source_change_aborts_atomic_export(fixture, tmp_path, monkeypatch):
    from researchbench.export.pacsr import export_pacsr
    import researchbench.export.pacsr as export
    path, config, _ = fixture
    original = export.summarize_pacsr
    def mutate(rows):
        result = original(rows)
        Path(config["runs"][0]["input"]["file"]["path"]).write_bytes(b"changed")
        return result
    monkeypatch.setattr(export, "summarize_pacsr", mutate)
    output = tmp_path / "aborted"
    with pytest.raises(ValueError, match="Source changed"):
        export_pacsr(path, output)
    assert not output.exists() and not list(tmp_path.glob(".pacsr-*"))
