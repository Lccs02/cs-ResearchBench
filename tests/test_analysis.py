from copy import deepcopy
import numpy as np
import pytest
from researchbench.analysis.summary import summarize, ablations


def test_equal_seed_weighting_and_ddof(records):
    group = [r for r in records if r.model == "baseline"]
    for r, value in zip(group, [1, 3, 8]):
        r.metrics["MAE"]["value"] = value
    # Strongly different sample counts must not change seed weighting.
    group[0].metrics["MAE"].update(n_total=1000, n_valid=1000, weight_sum=1000)
    row = summarize(group).query("metric == 'MAE'").iloc[0]
    assert row["mean"] == 4
    assert row["std"] == pytest.approx(np.std([1, 3, 8], ddof=1))
    assert row.n_seeds == 3


def test_single_seed_std_is_undefined(records):
    out = summarize(records[:1])
    assert out["std"].isna().all()
    assert out.std_status.str.startswith("Unverified").all()


def test_paired_ablation(records):
    out = ablations(records, "baseline")
    assert (out.n_pairs == 3).all()
    assert (out.delta_mean < 0).all()
    assert np.allclose(out.improvement_pct, 37.5)
    base = [r.metrics["MAE"]["value"] for r in records if r.model == "baseline"]
    variant = [r.metrics["MAE"]["value"] for r in records if r.model == "variant"]
    assert out.query("metric == 'MAE'").iloc[0].delta_std == pytest.approx(np.std(np.array(variant) - base, ddof=1))


def test_zero_baseline(records):
    for r in records:
        if r.model == "baseline":
            r.metrics["MAE"]["value"] = 0
    row = ablations(records, "baseline").query("metric == 'MAE'").iloc[0]
    assert np.isnan(row.improvement_pct)
    assert row.improvement_status.startswith("Unverified")


@pytest.mark.parametrize("change", ["duplicate", "config", "missing_seed", "protocol", "baseline"])
def test_incompatible_runs_rejected(records, change):
    if change == "duplicate":
        clone = deepcopy(records[0]); clone.run_id = "new-id"; records.append(clone)
    elif change == "config":
        records[0].config["lr"] = 1
    elif change == "missing_seed":
        records.pop()
    elif change == "protocol":
        records[-1].protocol["statistical_unit"] = "episode"
    with pytest.raises(ValueError):
        ablations(records, "missing" if change == "baseline" else "baseline")


def test_maximize_negative_baseline(records):
    for r in records:
        r.protocol = {"mode": "precomputed", "missing": "not_applicable", "weighting": "reported", "reduction": "reported_scalar", "statistical_unit": "mean evaluation return per seed", "metric_units": {"return": "reward"}}
        r.metrics = {"return": {"value": -10 if r.model == "baseline" else -5, "direction": "max"}}
    out = ablations(records, "baseline").iloc[0]
    assert out.improvement_pct == 50
    assert out.delta_mean == 5


def test_split_groups_are_not_pooled(records):
    clone = deepcopy(records[0])
    clone.run_id = "validation-run"
    clone.split = "validation"
    out = summarize(records + [clone])
    assert out.query("split == 'validation'").n_seeds.eq(1).all()
    assert out.query("split == 'test'").n_seeds.eq(3).all()
