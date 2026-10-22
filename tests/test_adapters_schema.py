from copy import deepcopy
import json
import numpy as np
import pytest
import yaml
from researchbench.adapters.files import ingest_config, read_arrays
from researchbench.adapters.base import PACSRAdapter, SatelliteMultiServiceAdapter, WorldModelRLAdapter
from researchbench.core.schema import ExperimentRecord, read_records, write_records


def test_formats_and_hashes(records):
    assert len(records) == 6
    assert {r.sources[0]["format"] for r in records} == {"csv", "json", "npz"}
    assert all(r.verification == "Unverified" and r.data_kind == "synthetic" for r in records)
    assert all(len(r.sources[0]["sha256"]) == 64 for r in records)


def test_schema_roundtrip(records, tmp_path):
    path = tmp_path / "records.json"
    write_records(records, path)
    assert [r.to_dict() for r in read_records(path)] == [r.to_dict() for r in records]


@pytest.mark.parametrize("field,value", [("schema_version", "2.0"), ("seed", True), ("seed", -1), ("verification", "Verified"), ("project", ""), ("config", [])])
def test_schema_rejects_invalid(records, field, value):
    obj = records[0].to_dict(); obj[field] = value
    with pytest.raises(ValueError):
        ExperimentRecord.from_dict(obj)


def test_nonfinite_and_unknown_fields(records):
    obj = records[0].to_dict(); obj["metrics"]["MAE"]["value"] = float("nan")
    with pytest.raises(ValueError):
        ExperimentRecord.from_dict(obj)
    obj = records[0].to_dict(); obj["not_a_field"] = 0
    with pytest.raises(ValueError):
        ExperimentRecord.from_dict(obj)


def test_csv_lists_and_json_nested_selection():
    a = read_arrays(b"a,b\n1,2\n3,4\n", "csv", {"x": ["b", "a"]})["x"]
    assert a.tolist() == [[2, 1], [4, 3]]
    a = read_arrays(b'{"root":[{"x":[1,2]}]}', "json", {"x": ["root", 0, "x"]})["x"]
    assert a.tolist() == [1, 2]


def test_precomputed_rl_scalar(generated):
    obj = yaml.safe_load(generated.read_text())
    r = obj["runs"][0]
    r["input"] = {"path": "rl.json", "format": "json", "mapping": {"return": ["eval", "return"]}}
    r["evaluation"] = {"mode": "precomputed", "statistical_unit": "per-seed mean over 10 fixed evaluation episodes", "units": {"return": "reward"}, "directions": {"return": "max"}}
    obj["runs"] = [r]
    path = generated.parent / "rl.json"
    path.write_text(json.dumps({"eval": {"return": 12.5}}))
    generated.write_text(yaml.safe_dump(obj))
    result = ingest_config(generated)[0]
    assert result.metrics["return"] == {"value": 12.5, "direction": "max"}
    path.write_text(json.dumps({"eval": {"return": [1, 2]}}))
    with pytest.raises(ValueError, match="scalar"):
        ingest_config(generated)


def test_shape_mismatch_and_config_typo(generated):
    obj = yaml.safe_load(generated.read_text())
    obj["runs"][0]["input"]["expected_shape"] = [999]
    generated.write_text(yaml.safe_dump(obj))
    with pytest.raises(ValueError, match="expected_shape"):
        ingest_config(generated)
    obj["runs"][0]["input"]["expected_shape"] = [24]
    obj["runs"][0]["evaluation"]["missng"] = "omit"
    generated.write_text(yaml.safe_dump(obj))
    with pytest.raises(ValueError, match="unknown"):
        ingest_config(generated)


def test_npz_objects_are_not_unpickled(tmp_path):
    path = tmp_path / "objects.npz"
    np.savez(path, x=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match="allow_pickle"):
        read_arrays(path.read_bytes(), "npz", {"x": "x"})


@pytest.mark.parametrize("adapter", [PACSRAdapter, SatelliteMultiServiceAdapter, WorldModelRLAdapter])
def test_reserved_adapters_do_not_invent_formats(adapter, tmp_path):
    with pytest.raises(NotImplementedError, match="Unverified"):
        adapter().read({}, base_dir=tmp_path, config_source={})
