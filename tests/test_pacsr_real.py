"""Opt-in real-data acceptance. Never search, create or upload private caches."""
import os
from pathlib import Path
import pytest
from researchbench.adapters.pacsr import load_audit
from researchbench.export.pacsr import export_pacsr


def test_real_pacsr_golden_reference(tmp_path):
    configured = os.environ.get("RESEARCHBENCH_PACSR_REAL_CONFIG")
    if not configured:
        pytest.skip("Unverified: real PACSR cache and original reference not supplied")
    path = Path(configured)
    config, _, audits = load_audit(path)
    assert all(run["data_kind"] == "real" for run in config["runs"])
    assert all(set(run["golden"]["identity"]) == {"dataset", "model", "variant", "seed", "split"} for run in config["runs"]), "Full original identity evidence required for acceptance"
    assert all("Unverified" not in str(audit["protocol"]) for audit in audits)
    passed, count = export_pacsr(path, tmp_path / "real-audit")
    assert count >= 1 and passed
