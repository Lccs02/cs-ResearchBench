import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest
from researchbench.cli import main
from researchbench.export.bundle import export_bundle, tex_escape


def cli(*args):
    return subprocess.run([sys.executable, "-m", "researchbench", *map(str, args)], capture_output=True, text=True)


def test_cli_end_to_end(generated, tmp_path):
    output = tmp_path / "report"
    result = cli("run", generated, "--output", output, "--baseline", "baseline")
    assert result.returncode == 0, result.stderr
    assert "Unverified" in result.stdout
    result = cli("validate", output / "records.json", "--check-sources")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    for name, sha in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == sha
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 4 and (summary.n_seeds == 3).all()
    assert summary.verification.eq("Unverified").all()
    assert (output / "summary.tex").read_text().count("\\begin{tabular}") == 2
    assert len(list(output.glob("*.png"))) == 2
    assert all(p.stat().st_size > 1000 for p in output.glob("*.png"))
    second = tmp_path / "second"
    assert cli("summarize", output / "records.json", "--output", second, "--baseline", "baseline").returncode == 0
    assert (output / "summary.csv").read_bytes() == (second / "summary.csv").read_bytes()
    assert cli("run", generated, "--output", output).returncode == 2


def test_ingest_and_tamper_detection(generated, tmp_path):
    records = tmp_path / "records.json"
    assert main(["ingest", str(generated), "--output", str(records)]) == 0
    assert main(["validate", str(records), "--check-sources"]) == 0
    generated.write_text(generated.read_text() + "\n# modified\n")
    assert main(["validate", str(records), "--check-sources"]) == 2
    assert main(["validate", str(records)]) == 0


def test_failed_comparison_leaves_no_report(records, tmp_path):
    out = tmp_path / "bad-report"
    with pytest.raises(ValueError):
        export_bundle(records[:-1], out, baseline="baseline")
    assert not out.exists()


def test_tex_escape():
    assert tex_escape("a_b%&") == r"a\_b\%\&"
    assert tex_escape("\\x{}") == r"\textbackslash{}x\{\}"


def test_malformed_yaml_is_clean_cli_error(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text('runs: [\n')
    assert main(["run", str(path), "--output", str(tmp_path / 'bad')]) == 2
    assert "Traceback" not in capsys.readouterr().err
