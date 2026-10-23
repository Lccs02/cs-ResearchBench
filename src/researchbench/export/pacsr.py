"""Atomic audit bundles, including failed comparisons; never publish source files."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from researchbench.adapters.pacsr import VERSION, load_audit
from researchbench.analysis.pacsr import compare_golden, summarize_pacsr


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def export_pacsr(config_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    config, config_source, audits = load_audit(config_path)
    rows = [row for audit in audits for row in audit["rows"]]
    summaries = summarize_pacsr(rows)
    comparisons = []
    for run, audit in zip(config["runs"], audits):
        compared, source = compare_golden(audit["rows"], run["golden"], base_dir=Path(config_path).resolve().parent,
                                          run=run, sources=audit["sources"])
        comparisons.extend(compared)
        audit["sources"].append(source)
        ids = ";".join(sorted({s["sha256"] for s in audit["sources"]} | {config_source["sha256"]}))
        for row in audit["rows"]:
            row["source_ids"] = ids
        for row in compared:
            row["source_ids"] = ids
    passed = all(row["comparison"] == "pass" for row in comparisons)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pacsr-", dir=output.parent))
    try:
        for name, values in (("metrics", rows), ("summary", summaries), ("comparison", comparisons)):
            pd.DataFrame(values).to_csv(stage / f"{name}.csv", index=False, na_rep="Unverified")
        evidence = {"schema_version": VERSION, "verification": "Unverified", "numerical_match": passed,
                    "config": config, "config_source": config_source,
                    "audits": audits, "comparisons": comparisons, "summary": summaries}
        (stage / "audit.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lines = [r"% Unverified. Synthetic rows are software tests, not paper results.",
                 r"\begin{tabular}{llllllll}", r"\toprule",
                 r"Dataset & Model / variant & Horizon & Metric & Mean & Std & Seeds & Status \\", r"\midrule"]
        for row in summaries:
            numeric = lambda x: "Unverified" if x is None else f"{x:.8g}"
            values = [row["dataset"], row["model"] + " / " + row["variant"], row["horizon"], row["metric"],
                      numeric(row["mean"]), numeric(row["std"]), row["n_seeds"], row["data_kind"] + " / Unverified"]
            lines.append(" & ".join(tex_escape(v) for v in values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (stage / "summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Separate protocols/units/data kinds. Never plot missing values as zero.
        frame = pd.DataFrame(summaries)
        plots = []
        keys = ["dataset", "evaluation_id", "protocol_id", "data_kind", "metric", "unit"]
        for index, (identity, group) in enumerate(frame.groupby(keys, sort=True)):
            fig, ax = plt.subplots(figsize=(7, 4))
            try:
                valid = group[group["mean"].notna()].copy()
                labels = [f"{r.model}/{r.variant}/{r.horizon}" for r in valid.itertuples()]
                ax.scatter(range(len(valid)), valid["mean"], label="Seed mean")
                for pos, row in enumerate(valid.itertuples()):
                    if pd.notna(row.std):
                        ax.errorbar(pos, row.mean, yerr=row.std, capsize=3, color="C0")
                ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
                ax.set_ylabel(f"{identity[4]} ({identity[5]})")
                ax.set_title(f"{identity[0]} — {identity[3]} / Unverified")
                fig.text(.02, .01, "Equal training seeds; bars = sample std (ddof=1); singleton std Unverified", fontsize=7)
                fig.tight_layout(rect=(0, .05, 1, 1))
                files = [f"figure_{index:03d}.{ext}" for ext in ("svg", "pdf")]
                for name in files:
                    fig.savefig(stage / name)
                plots.append({"group": dict(zip(keys, identity)), "files": files})
            finally:
                plt.close(fig)
        # Detect sources changed while the report was being computed.
        for source in [config_source, *[s for audit in audits for s in audit["sources"]]]:
            if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Source changed during audit")
        manifest = {"schema_version": VERSION, "verification": "Unverified", "numerical_match": passed,
                    "real_cache_acceptance": "Unverified", "plots": plots,
                    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib", "PyYAML", "researchbench")},
                    "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.iterdir())}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError("Output appeared during export")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return passed, len(config["runs"])
