---
name: research-analysis
description: Use ResearchBench to ingest CSV/JSON/NPZ experiment outputs, validate provenance, summarize random seeds, compare ablations, and export tables or figures through its Python CLI.
---

1. Read the repository README and AGENTS.md. Run `researchbench --help`.
2. Inspect only the user-authorized files. Confirm project, dataset, model variant,
   seed, split, units, evaluation_id, array keys/columns and statistical unit.
   Keep unknown facts Unverified. Never guess PACSR, satellite or RL formats.
3. Create an explicit YAML mapping following configs/synthetic.yaml. For predictions,
   use same-shape target/prediction/weights. Default to missing=error. For reported
   RL metrics use scalar precomputed mode; never silently average steps or episodes.
4. Run `researchbench ingest CONFIG --output NEW_RECORDS.json`, then
   `researchbench validate NEW_RECORDS.json --check-sources`.
5. Run `researchbench summarize NEW_RECORDS.json --output NEW_OUTPUT_DIR`.
   Add `--baseline MODEL` only when evaluation protocols and seed sets match.
   Alternatively run `researchbench run CONFIG --output NEW_OUTPUT_DIR --baseline MODEL`.
6. Inspect summary.csv, ablations.csv (if requested) and manifest.json. Report exact
   commands, seed counts, missing-value policy, weights and Unverified status.
   If validation fails, explain the mismatch; do not drop runs or alter data to pass.
7. Do not calculate metrics independently, retrain, select favorable seeds,
   fabricate experimental outcomes, overwrite existing outputs, or upload private inputs.
   Label synthetic examples as demonstrations, never paper results.
