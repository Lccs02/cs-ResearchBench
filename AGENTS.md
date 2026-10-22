# ResearchBench development rules

- Keep this repository independent from all existing research projects. Do not retrain.
- Read docs/architecture.md and README.md before changing statistical contracts.
- Keep metric computation in metrics/, statistical aggregation in analysis/.
- Skills must call the CLI; never reimplement metrics or fabricate results in a prompt.
- Treat each training seed as the aggregation unit. Do not pool episodes, nodes,
  horizons, batches, or repeated checkpoints as independent seeds.
- Preserve source SHA-256, mappings, configs, data_kind and Unverified labels.
- Schema validation does not establish source authenticity or scientific correctness.
- Never infer private dataset formats. Require inspected samples and explicit mappings.
- Never commit private data, credentials, generated outputs, checkpoints or weights.
- Synthetic fixtures must use fixed seeds and must never be presented as paper results.
- Reject unknown schema versions, duplicate runs/seeds and incompatible paired comparisons.
- Before a commit run `python -m pytest -q` and the README end-to-end example.
- Do not change statistical semantics without tests, documentation and schema review.
