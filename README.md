# IDN Capability Extraction

This repository implements the V1 experiment pipeline described in
`CAPABILITY_EXTRACTION_PLAN.md`.

Phase 0 provides the package skeleton, YAML config loading and machine overrides,
the capability registry, and a CLI entry point.

```bash
python -m idn.cli --help
python -m idn.cli list-capabilities
python -m idn.cli --machine h100 eval --config configs/capabilities/sst2.yaml --model Qwen2.5-7B-Instruct
python -m pytest
```

Machine configs are selected with `--machine 3090` or `--machine h100`. Pass model
and dataset directories as paths relative to the selected machine config's
`model_dir` and `data_dir`; the pipeline loads from disk, not from the Hugging
Face Hub.
