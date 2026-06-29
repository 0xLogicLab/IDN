# IDN Capability Extraction

This repository implements the V1 experiment pipeline described in
`CAPABILITY_EXTRACTION_PLAN.md`.

Phase 0 provides the package skeleton, YAML config loading and machine overrides,
the capability registry, and a CLI entry point.

```bash
python -m idn.cli --help
python -m idn.cli list-capabilities
python -m pytest
```

All model and dataset paths are configured locally. The pipeline is designed to
load models and datasets from disk, not from the Hugging Face Hub.

