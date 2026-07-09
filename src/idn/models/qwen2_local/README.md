# IDN Local Qwen2 Backend

This package vendors the Transformers Qwen2 implementation so IDN can modify
model structure for carving without patching site-packages.

Current state:
- `configuration_qwen2.py` and `modeling_qwen2.py` are copied from the installed
  Transformers Qwen2 implementation.
- Imports are rewritten to use absolute `transformers.*` utility modules.
- `ChatModel` routes `model_type: qwen2` checkpoints through this backend.

