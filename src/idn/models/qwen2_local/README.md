# IDN Local Qwen2 Backend

This package vendors the Transformers Qwen2 implementation so IDN can modify
model structure for carving without patching site-packages.

Current state:
- `configuration_qwen2.py` and `modeling_qwen2.py` are copied from the installed
  Transformers Qwen2 implementation.
- Imports are rewritten to use absolute `transformers.*` utility modules.
- `ChatModel` routes `model_type: qwen2` checkpoints through this backend.
- Qwen2.5 1.5B, 7B, 14B, and 72B checkpoints share this architecture route; each
  size still needs its own external weight directory and resource profile.
- Other supported text causal-LM architectures use Transformers' automatic
  model mapping. Architecture-specific extraction operations remain separate.

