"""Loading helpers for IDN's local Qwen2 backend."""

from __future__ import annotations

from typing import Any

from idn.models.qwen2_local import Qwen2Config, Qwen2ForCausalLM


def load_qwen2_for_causal_lm(model_path: str, **kwargs: Any) -> Qwen2ForCausalLM:
    """Load a Qwen2 causal LM using IDN's local model classes."""

    config = Qwen2Config.from_pretrained(
        model_path,
        local_files_only=kwargs.get("local_files_only", True),
        trust_remote_code=kwargs.get("trust_remote_code", True),
    )
    return Qwen2ForCausalLM.from_pretrained(
        model_path,
        config=config,
        **kwargs,
    )
