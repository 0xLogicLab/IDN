"""Local chat model loading and greedy generation."""

from __future__ import annotations

import json
import os
from threading import Thread
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ModelLoadError(RuntimeError):
    """Raised when local model loading or generation cannot proceed."""


def validate_dtype(dtype: str) -> str:
    if dtype in ("float16", "fp16"):
        raise ModelLoadError("fp16 is forbidden; use bfloat16 or float32")
    if dtype in ("bfloat16", "bf16"):
        return "bfloat16"
    if dtype in ("float32", "fp32"):
        return "float32"
    raise ModelLoadError("Unsupported dtype: {0}".format(dtype))


def torch_dtype(dtype: str):
    dtype = validate_dtype(dtype)
    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError("torch is required for model loading") from exc
    return torch.bfloat16 if dtype == "bfloat16" else torch.float32


def load_local_qwen2_for_causal_lm(model_path: str, **kwargs):
    """Load Qwen2 through IDN's local editable backend."""

    try:
        from idn.models.qwen2_local.loader import load_qwen2_for_causal_lm
    except ImportError as exc:
        raise ModelLoadError("local Qwen2 backend is unavailable") from exc
    return load_qwen2_for_causal_lm(model_path, **kwargs)


@dataclass
class ChatModel:
    model_path: str
    adapter_path: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        validate_dtype(self.dtype)
        self.processor = None
        self.tokenizer = None
        self.model = None

    def _model_type(self) -> Optional[str]:
        config_path = Path(self.model_path) / "config.json"
        if not config_path.exists():
            return None
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            return None
        model_type = config.get("model_type")
        return str(model_type) if isinstance(model_type, str) else None

    def _load_causal_lm(self, auto_model_class, model_type, model_kwargs):
        """Dispatch by architecture; checkpoint size does not affect this choice.

        Hugging Face identifies Qwen2.5 checkpoints as ``model_type=qwen2``,
        so all Qwen2.5 sizes use IDN's editable Qwen2 implementation. Other
        text causal-LM architectures use Transformers' AutoModel mapping.
        """
        if model_type == "qwen2":
            return load_local_qwen2_for_causal_lm(self.model_path, **model_kwargs)
        return auto_model_class.from_pretrained(self.model_path, **model_kwargs)

    def load(self) -> "ChatModel":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            import transformers
        except ImportError as exc:
            raise ModelLoadError(
                "torch and transformers are required for model loading"
            ) from exc

        dtype = torch_dtype(self.dtype)
        kwargs = {
            "torch_dtype": dtype,
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
        }
        if self.device == "cuda" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": "cpu"}
            if dtype == torch.bfloat16:
                kwargs["torch_dtype"] = torch.float32

        model_type = self._model_type()
        tokenizer = None
        if model_type == "qwen3_5":
            version = getattr(transformers, "__version__", "unknown")
            image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
            if "qwen3_5" not in CONFIG_MAPPING:
                raise ModelLoadError(
                    "Qwen3.5 checkpoints need a newer transformers release than {0}. "
                    "Upgrade the control env transformers to a build that includes "
                    "qwen3_5/Qwen3_5ForConditionalGeneration support.".format(version)
                )
            if image_text_cls is None:
                raise ModelLoadError(
                    "Qwen3.5 checkpoints need a newer transformers release than {0}. "
                    "Upgrade transformers to a build with AutoModelForImageTextToText.".format(
                        version
                    )
                )
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
            try:
                model = image_text_cls.from_pretrained(
                    self.model_path,
                    **kwargs,
                )
            except ValueError as exc:
                raise ModelLoadError(
                    "Qwen3.5 is not supported by the installed transformers "
                    "version ({0}). Upgrade to a release that includes "
                    "qwen3_5/Qwen3_5ForConditionalGeneration support.".format(
                        version
                    )
                ) from exc
            self.processor = processor
            self.tokenizer = getattr(processor, "tokenizer", None)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            model = self._load_causal_lm(
                AutoModelForCausalLM,
                model_type,
                kwargs,
            )
            self.tokenizer = tokenizer

        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ModelLoadError("peft is required to load LoRA adapters") from exc
            model = PeftModel.from_pretrained(
                model,
                self.adapter_path,
                local_files_only=self.local_files_only,
            )

        model.eval()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.model = model
        return self

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()

        try:
            import torch
        except ImportError as exc:
            raise ModelLoadError("torch is required for generation") from exc

        prompt = self.render_prompt(messages)
        inputs = self._encode_prompt(prompt)
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][prompt_tokens:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_new_tokens: int,
    ) -> List[str]:
        """Greedily generate a response for each left-padded chat prompt."""

        if not messages_batch:
            return []
        if self.model is None or self.tokenizer is None:
            self.load()
        try:
            import torch
        except ImportError as exc:
            raise ModelLoadError("torch is required for generation") from exc
        prompts = [self.render_prompt(messages) for messages in messages_batch]
        if self.processor is not None:
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        else:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }
        prompt_width = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return [
            self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
            for row in output_ids
        ]

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
    ) -> Iterator[str]:
        if self.model is None or self.tokenizer is None:
            self.load()

        try:
            import torch
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise ModelLoadError(
                "torch and transformers are required for streaming generation"
            ) from exc

        prompt = self.render_prompt(messages)
        inputs = self._encode_prompt(prompt)
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        def _generate() -> None:
            with torch.no_grad():
                self.model.generate(**generation_kwargs)

        thread = Thread(target=_generate)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()

    def render_prompt(self, messages: List[Dict[str, str]]) -> str:
        renderer = self.processor or self.tokenizer
        if renderer is not None and hasattr(renderer, "apply_chat_template"):
            return renderer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(
            "{0}: {1}".format(message["role"], message["content"])
            for message in messages
        ) + "\nassistant:"

    def _encode_prompt(self, prompt: str):
        if self.processor is not None:
            return self.processor(text=prompt, return_tensors="pt")
        if self.tokenizer is None:
            raise ModelLoadError("Tokenizer is not loaded")
        return self.tokenizer(prompt, return_tensors="pt")
