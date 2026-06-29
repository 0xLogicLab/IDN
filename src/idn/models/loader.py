"""Local chat model loading and greedy generation."""

from __future__ import annotations

import os
from threading import Thread
from dataclasses import dataclass
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


@dataclass
class ChatModel:
    model_path: str
    adapter_path: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        validate_dtype(self.dtype)
        self.tokenizer = None
        self.model = None

    def load(self) -> "ChatModel":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                "torch and transformers are required for model loading"
            ) from exc

        dtype = torch_dtype(self.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

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

        model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)

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
        inputs = self.tokenizer(prompt, return_tensors="pt")
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
        inputs = self.tokenizer(prompt, return_tensors="pt")
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
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(
            "{0}: {1}".format(message["role"], message["content"])
            for message in messages
        ) + "\nassistant:"
