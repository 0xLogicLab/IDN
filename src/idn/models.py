from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 160
    temperature: float = 0.0
    top_p: float = 1.0


class ChatModel:
    def __init__(
        self,
        model_name: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_memory: dict[int | str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        dtype = self._dtype(torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _dtype(name: str) -> torch.dtype | str:
        mapping = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(name.lower(), torch.bfloat16)

    def format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    @torch.inference_mode()
    def generate(self, prompt: str, config: GenerationConfig) -> dict[str, Any]:
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = config.temperature > 0
        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=do_sample,
            temperature=config.temperature if do_sample else None,
            top_p=config.top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "text": text,
            "latency_ms": elapsed_ms,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": int(generated.shape[-1]),
        }
