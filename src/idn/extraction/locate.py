"""Gradient x activation scoring for Qwen2 prunable units."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from idn.capabilities.base import CapabilitySpec, Example
from idn.extraction.units import FFN_CHANNEL, KV_HEAD_GROUP, QUERY_HEAD, qwen2_dimensions
from idn.models.loader import torch_dtype
from idn.models.qwen2_local.loader import load_qwen2_for_causal_lm


class LocateError(RuntimeError):
    """Raised when unit attribution cannot be computed."""


@dataclass(frozen=True)
class LocateResult:
    scores: Mapping[str, float]
    report: Mapping[str, Any]


class ActivationAttributor:
    """Collect grad x activation scores from selected Qwen2 module outputs."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.handles = []
        self.attn_scores: Dict[int, Any] = {}
        self.ffn_scores: Dict[int, Any] = {}

    def __enter__(self) -> "ActivationAttributor":
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_hook(
                    self._capture_input(self.attn_scores, layer_idx)
                )
            )
            self.handles.append(
                layer.mlp.down_proj.register_forward_hook(
                    self._capture_input(self.ffn_scores, layer_idx)
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def reset(self) -> None:
        self.attn_scores.clear()
        self.ffn_scores.clear()

    def scores_for_last_backward(self) -> Dict[str, float]:
        config = self.model.config
        dims = qwen2_dimensions(config)
        scores: Dict[str, float] = {}
        for layer_idx, activation in self.attn_scores.items():
            if activation.grad is None:
                continue
            values = gradxact_vector(activation, activation.grad)
            scores.update(attention_scores_to_units(layer_idx, values, dims))
        for layer_idx, activation in self.ffn_scores.items():
            if activation.grad is None:
                continue
            values = gradxact_vector(activation, activation.grad)
            scores.update(ffn_scores_to_units(layer_idx, values))
        return scores

    def _capture_input(self, store: Dict[int, Any], layer_idx: int) -> Callable:
        def hook(_module, inputs, _output):
            activation = inputs[0]
            activation.retain_grad()
            store[layer_idx] = activation
        return hook


def locate_qwen2_units(
    *,
    model_path: Path,
    spec: CapabilitySpec,
    split: str,
    limit: int,
    output_dir: Path,
    dtype: str,
    device: str,
    local_files_only: bool = True,
    max_length: int = 512,
    progress: Optional[Callable[[str], None]] = None,
) -> LocateResult:
    """Compute first-order unit scores and write locate artifacts."""

    if limit <= 0:
        raise LocateError("limit must be positive")

    _report(progress, "loading tokenizer")
    tokenizer = load_tokenizer(model_path, local_files_only=local_files_only)
    _report(progress, "loading local Qwen2 model")
    model = load_model(
        model_path=model_path,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )
    examples = spec.load_split(split=split, limit=limit)
    if not examples:
        raise LocateError("no examples loaded for split: {0}".format(split))

    aggregate: Dict[str, float] = {}
    completed = 0
    with ActivationAttributor(model) as attributor:
        for index, example in enumerate(examples, start=1):
            _report(progress, "scoring example {0}/{1}".format(index, len(examples)))
            record = score_example(
                model=model,
                tokenizer=tokenizer,
                spec=spec,
                example=example,
                attributor=attributor,
                max_length=max_length,
            )
            add_scores(aggregate, record)
            completed += 1

    normalized = {
        key: value / completed
        for key, value in sorted(aggregate.items())
    }
    report = {
        "method": "gradxact",
        "model_path": str(model_path),
        "capability": spec.id,
        "split": split,
        "examples": completed,
        "num_scores": len(normalized),
        "dtype": dtype,
        "device": device,
        "max_length": max_length,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_scores_json(output_dir / "scores.json", normalized, report)
    write_report_json(output_dir / "report.json", report)
    return LocateResult(scores=normalized, report=report)


def score_example(
    *,
    model: Any,
    tokenizer: Any,
    spec: CapabilitySpec,
    example: Example,
    attributor: ActivationAttributor,
    max_length: int,
) -> Mapping[str, float]:
    import torch

    prompt = render_prompt(tokenizer, spec.format_prompt(example))
    target = spec.target_text(example)
    encoded = encode_prompt_and_target(
        tokenizer=tokenizer,
        prompt=prompt,
        target=target,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(model.device)
    labels = encoded["labels"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    model.zero_grad(set_to_none=True)
    attributor.reset()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    if output.loss is None:
        raise LocateError("model did not return a loss")
    output.loss.backward()
    scores = attributor.scores_for_last_backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache() if input_ids.device.type == "cuda" else None
    return scores


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(
        "{0}: {1}".format(message["role"], message["content"])
        for message in messages
    ) + "\nassistant:"


def encode_prompt_and_target(
    *,
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
) -> Mapping[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        target_ids = list(target_ids) + [int(eos_token_id)]
    input_ids = list(prompt_ids) + list(target_ids)
    labels = [-100] * len(prompt_ids) + list(target_ids)
    if max_length and len(input_ids) > max_length:
        trim = len(input_ids) - max_length
        input_ids = input_ids[trim:]
        labels = labels[trim:]
    if all(label == -100 for label in labels):
        raise LocateError("no target labels remain after encoding")
    import torch

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
    }


def load_tokenizer(model_path: Path, local_files_only: bool = True) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise LocateError("transformers is required for tokenization") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    *,
    model_path: Path,
    dtype: str,
    device: str,
    local_files_only: bool,
) -> Any:
    import torch

    resolved_dtype = torch_dtype(dtype)
    kwargs = {
        "torch_dtype": resolved_dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    if device == "cuda" and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": "cpu"}
        if resolved_dtype == torch.bfloat16:
            kwargs["torch_dtype"] = torch.float32
    model = load_qwen2_for_causal_lm(str(model_path), **kwargs)
    model.eval()
    model.config.use_cache = False
    return model


def gradxact_vector(activation: Any, gradient: Any) -> Any:
    values = (activation.detach() * gradient.detach()).abs()
    reduce_dims = tuple(range(values.ndim - 1))
    return values.sum(dim=reduce_dims).cpu()


def attention_scores_to_units(layer: int, values: Any, dims: Any) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for head in range(dims.num_attention_heads):
        start = head * dims.head_dim
        stop = start + dims.head_dim
        scores["{0}:{1}:{2}".format(QUERY_HEAD, layer, head)] = float(
            values[start:stop].sum().item()
        )
    for kv_group in range(dims.num_key_value_heads):
        query_start = kv_group * dims.query_heads_per_kv
        query_stop = query_start + dims.query_heads_per_kv
        total = 0.0
        for head in range(query_start, query_stop):
            total += scores["{0}:{1}:{2}".format(QUERY_HEAD, layer, head)]
        scores["{0}:{1}:{2}".format(KV_HEAD_GROUP, layer, kv_group)] = total
    return scores


def ffn_scores_to_units(layer: int, values: Any) -> Dict[str, float]:
    return {
        "{0}:{1}:{2}".format(FFN_CHANNEL, layer, index): float(value)
        for index, value in enumerate(values.tolist())
    }


def add_scores(total: Dict[str, float], update: Mapping[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def write_scores_json(
    path: Path,
    scores: Mapping[str, float],
    report: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            {"scores": dict(scores), "report": dict(report)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_report_json(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
