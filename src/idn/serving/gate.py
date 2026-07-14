"""Scope-based routing for healed capability models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from idn.capabilities.base import CapabilitySpec
from idn.extraction.certify import scope_mahalanobis
from idn.extraction.heal import render_prompt, write_json_atomic
from idn.models.loader import torch_dtype
from idn.models.qwen2_local.loader import load_qwen2_for_causal_lm


class GateError(RuntimeError):
    """Raised when scope routing cannot be performed."""


@dataclass(frozen=True)
class GateDecision:
    distance: float
    threshold: float
    in_scope: bool
    route: str


@dataclass(frozen=True)
class RoutedResponse:
    text: str
    decision: GateDecision
    normalized_entropy: float


@dataclass(frozen=True)
class ScopeGate:
    center: Any
    components: Any
    inverse_covariance: Any
    threshold: float

    @classmethod
    def from_npz(
        cls,
        path: Path,
        threshold: Optional[float] = None,
    ) -> "ScopeGate":
        try:
            import numpy as np
        except ImportError as exc:
            raise GateError("numpy is required for scope routing") from exc
        path = Path(path)
        if not path.exists():
            raise GateError("scope artifact does not exist: {0}".format(path))
        with np.load(path) as payload:
            stored_threshold = float(payload["threshold"][0])
            return cls(
                center=payload["mean"].copy(),
                components=payload["components"].copy(),
                inverse_covariance=payload["inverse_covariance"].copy(),
                threshold=float(
                    stored_threshold if threshold is None else threshold
                ),
            )

    def decide(self, vector: Any) -> GateDecision:
        distance = scope_mahalanobis(
            vector=vector,
            center=self.center,
            components=self.components,
            inverse_covariance=self.inverse_covariance,
        )
        in_scope = distance <= self.threshold
        return GateDecision(
            distance=distance,
            threshold=self.threshold,
            in_scope=in_scope,
            route="student" if in_scope else "teacher",
        )


class ScopeRoutedChatModel:
    """Reference runtime that routes between a local carve and teacher fallback."""

    def __init__(
        self,
        *,
        carved_model_path: Path,
        adapter_path: Path,
        teacher_model_path: Path,
        scope_path: Path,
        scope_layer: int,
        max_length: int = 512,
        dtype: str = "bfloat16",
        device: str = "cuda",
        local_files_only: bool = True,
        threshold: Optional[float] = None,
    ) -> None:
        self.carved_model_path = Path(carved_model_path)
        self.adapter_path = Path(adapter_path)
        self.teacher_model_path = Path(teacher_model_path)
        self.scope_path = Path(scope_path)
        self.scope_layer = int(scope_layer)
        self.max_length = int(max_length)
        self.dtype = dtype
        self.device_name = device
        self.local_files_only = local_files_only
        self.threshold = threshold
        self.student = None
        self.teacher = None
        self.tokenizer = None
        self.device = None
        self.gate = None

    def load(self) -> "ScopeRoutedChatModel":
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise GateError(
                "torch, transformers, and peft are required for routed generation"
            ) from exc
        if self.device_name.startswith("cuda"):
            if not torch.cuda.is_available():
                raise GateError("CUDA was requested but is unavailable")
            self.device = torch.device(self.device_name)
        elif self.device_name == "cpu":
            self.device = torch.device("cpu")
        else:
            raise GateError("unsupported routing device: {0}".format(self.device_name))
        resolved_dtype = torch_dtype(self.dtype)
        if self.device.type == "cpu" and resolved_dtype == torch.bfloat16:
            resolved_dtype = torch.float32
        load_kwargs = {
            "torch_dtype": resolved_dtype,
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
            "device_map": {"": str(self.device)},
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.teacher_model_path),
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        student_base = load_qwen2_for_causal_lm(
            str(self.carved_model_path),
            **load_kwargs,
        )
        self.student = PeftModel.from_pretrained(
            student_base,
            str(self.adapter_path),
            local_files_only=self.local_files_only,
            is_trainable=False,
        ).eval()
        self.teacher = load_qwen2_for_causal_lm(
            str(self.teacher_model_path),
            **load_kwargs,
        ).eval()
        self.gate = ScopeGate.from_npz(self.scope_path, threshold=self.threshold)
        return self

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
    ) -> RoutedResponse:
        if self.student is None or self.teacher is None or self.tokenizer is None:
            self.load()
        import torch
        import torch.nn.functional as functional

        prompt_budget = self.max_length - max_new_tokens
        if prompt_budget <= 0:
            raise GateError("max_length must exceed max_new_tokens")
        prompt = render_prompt(self.tokenizer, messages)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = list(prompt_ids)[-prompt_budget:]
        if not prompt_ids:
            raise GateError("encoded routing prompt is empty")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            gate_output = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        vector = gate_output.hidden_states[self.scope_layer][
            0,
            -1,
            :,
        ].float().cpu().numpy()
        decision = self.gate.decide(vector)
        probabilities = functional.softmax(gate_output.logits[0, -1].float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        normalized_entropy = float(entropy.cpu()) / math.log(probabilities.numel())
        model = self.student if decision.in_scope else self.teacher
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0, input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return RoutedResponse(
            text=text,
            decision=decision,
            normalized_entropy=normalized_entropy,
        )


def evaluate_gate_records(
    *,
    spec: CapabilitySpec,
    records: Sequence[Mapping[str, Any]],
    threshold: float,
) -> Mapping[str, Any]:
    if not records:
        raise GateError("gate evaluation requires probe records")
    routed_predictions = []
    references = []
    decisions = []
    local_count = 0
    rescued = 0
    harmful = 0
    unchanged = 0
    for record in records:
        distance = float(record["scope_mahalanobis"])
        in_scope = distance <= threshold
        route = "student" if in_scope else "teacher"
        student_prediction = record.get("student_prediction")
        teacher_prediction = record.get("teacher_prediction")
        reference = record.get("reference")
        routed_prediction = student_prediction if in_scope else teacher_prediction
        routed_predictions.append(routed_prediction)
        references.append(reference)
        if in_scope:
            local_count += 1
        else:
            student_correct = student_prediction == reference
            teacher_correct = teacher_prediction == reference
            if teacher_correct and not student_correct:
                rescued += 1
            elif student_correct and not teacher_correct:
                harmful += 1
            else:
                unchanged += 1
        decisions.append(
            {
                "id": record.get("id"),
                "distance": distance,
                "threshold": threshold,
                "route": route,
                "routed_prediction": routed_prediction,
                "reference": reference,
            }
        )
    total = len(records)
    fallback_count = total - local_count
    return {
        "status": "complete",
        "examples": total,
        "threshold": threshold,
        "local_coverage": local_count / total,
        "fallback_rate": fallback_count / total,
        "local_examples": local_count,
        "fallback_examples": fallback_count,
        "routed_metrics": spec.metric(routed_predictions, references),
        "fallback_effect": {
            "rescued_student_errors": rescued,
            "harmful_fallbacks": harmful,
            "unchanged_fallbacks": unchanged,
        },
        "decisions": decisions,
    }


def evaluate_certificate_gate(
    *,
    spec: CapabilitySpec,
    certificate_dir: Path,
    output_path: Path,
    threshold: Optional[float] = None,
) -> Mapping[str, Any]:
    certificate_dir = Path(certificate_dir)
    certificate = _read_json(certificate_dir / "certificate.json")
    records_payload = _read_json(certificate_dir / "probe_records.json")
    records = records_payload.get("records")
    if not isinstance(records, list):
        raise GateError("probe_records.json is missing a records list")
    resolved_threshold = float(
        certificate["scope"]["mahalanobis_threshold"]
        if threshold is None
        else threshold
    )
    result = dict(
        evaluate_gate_records(
            spec=spec,
            records=records,
            threshold=resolved_threshold,
        )
    )
    result["capability"] = spec.id
    result["certificate_path"] = str(certificate_dir / "certificate.json")
    result["student_metrics"] = certificate["quality"]["student_metrics"]
    result["teacher_metrics"] = certificate["quality"]["teacher_metrics"]
    result["scope_calibration_split"] = certificate["scope"].get(
        "calibration_split"
    )
    write_json_atomic(Path(output_path), result)
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise GateError("gate artifact does not exist: {0}".format(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError("could not read gate artifact: {0}".format(path)) from exc
    if not isinstance(payload, dict):
        raise GateError("gate artifact must contain a JSON object: {0}".format(path))
    return payload
