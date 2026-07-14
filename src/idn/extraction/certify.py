"""Fidelity certification for healed Qwen2 carve artifacts."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from idn.capabilities.base import CapabilitySpec
from idn.extraction.heal import hidden_state_indices, render_prompt, write_json_atomic
from idn.models.loader import torch_dtype
from idn.models.qwen2_local.loader import load_qwen2_for_causal_lm


class CertifyError(RuntimeError):
    """Raised when a healed carve cannot be certified."""


@dataclass(frozen=True)
class CertifyResult:
    certificate_path: Path
    scope_path: Path
    records_path: Path
    certificate: Mapping[str, Any]


def certify_qwen2(
    *,
    carved_model_path: Path,
    adapter_path: Path,
    teacher_model_path: Path,
    spec: CapabilitySpec,
    split: str,
    limit: Optional[int],
    output_dir: Path,
    epsilon: float,
    max_length: int = 512,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    hidden_layer_stride: int = 4,
    scope_layer: int = 4,
    scope_components: int = 32,
    scope_regularization: float = 1.0e-4,
    scope_percentile: float = 95.0,
    scope_split: Optional[str] = None,
    scope_limit: Optional[int] = 300,
    dtype: str = "bfloat16",
    device: str = "cuda",
    local_files_only: bool = True,
    override: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> CertifyResult:
    """Compare a healed carve to its teacher and write certificate artifacts."""

    _validate_args(
        limit=limit,
        epsilon=epsilon,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        hidden_layer_stride=hidden_layer_stride,
        scope_layer=scope_layer,
        scope_components=scope_components,
        scope_regularization=scope_regularization,
        scope_percentile=scope_percentile,
        scope_limit=scope_limit,
    )
    carved_model_path = Path(carved_model_path)
    adapter_path = Path(adapter_path)
    teacher_model_path = Path(teacher_model_path)
    output_dir = Path(output_dir)
    _validate_directory(carved_model_path, "carved model")
    _validate_directory(adapter_path, "adapter")
    _validate_directory(teacher_model_path, "teacher model")
    _prepare_output_dir(output_dir, override=override)

    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from peft import PeftModel
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise CertifyError(
            "numpy, torch, transformers, and peft are required for certification"
        ) from exc

    target_device = _resolve_device(torch, device)
    resolved_dtype = torch_dtype(dtype)
    if target_device.type == "cpu" and resolved_dtype == torch.bfloat16:
        resolved_dtype = torch.float32
    load_kwargs = {
        "torch_dtype": resolved_dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "device_map": {"": str(target_device)},
    }

    _report(progress, "loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_model_path),
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _report(progress, "loading teacher")
    teacher = load_qwen2_for_causal_lm(str(teacher_model_path), **load_kwargs)
    teacher.eval()
    teacher.requires_grad_(False)

    _report(progress, "loading healed student")
    student_base = load_qwen2_for_causal_lm(str(carved_model_path), **load_kwargs)
    student = PeftModel.from_pretrained(
        student_base,
        str(adapter_path),
        local_files_only=local_files_only,
        is_trainable=False,
    )
    student.eval()
    student.requires_grad_(False)

    teacher_config = teacher.config
    student_config = student.base_model.model.config
    if int(teacher_config.vocab_size) != int(student_config.vocab_size):
        raise CertifyError("teacher and student vocabularies do not match")
    if int(teacher_config.hidden_size) != int(student_config.hidden_size):
        raise CertifyError("teacher and student residual widths do not match")
    num_hidden_layers = int(student_config.num_hidden_layers)
    if scope_layer > num_hidden_layers:
        raise CertifyError(
            "scope layer {0} exceeds student layer count {1}".format(
                scope_layer,
                num_hidden_layers,
            )
        )
    faithfulness_indices = hidden_state_indices(
        num_hidden_layers=num_hidden_layers,
        stride=hidden_layer_stride,
    )

    examples = spec.load_split(split=split, limit=limit)
    if not examples:
        raise CertifyError("no probe examples loaded for split: {0}".format(split))
    prediction_records = []
    teacher_predictions = []
    student_predictions = []
    references = []
    output_kls = []
    hidden_cosines: Dict[int, List[float]] = {
        index: [] for index in faithfulness_indices
    }
    probe_scope_vectors = []
    prompt_budget = max_length - max_new_tokens
    started_at = _utc_now()

    for probe_index, example in enumerate(examples, start=1):
        if probe_index == 1 or probe_index % 10 == 0 or probe_index == len(examples):
            _report(progress, "probe {0}/{1}".format(probe_index, len(examples)))
        prompt = render_prompt(tokenizer, spec.format_prompt(example))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = list(prompt_ids)[-prompt_budget:]
        if not prompt_ids:
            raise CertifyError("probe example has an empty encoded prompt: {0}".format(example.id))
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            teacher_ids = teacher.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            student_ids = student.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = int(input_ids.shape[-1])
        teacher_generated = teacher_ids[0, prompt_length:]
        student_generated = student_ids[0, prompt_length:]
        teacher_text = tokenizer.decode(
            teacher_generated,
            skip_special_tokens=True,
        ).strip()
        student_text = tokenizer.decode(
            student_generated,
            skip_special_tokens=True,
        ).strip()
        teacher_parsed = spec.parse_prediction(teacher_text)
        student_parsed = spec.parse_prediction(student_text)

        if teacher_ids.shape[-1] <= prompt_length:
            raise CertifyError(
                "teacher generated no output tokens for probe: {0}".format(example.id)
            )
        sequence_mask = torch.ones_like(teacher_ids)
        with torch.no_grad():
            teacher_output = teacher(
                input_ids=teacher_ids,
                attention_mask=sequence_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            student_output = student(
                input_ids=teacher_ids,
                attention_mask=sequence_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        prediction_start = prompt_length - 1
        prediction_stop = int(teacher_ids.shape[-1]) - 1
        teacher_logits = teacher_output.logits[
            :, prediction_start:prediction_stop, :
        ]
        student_logits = student_output.logits[
            :, prediction_start:prediction_stop, :
        ]
        output_kl = student_teacher_kl(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=temperature,
        )
        output_kl_value = float(output_kl.detach().cpu())
        output_kls.append(output_kl_value)

        record_layer_cosines = {}
        for hidden_index in faithfulness_indices:
            teacher_hidden = teacher_output.hidden_states[hidden_index][
                :, prediction_start:prediction_stop, :
            ].float()
            student_hidden = student_output.hidden_states[hidden_index][
                :, prediction_start:prediction_stop, :
            ].float()
            cosine = functional.cosine_similarity(
                student_hidden,
                teacher_hidden,
                dim=-1,
            ).mean()
            cosine_value = float(cosine.detach().cpu())
            hidden_cosines[hidden_index].append(cosine_value)
            record_layer_cosines[str(hidden_index)] = cosine_value

        scope_vector = student_output.hidden_states[scope_layer][
            0,
            prompt_length - 1,
            :,
        ].float().detach().cpu().numpy()
        probe_scope_vectors.append(scope_vector)
        teacher_predictions.append(teacher_parsed)
        student_predictions.append(student_parsed)
        references.append(example.reference)
        prediction_records.append(
            {
                "id": example.id,
                "reference": example.reference,
                "teacher_text": teacher_text,
                "teacher_prediction": teacher_parsed,
                "student_text": student_text,
                "student_prediction": student_parsed,
                "prediction_agreement": teacher_parsed == student_parsed,
                "teacher_valid": bool(spec.verify(teacher_text, teacher_parsed)),
                "student_valid": bool(spec.verify(student_text, student_parsed)),
                "student_teacher_kl": output_kl_value,
                "hidden_cosine_by_index": record_layer_cosines,
            }
        )

    teacher_metrics = spec.metric(teacher_predictions, references)
    student_metrics = spec.metric(student_predictions, references)
    teacher_metric = float(teacher_metrics[spec.metric_name])
    student_metric = float(student_metrics[spec.metric_name])
    metric_gap = (
        teacher_metric - student_metric
        if spec.higher_is_better
        else student_metric - teacher_metric
    )
    passed = metric_gap <= epsilon
    agreement_rate = mean(
        float(teacher_value == student_value)
        for teacher_value, student_value in zip(
            teacher_predictions,
            student_predictions,
        )
    )
    teacher_valid_rate = mean(
        float(record["teacher_valid"]) for record in prediction_records
    )
    student_valid_rate = mean(
        float(record["student_valid"]) for record in prediction_records
    )
    paired = paired_correctness(
        teacher_predictions=teacher_predictions,
        student_predictions=student_predictions,
        references=references,
    )

    calibration_split = scope_split or split
    calibration_examples = spec.load_split(
        split=calibration_split,
        limit=scope_limit,
    )
    if len(calibration_examples) < 2:
        raise CertifyError("scope calibration requires at least two examples")
    _report(
        progress,
        "calibrating scope on {0} {1} examples".format(
            len(calibration_examples),
            calibration_split,
        ),
    )
    calibration_vectors = collect_scope_vectors(
        student=student,
        tokenizer=tokenizer,
        spec=spec,
        examples=calibration_examples,
        max_prompt_tokens=prompt_budget,
        scope_layer=scope_layer,
        device=target_device,
        progress=progress,
    )
    scope = build_scope_model(
        vectors=np.stack(calibration_vectors),
        components=scope_components,
        regularization=scope_regularization,
        percentile=scope_percentile,
    )
    probe_distances = scope_distances(
        vectors=np.stack(probe_scope_vectors),
        center=scope["mean"],
        components=scope["components"],
        inverse_covariance=scope["inverse_covariance"],
    )
    for record, distance in zip(prediction_records, probe_distances):
        record["scope_mahalanobis"] = float(distance)

    scope_path = output_dir / "scope.npz"
    write_scope_npz(scope_path, scope)
    records_path = output_dir / "probe_records.json"
    write_json_atomic(records_path, {"records": prediction_records})
    per_layer_cosine = {
        str(index): mean(values) for index, values in hidden_cosines.items()
    }
    certificate = {
        "artifact_version": 1,
        "status": "pass" if passed else "fail",
        "capability": spec.id,
        "split": split,
        "probe_examples": len(examples),
        "quality": {
            "metric_name": spec.metric_name,
            "higher_is_better": spec.higher_is_better,
            "teacher_metrics": teacher_metrics,
            "student_metrics": student_metrics,
            "metric_gap": metric_gap,
            "epsilon": epsilon,
            "passed": passed,
        },
        "agreement": {
            "prediction_agreement": agreement_rate,
            "teacher_valid_output_rate": teacher_valid_rate,
            "student_valid_output_rate": student_valid_rate,
            "paired_correctness": paired,
        },
        "output_fidelity": {
            "direction": "KL(student || teacher)",
            "temperature": temperature,
            "mean_kl": mean(output_kls),
            "p95_kl": percentile(output_kls, 95.0),
        },
        "faithfulness": {
            "mean_hidden_cosine": mean(per_layer_cosine.values()),
            "hidden_cosine_by_index": per_layer_cosine,
        },
        "scope": {
            "artifact": scope_path.name,
            "hidden_state_index": scope_layer,
            "pooling": "last_prompt_token",
            "input_dimension": int(scope["mean"].shape[0]),
            "components": int(scope["components"].shape[0]),
            "explained_variance_ratio": float(scope["explained_variance_ratio"]),
            "regularization": scope_regularization,
            "distance_percentile": scope_percentile,
            "mahalanobis_threshold": float(scope["threshold"]),
            "calibration_split": calibration_split,
            "calibration_examples": len(calibration_examples),
            "calibration_false_positive_rate": float(
                mean(float(value > scope["threshold"]) for value in scope["distances"])
            ),
            "probe_out_of_scope_rate": float(
                mean(float(value > scope["threshold"]) for value in probe_distances)
            ),
        },
        "resources": {
            "carved_model_bytes": directory_size(carved_model_path),
            "adapter_bytes": directory_size(adapter_path),
        },
        "lineage": build_lineage(
            carved_model_path=carved_model_path,
            adapter_path=adapter_path,
            teacher_model_path=teacher_model_path,
        ),
        "artifacts": {
            "probe_records": records_path.name,
            "scope": scope_path.name,
        },
        "runtime": {
            "dtype": dtype,
            "device": str(target_device),
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
        },
        "created_at": started_at,
        "completed_at": _utc_now(),
    }
    certificate_path = output_dir / "certificate.json"
    write_json_atomic(certificate_path, certificate)
    return CertifyResult(
        certificate_path=certificate_path,
        scope_path=scope_path,
        records_path=records_path,
        certificate=certificate,
    )


def student_teacher_kl(
    *,
    student_logits: Any,
    teacher_logits: Any,
    temperature: float,
) -> Any:
    import torch.nn.functional as functional

    student_log = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_log = functional.log_softmax(teacher_logits / temperature, dim=-1)
    student_probabilities = student_log.exp()
    return (
        student_probabilities * (student_log - teacher_log)
    ).sum(dim=-1).mean() * (temperature ** 2)


def paired_correctness(
    *,
    teacher_predictions: Sequence[Any],
    student_predictions: Sequence[Any],
    references: Sequence[Any],
) -> Mapping[str, int]:
    result = {
        "both_correct": 0,
        "teacher_only_correct": 0,
        "student_only_correct": 0,
        "both_incorrect": 0,
    }
    for teacher_value, student_value, reference in zip(
        teacher_predictions,
        student_predictions,
        references,
    ):
        teacher_correct = teacher_value == reference
        student_correct = student_value == reference
        if teacher_correct and student_correct:
            result["both_correct"] += 1
        elif teacher_correct:
            result["teacher_only_correct"] += 1
        elif student_correct:
            result["student_only_correct"] += 1
        else:
            result["both_incorrect"] += 1
    return result


def build_scope_model(
    *,
    vectors: Any,
    components: int,
    regularization: float,
    percentile: float,
) -> Mapping[str, Any]:
    import numpy as np

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise CertifyError("scope fitting requires at least two embedding vectors")
    center = values.mean(axis=0)
    centered = values - center
    _u, singular_values, right_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    component_count = min(components, values.shape[0] - 1, values.shape[1])
    basis = right_vectors[:component_count]
    projected = centered @ basis.T
    covariance = np.cov(projected, rowvar=False)
    covariance = np.atleast_2d(covariance)
    variance_scale = max(float(np.trace(covariance) / component_count), 1.0e-12)
    regularized_covariance = covariance + (
        regularization * variance_scale * np.eye(component_count)
    )
    inverse_covariance = np.linalg.pinv(regularized_covariance)
    distances = np.sqrt(
        np.maximum(
            0.0,
            np.einsum("ni,ij,nj->n", projected, inverse_covariance, projected),
        )
    )
    total_variance = float((singular_values ** 2).sum())
    explained = (
        float((singular_values[:component_count] ** 2).sum() / total_variance)
        if total_variance > 0.0
        else 0.0
    )
    return {
        "mean": center.astype(np.float32),
        "components": basis.astype(np.float32),
        "covariance": regularized_covariance.astype(np.float32),
        "inverse_covariance": inverse_covariance.astype(np.float32),
        "distances": distances.astype(np.float32),
        "threshold": float(np.percentile(distances, percentile)),
        "explained_variance_ratio": explained,
    }


def collect_scope_vectors(
    *,
    student: Any,
    tokenizer: Any,
    spec: CapabilitySpec,
    examples: Sequence[Any],
    max_prompt_tokens: int,
    scope_layer: int,
    device: Any,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Any]:
    import torch

    vectors = []
    for index, example in enumerate(examples, start=1):
        if index == 1 or index % 25 == 0 or index == len(examples):
            _report(progress, "scope calibration {0}/{1}".format(index, len(examples)))
        prompt = render_prompt(tokenizer, spec.format_prompt(example))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = list(prompt_ids)[-max_prompt_tokens:]
        if not prompt_ids:
            raise CertifyError(
                "scope example has an empty encoded prompt: {0}".format(example.id)
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output = student(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
        vectors.append(
            output.hidden_states[scope_layer][0, -1, :].float().detach().cpu().numpy()
        )
    return vectors


def scope_distances(
    *,
    vectors: Any,
    center: Any,
    components: Any,
    inverse_covariance: Any,
) -> Any:
    import numpy as np

    centered = np.asarray(vectors, dtype=np.float64) - np.asarray(
        center,
        dtype=np.float64,
    )
    projected = centered @ np.asarray(components, dtype=np.float64).T
    squared = np.einsum(
        "ni,ij,nj->n",
        projected,
        np.asarray(inverse_covariance, dtype=np.float64),
        projected,
    )
    return np.sqrt(np.maximum(0.0, squared))


def scope_mahalanobis(
    *,
    vector: Any,
    center: Any,
    components: Any,
    inverse_covariance: Any,
) -> float:
    import numpy as np

    return float(
        scope_distances(
            vectors=np.asarray([vector]),
            center=center,
            components=components,
            inverse_covariance=inverse_covariance,
        )[0]
    )


def write_scope_npz(path: Path, scope: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{0}.tmp".format(path.name))
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            mean=scope["mean"],
            components=scope["components"],
            covariance=scope["covariance"],
            inverse_covariance=scope["inverse_covariance"],
            threshold=np.asarray([scope["threshold"]], dtype=np.float32),
        )
    os.replace(str(temporary), str(path))


def percentile(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    index = int(round((value / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_lineage(
    *,
    carved_model_path: Path,
    adapter_path: Path,
    teacher_model_path: Path,
) -> Mapping[str, Any]:
    manifest_path = carved_model_path / "manifest.json"
    adapter_config_path = adapter_path / "adapter_config.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    adapter_config = (
        _read_json(adapter_config_path) if adapter_config_path.exists() else {}
    )
    return {
        "teacher_model_path": str(teacher_model_path),
        "carved_model_path": str(carved_model_path),
        "adapter_path": str(adapter_path),
        "teacher_hash": manifest.get("teacher_hash"),
        "carve_type": manifest.get("carve_type"),
        "carve_keep_ratio": manifest.get("keep_ratio"),
        "carve_unit_keep_ratios": manifest.get("unit_keep_ratios"),
        "adapter_base_model_path": adapter_config.get("base_model_name_or_path"),
        "adapter_peft_type": adapter_config.get("peft_type"),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CertifyError("could not read JSON artifact: {0}".format(path)) from exc
    if not isinstance(payload, dict):
        raise CertifyError("JSON artifact must be an object: {0}".format(path))
    return payload


def _prepare_output_dir(output_dir: Path, override: bool) -> None:
    if output_dir.exists():
        if not override:
            raise CertifyError("output directory already exists: {0}".format(output_dir))
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _validate_directory(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise CertifyError("{0} directory does not exist: {1}".format(label, path))


def _validate_args(**values: Any) -> None:
    if values["limit"] is not None and int(values["limit"]) <= 0:
        raise CertifyError("limit must be positive or None")
    if values["scope_limit"] is not None and int(values["scope_limit"]) <= 0:
        raise CertifyError("scope_limit must be positive or None")
    for name in (
        "max_length",
        "max_new_tokens",
        "hidden_layer_stride",
        "scope_layer",
        "scope_components",
    ):
        if int(values[name]) <= 0:
            raise CertifyError("{0} must be positive".format(name))
    for name in ("temperature", "scope_regularization"):
        if float(values[name]) <= 0.0:
            raise CertifyError("{0} must be positive".format(name))
    if float(values["epsilon"]) < 0.0:
        raise CertifyError("epsilon must be non-negative")
    if not 0.0 < float(values["scope_percentile"]) < 100.0:
        raise CertifyError("scope_percentile must be between 0 and 100")
    if int(values["max_length"]) <= int(values["max_new_tokens"]):
        raise CertifyError("max_length must exceed max_new_tokens")


def _resolve_device(torch: Any, device: str) -> Any:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise CertifyError("CUDA was requested but is unavailable")
        return torch.device(device)
    if device == "cpu":
        return torch.device("cpu")
    raise CertifyError("unsupported certify device: {0}".format(device))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
