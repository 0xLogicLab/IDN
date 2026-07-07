"""Visualize transformer hidden-unit activations across configured capabilities."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from idn.analysis.embedding_viz import collect_request_samples
from idn.config import DEFAULT_CAPABILITY_DIR, DEFAULT_MACHINE_CONFIG, PathLike
from idn.models.loader import ChatModel, ModelLoadError


@dataclass(frozen=True)
class ActivationSample:
    capability: str
    sample_id: str
    text: str


def visualize_model_activations(
    model_path: str,
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: PathLike = DEFAULT_MACHINE_CONFIG,
    adapter_path: Optional[str] = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    local_files_only: bool = True,
    capabilities: Optional[Sequence[str]] = None,
    split: Optional[str] = "train",
    samples_per_capability: int = 2,
    seed: int = 13,
    max_chars: int = 2000,
    output_dir: Path = Path("artifacts/analysis"),
    max_length: int = 1024,
    neuron_buckets: int = 96,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Collect hidden-state activations and render an interactive HTML report."""

    samples = load_activation_samples_from_capabilities(
        config_dir=config_dir,
        machine_path=machine_path,
        split=split,
        capabilities=capabilities,
        samples_per_capability=samples_per_capability,
        seed=seed,
        max_chars=max_chars,
    )
    if not samples:
        raise ValueError("No capability samples selected for activation visualization")

    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, model_info = collect_hidden_state_activations(
        model_path=model_path,
        adapter_path=adapter_path,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        samples=samples,
        max_length=max_length,
    )
    report = build_activation_report(
        capability_matrices=matrices,
        samples=samples,
        model_info=model_info,
        neuron_buckets=neuron_buckets,
        top_k=top_k,
    )

    capability_slug = "-".join(safe_slug(name) for name in report["capability_order"])
    stem = "model_activations_{model}_{capabilities}".format(
        model=safe_slug(Path(model_path).name or model_path),
        capabilities=capability_slug or "capabilities",
    )
    json_path = output_dir / "{0}.json".format(stem)
    csv_path = output_dir / "{0}_top_neurons.csv".format(stem)
    html_path = output_dir / "{0}.html".format(stem)

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_top_neurons_csv(csv_path, report)
    html_path.write_text(render_activation_html(report), encoding="utf-8")

    result = dict(report["summary"])
    result["outputs"] = {
        "json": str(json_path),
        "top_neurons_csv": str(csv_path),
        "html": str(html_path),
    }
    return result


def load_activation_samples(
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: PathLike = DEFAULT_MACHINE_CONFIG,
    split: Optional[str] = "train",
    capabilities: Optional[Sequence[str]] = None,
    samples_per_capability: int = 2,
    seed: int = 13,
    max_chars: int = 2000,
) -> List[ActivationSample]:
    return load_activation_samples_from_capabilities(
        config_dir=config_dir,
        machine_path=machine_path,
        split=split,
        capabilities=capabilities,
        samples_per_capability=samples_per_capability,
        seed=seed,
        max_chars=max_chars,
    )


def load_activation_samples_from_capabilities(
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: PathLike = DEFAULT_MACHINE_CONFIG,
    split: Optional[str] = "train",
    capabilities: Optional[Sequence[str]] = None,
    samples_per_capability: int = 2,
    seed: int = 13,
    max_chars: int = 2000,
) -> List[ActivationSample]:
    """Sample request text from configured capability datasets."""

    request_samples = collect_request_samples(
        config_dir=config_dir,
        machine_path=machine_path,
        split=split,
        samples_per_dataset=samples_per_capability,
        seed=seed,
        max_chars=max_chars,
    )
    selected = {capability for capability in capabilities or []}
    samples: List[ActivationSample] = []
    seen_capabilities = set()
    for sample in request_samples:
        seen_capabilities.add(sample.dataset)
        if selected and sample.dataset not in selected:
            continue
        samples.append(
            ActivationSample(
                capability=sample.dataset,
                sample_id="{0}:{1}".format(sample.split, sample.example_id),
                text=sample.text,
            )
        )
    missing = sorted(selected - seen_capabilities)
    if missing:
        raise ValueError("No configured capability dataset found for: {0}".format(", ".join(missing)))
    return samples


def collect_hidden_state_activations(
    model_path: str,
    adapter_path: Optional[str],
    dtype: str,
    device: str,
    local_files_only: bool,
    samples: Sequence[ActivationSample],
    max_length: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Run prompts and return mean absolute hidden-state activation per capability."""

    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError("torch is required for activation visualization") from exc

    chat_model = ChatModel(
        model_path=model_path,
        adapter_path=adapter_path,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    ).load()
    if chat_model.model is None or chat_model.tokenizer is None:
        raise ModelLoadError("Model and tokenizer must be loaded")

    model = chat_model.model
    tokenizer = chat_model.tokenizer
    input_device = model_input_device(model)
    capability_sums: Dict[str, Optional[np.ndarray]] = {}
    capability_counts: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}

    for sample in samples:
        prompt = chat_model.render_prompt([{"role": "user", "content": sample.text}])
        encode_kwargs = {"return_tensors": "pt"}
        if max_length > 0:
            encode_kwargs.update({"truncation": True, "max_length": max_length})
        inputs = tokenizer(prompt, **encode_kwargs)
        inputs = {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ModelLoadError("Model did not return hidden states")
        matrix, token_count = hidden_states_to_matrix(
            hidden_states=hidden_states[1:],
            attention_mask=inputs.get("attention_mask"),
        )
        current = capability_sums.get(sample.capability)
        capability_sums[sample.capability] = (
            matrix if current is None else current + matrix
        )
        capability_counts[sample.capability] = (
            capability_counts.get(sample.capability, 0) + 1
        )
        token_counts[sample.capability] = (
            token_counts.get(sample.capability, 0) + token_count
        )

    matrices = {
        capability: summed / float(capability_counts[capability])
        for capability, summed in capability_sums.items()
        if summed is not None and capability_counts.get(capability, 0) > 0
    }
    config = getattr(model, "config", None)
    model_info = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "dtype": dtype,
        "device": str(input_device),
        "local_files_only": local_files_only,
        "model_type": str(getattr(config, "model_type", "unknown")),
        "num_layers": int(next(iter(matrices.values())).shape[0]) if matrices else 0,
        "hidden_size": int(next(iter(matrices.values())).shape[1]) if matrices else 0,
        "capability_prompt_counts": capability_counts,
        "capability_token_counts": token_counts,
    }
    return matrices, model_info


def model_input_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ModelLoadError("Model has no parameters") from exc


def hidden_states_to_matrix(
    hidden_states: Sequence[Any],
    attention_mask: Optional[Any],
) -> Tuple[np.ndarray, int]:
    """Convert model hidden states into layers x hidden-unit mean abs activations."""

    if not hidden_states:
        raise ValueError("hidden_states cannot be empty")
    if attention_mask is None:
        mask = None
        token_count = int(hidden_states[0].shape[1])
    else:
        mask = attention_mask[0].bool()
        token_count = int(mask.sum().item())
    if token_count <= 0:
        token_count = int(hidden_states[0].shape[1])
        mask = None

    rows = []
    for layer_state in hidden_states:
        values = layer_state[0]
        if mask is not None:
            values = values[mask]
        mean_abs = values.detach().float().abs().mean(dim=0)
        rows.append(mean_abs.cpu().numpy())
    return np.vstack(rows), token_count


def build_activation_report(
    capability_matrices: Mapping[str, np.ndarray],
    samples: Sequence[ActivationSample],
    model_info: Mapping[str, Any],
    neuron_buckets: int,
    top_k: int,
) -> Dict[str, Any]:
    if not capability_matrices:
        raise ValueError("No activation matrices supplied")

    capability_order = sorted(capability_matrices.keys())
    shapes = {name: tuple(matrix.shape) for name, matrix in capability_matrices.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError("All capability activation matrices must have the same shape")

    stacked = np.stack([capability_matrices[name] for name in capability_order], axis=0)
    global_mean = np.mean(stacked, axis=0)
    global_floor = np.maximum(global_mean, 1e-8)
    similarities = capability_cosine_similarities(
        capability_matrices,
        capability_order,
    )

    capability_reports: Dict[str, Any] = {}
    bucketed_matrices = []
    for name in capability_order:
        raw = np.asarray(capability_matrices[name], dtype=np.float64)
        relative = raw / global_floor
        layer_mean = raw.mean(axis=1)
        relative_layer_mean = relative.mean(axis=1)
        bucketed = bucket_neuron_activations(relative, neuron_buckets)
        bucketed_matrices.append(bucketed)
        capability_reports[name] = {
            "prompt_count": int(
                model_info.get("capability_prompt_counts", {}).get(name, 0)
            ),
            "token_count": int(
                model_info.get("capability_token_counts", {}).get(name, 0)
            ),
            "raw_matrix": round_matrix(raw, 6),
            "relative_matrix": round_matrix(relative, 4),
            "bucketed_relative_matrix": round_matrix(bucketed, 4),
            "layer_mean_activation": round_vector(layer_mean, 6),
            "relative_layer_mean": round_vector(relative_layer_mean, 4),
            "top_neurons": top_neurons(raw, relative, top_k=top_k),
        }

    prompt_records = [
        {
            "capability": sample.capability,
            "sample_id": sample.sample_id,
            "text": sample.text,
        }
        for sample in samples
    ]
    num_layers, hidden_size = next(iter(shapes.values()))
    bucketed_stack = np.stack(bucketed_matrices, axis=0)
    bucketed_deviation = float(np.max(np.abs(bucketed_stack - 1.0)))
    return {
        "schema_version": 1,
        "summary": {
            "model_path": str(model_info.get("model_path", "")),
            "adapter_path": model_info.get("adapter_path"),
            "model_type": model_info.get("model_type", "unknown"),
            "num_capabilities": len(capability_order),
            "num_prompts": len(samples),
            "num_layers": int(num_layers),
            "hidden_size": int(hidden_size),
            "neuron_buckets": int(min(max(1, neuron_buckets), hidden_size)),
            "relative_bucket_deviation": round(bucketed_deviation, 6),
            "top_k": int(top_k),
            "activation_definition": (
                "Mean absolute residual-stream hidden-state value per layer and "
                "hidden dimension, averaged over non-padding prompt tokens."
            ),
        },
        "model": dict(model_info),
        "capability_order": capability_order,
        "prompts": prompt_records,
        "capability_similarities": similarities,
        "capabilities": capability_reports,
    }


def bucket_neuron_activations(matrix: np.ndarray, buckets: int) -> np.ndarray:
    """Average contiguous hidden dimensions into a fixed number of visual buckets."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be 2D")
    layer_count, hidden_size = values.shape
    bucket_count = min(max(1, int(buckets)), hidden_size)
    edges = np.linspace(0, hidden_size, bucket_count + 1, dtype=int)
    output = np.zeros((layer_count, bucket_count), dtype=np.float64)
    for bucket_index in range(bucket_count):
        start = int(edges[bucket_index])
        end = int(edges[bucket_index + 1])
        if end <= start:
            end = start + 1
        output[:, bucket_index] = values[:, start:end].mean(axis=1)
    return output


def top_neurons(raw: np.ndarray, relative: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    limit = min(int(top_k), relative.size)
    flat_indices = np.argpartition(relative.ravel(), -limit)[-limit:]
    flat_indices = flat_indices[np.argsort(relative.ravel()[flat_indices])[::-1]]
    rows = []
    hidden_size = relative.shape[1]
    for flat_index in flat_indices:
        layer_index = int(flat_index // hidden_size)
        neuron_index = int(flat_index % hidden_size)
        rows.append(
            {
                "layer": layer_index + 1,
                "neuron": neuron_index,
                "mean_abs_activation": round(float(raw[layer_index, neuron_index]), 6),
                "relative_to_global_mean": round(
                    float(relative[layer_index, neuron_index]),
                    4,
                ),
            }
        )
    return rows


def capability_cosine_similarities(
    capability_matrices: Mapping[str, np.ndarray],
    capability_order: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = []
    vectors = {
        name: np.asarray(capability_matrices[name]).ravel()
        for name in capability_order
    }
    for left in capability_order:
        for right in capability_order:
            rows.append(
                {
                    "capability_a": left,
                    "capability_b": right,
                    "cosine": round(float(cosine(vectors[left], vectors[right])), 6),
                }
            )
    return rows


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def write_top_neurons_csv(path: Path, report: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "capability",
                "rank",
                "layer",
                "neuron",
                "mean_abs_activation",
                "relative_to_global_mean",
            ],
        )
        writer.writeheader()
        for capability in report["capability_order"]:
            for rank, row in enumerate(
                report["capabilities"][capability]["top_neurons"],
                start=1,
            ):
                writer.writerow(
                    {
                        "capability": capability,
                        "rank": rank,
                        "layer": row["layer"],
                        "neuron": row["neuron"],
                        "mean_abs_activation": row["mean_abs_activation"],
                        "relative_to_global_mean": row["relative_to_global_mean"],
                    }
                )


def render_activation_html(report: Mapping[str, Any]) -> str:
    """Render a self-contained HTML dashboard for activation inspection."""

    data = json.dumps(report, sort_keys=True, separators=(",", ":"))
    title = "Model Activation Report"
    summary = report["summary"]
    escaped_model = escape(str(summary.get("model_path", "")))
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #172033;
  --muted: #667085;
  --line: #d9dee8;
  --accent: #0f766e;
  --accent-ink: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{
  padding: 24px 28px 14px;
  border-bottom: 1px solid var(--line);
  background: #ffffff;
}}
h1 {{ margin: 0 0 6px; font-size: 24px; font-weight: 700; letter-spacing: 0; }}
h2 {{ margin: 0 0 12px; font-size: 16px; font-weight: 700; letter-spacing: 0; }}
.subtle {{ color: var(--muted); }}
main {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-top: 16px;
}}
.metric {{
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 10px 12px;
}}
.metric strong {{ display: block; font-size: 18px; }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 16px; }}
button {{
  border: 1px solid var(--line);
  background: #ffffff;
  color: var(--ink);
  border-radius: 7px;
  padding: 8px 11px;
  cursor: pointer;
}}
button.active {{ background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}}
.heatmap {{
  display: grid;
  gap: 2px;
  align-items: stretch;
  overflow-x: auto;
  padding-bottom: 4px;
}}
.layer-label {{
  font-size: 11px;
  color: var(--muted);
  text-align: right;
  padding-right: 6px;
  min-width: 58px;
}}
.cell {{
  min-width: 7px;
  height: 14px;
  border-radius: 2px;
}}
.controls {{
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}}
select {{ padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px; background: #fff; }}
canvas {{
  width: 100%;
  height: 48px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #f8fafc;
}}
.grid2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 0.72fr); gap: 16px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; }}
th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
.bars {{ display: grid; gap: 5px; }}
.bar-row {{ display: grid; grid-template-columns: 54px 1fr 54px; gap: 8px; align-items: center; }}
.bar-track {{ height: 10px; background: #edf1f7; border-radius: 4px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: #2563eb; }}
.note {{ max-width: 980px; margin-top: 10px; color: var(--muted); font-size: 13px; }}
@media (max-width: 900px) {{
  main {{ padding: 14px; }}
  .grid2 {{ grid-template-columns: 1fr; }}
  header {{ padding: 18px 16px 12px; }}
}}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="subtle">{model}</div>
  <div class="metrics">
    <div class="metric"><span class="subtle">Capabilities</span><strong>{capabilities}</strong></div>
    <div class="metric"><span class="subtle">Prompts</span><strong>{prompts}</strong></div>
    <div class="metric"><span class="subtle">Layers</span><strong>{layers}</strong></div>
    <div class="metric"><span class="subtle">Hidden units</span><strong>{hidden}</strong></div>
  </div>
  <div class="note">{definition}</div>
</header>
<main>
  <div id="tabs" class="tabs"></div>
  <section class="panel">
    <h2>Layer x Neuron-Bucket Contrast</h2>
    <div id="heatmap" class="heatmap"></div>
  </section>
  <div class="grid2">
    <section class="panel">
      <h2>Selected Layer Neurons</h2>
      <div class="controls">
        <label for="layerSelect" class="subtle">Layer</label>
        <select id="layerSelect"></select>
        <span id="layerStats" class="subtle"></span>
      </div>
      <canvas id="neuronCanvas"></canvas>
    </section>
    <section class="panel">
      <h2>Layer Profile</h2>
      <div id="layerBars" class="bars"></div>
    </section>
  </div>
  <section class="panel">
    <h2>Top Capability-Contrasting Hidden Units</h2>
    <table>
      <thead><tr><th>Rank</th><th>Layer</th><th>Neuron</th><th>Mean abs</th><th>Relative</th></tr></thead>
      <tbody id="topTable"></tbody>
    </table>
  </section>
</main>
<script>
const REPORT = {data};
let currentCapability = REPORT.capability_order[0];
const tabs = document.getElementById("tabs");
const heatmap = document.getElementById("heatmap");
const layerSelect = document.getElementById("layerSelect");
const layerStats = document.getElementById("layerStats");
const layerBars = document.getElementById("layerBars");
const topTable = document.getElementById("topTable");
const canvas = document.getElementById("neuronCanvas");

function clamp(value, min, max) {{ return Math.max(min, Math.min(max, value)); }}
function lerp(a, b, t) {{ return Math.round(a + (b - a) * t); }}
function colorFor(value) {{
  const deviation = Math.max(0.01, REPORT.summary.relative_bucket_deviation || 0.05);
  const x = clamp(0.5 + (value - 1.0) / (2.0 * deviation), 0, 1);
  const stops = [[37,99,235], [229,231,235], [220,38,38]];
  const left = x < 0.5 ? stops[0] : stops[1];
  const right = x < 0.5 ? stops[1] : stops[2];
  const t = x < 0.5 ? x * 2 : (x - 0.5) * 2;
  return `rgb(${{lerp(left[0], right[0], t)}}, ${{lerp(left[1], right[1], t)}}, ${{lerp(left[2], right[2], t)}})`;
}}
function fmt(value, digits=3) {{ return Number(value).toFixed(digits); }}

function renderTabs() {{
  tabs.innerHTML = "";
  REPORT.capability_order.forEach(capability => {{
    const button = document.createElement("button");
    button.textContent = capability;
    button.className = capability === currentCapability ? "active" : "";
    button.onclick = () => {{ currentCapability = capability; render(); }};
    tabs.appendChild(button);
  }});
}}

function renderHeatmap() {{
  const matrix = REPORT.capabilities[currentCapability].bucketed_relative_matrix;
  const rows = matrix.length;
  const cols = matrix[0].length;
  heatmap.style.gridTemplateColumns = `58px repeat(${{cols}}, minmax(7px, 1fr))`;
  heatmap.innerHTML = "";
  matrix.forEach((row, layerIndex) => {{
    const label = document.createElement("div");
    label.className = "layer-label";
    label.textContent = `L${{layerIndex + 1}}`;
    heatmap.appendChild(label);
    row.forEach((value, bucketIndex) => {{
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.style.background = colorFor(value);
      cell.title = `Capability ${{currentCapability}}, layer ${{layerIndex + 1}}, bucket ${{bucketIndex + 1}}/${{cols}}, relative ${{fmt(value)}}`;
      heatmap.appendChild(cell);
    }});
  }});
  layerSelect.innerHTML = "";
  for (let index = 0; index < rows; index += 1) {{
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Layer ${{index + 1}}`;
    layerSelect.appendChild(option);
  }}
}}

function drawNeuronCanvas() {{
  const layerIndex = Number(layerSelect.value || 0);
  const values = REPORT.capabilities[currentCapability].relative_matrix[layerIndex];
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const widthPerNeuron = canvas.width / values.length;
  values.forEach((value, index) => {{
    ctx.fillStyle = colorFor(value);
    ctx.fillRect(Math.floor(index * widthPerNeuron), 0, Math.ceil(widthPerNeuron), canvas.height);
  }});
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const max = Math.max(...values);
  layerStats.textContent = `${{values.length}} units, mean relative ${{fmt(mean)}}, max ${{fmt(max)}}`;
}}

function renderLayerBars() {{
  const values = REPORT.capabilities[currentCapability].relative_layer_mean;
  const max = Math.max(...values, 1);
  layerBars.innerHTML = "";
  values.forEach((value, index) => {{
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span class="subtle">L${{index + 1}}</span><div class="bar-track"><div class="bar-fill" style="width:${{Math.max(3, value / max * 100)}}%; background:${{colorFor(value)}}"></div></div><span>${{fmt(value)}}</span>`;
    layerBars.appendChild(row);
  }});
}}

function renderTopTable() {{
  const rows = REPORT.capabilities[currentCapability].top_neurons;
  topTable.innerHTML = "";
  rows.forEach((row, index) => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${{index + 1}}</td><td>${{row.layer}}</td><td>${{row.neuron}}</td><td>${{fmt(row.mean_abs_activation, 5)}}</td><td>${{fmt(row.relative_to_global_mean)}}</td>`;
    topTable.appendChild(tr);
  }});
}}

function render() {{
  renderTabs();
  renderHeatmap();
  drawNeuronCanvas();
  renderLayerBars();
  renderTopTable();
}}
layerSelect.addEventListener("change", drawNeuronCanvas);
window.addEventListener("resize", drawNeuronCanvas);
render();
</script>
</body>
</html>
""".format(
        title=escape(title),
        model=escaped_model,
        capabilities=int(summary.get("num_capabilities", 0)),
        prompts=int(summary.get("num_prompts", 0)),
        layers=int(summary.get("num_layers", 0)),
        hidden=int(summary.get("hidden_size", 0)),
        definition=escape(str(summary.get("activation_definition", ""))),
        data=data,
    )


def round_matrix(values: np.ndarray, decimals: int) -> List[List[float]]:
    return [
        [round(float(value), decimals) for value in row]
        for row in np.asarray(values).tolist()
    ]


def round_vector(values: Iterable[float], decimals: int) -> List[float]:
    return [round(float(value), decimals) for value in values]


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "unknown"


def format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return "{0:.3f}".format(float(value))
