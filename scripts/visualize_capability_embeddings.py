#!/usr/bin/env python3
"""Compare TF-IDF and Qwen2.5-1.5B embeddings for six capabilities."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from idn.analysis.embedding_viz import (
    RequestSample,
    cluster_diagnostics,
    collect_request_samples,
    embed_tfidf,
    project_embeddings,
)
from idn.routing.router import SemanticEmbedder


CAPABILITY_ORDER = ["cnn_dailymail", "cola", "gsm8k", "mbpp", "squad", "sst2"]
DISPLAY_NAMES = {
    "cnn_dailymail": "CNN/DailyMail",
    "cola": "CoLA",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "squad": "SQuAD",
    "sst2": "SST-2",
}
COLORS = {
    "cnn_dailymail": "#4c78a8",
    "cola": "#f58518",
    "gsm8k": "#54a24b",
    "mbpp": "#e45756",
    "squad": "#9d6bb3",
    "sst2": "#8c6d31",
}


def _configure_font() -> None:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})


def _balanced_samples(args: argparse.Namespace) -> List[RequestSample]:
    samples = collect_request_samples(
        config_dir=Path(args.config_dir),
        machine_path=args.machine,
        split=args.split,
        samples_per_dataset=args.samples_per_capability,
        seed=args.seed,
        max_chars=args.max_chars,
    )
    counts = {name: 0 for name in CAPABILITY_ORDER}
    for sample in samples:
        counts[sample.dataset] = counts.get(sample.dataset, 0) + 1
    missing = [name for name, count in counts.items() if count < args.samples_per_capability]
    if missing:
        raise ValueError(
            "balanced comparison needs {0} samples per capability; short datasets: {1}".format(
                args.samples_per_capability,
                ", ".join("{0}={1}".format(name, counts[name]) for name in missing),
            )
        )
    samples = [sample for sample in samples if sample.dataset in CAPABILITY_ORDER]
    samples.sort(key=lambda sample: (CAPABILITY_ORDER.index(sample.dataset), sample.example_id))
    return samples


def _qwen_reduced(
    texts: Sequence[str],
    model_path: str,
    cache_path: Path,
    seed: int,
    max_length: int,
    batch_size: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    if cache_path.exists():
        with np.load(str(cache_path), allow_pickle=False) as cached:
            raw = np.asarray(cached["embeddings"], dtype=np.float64)
        if len(raw) != len(texts):
            raise ValueError("cached embedding count does not match sampled inputs")
        print("Loaded cached Qwen embeddings: {0}".format(cache_path), flush=True)
    else:
        print("Loading Qwen2.5-1.5B embedding model", flush=True)
        embedder = SemanticEmbedder(
            model_path=model_path,
            dtype="bfloat16",
            device="cuda",
            local_files_only=True,
            max_length=max_length,
            batch_size=batch_size,
        ).load()
        raw = embedder.embed(texts, progress=lambda message: print(message, flush=True))
        np.savez_compressed(str(cache_path), embeddings=np.asarray(raw, dtype=np.float32))
        del embedder
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    raw_shape = (int(raw.shape[0]), int(raw.shape[1]))
    unit = normalize(raw, norm="l2", axis=1)
    dimensions = min(50, unit.shape[0] - 1, unit.shape[1])
    reduced = PCA(n_components=dimensions, random_state=seed).fit_transform(unit)
    return np.asarray(reduced, dtype=np.float64), raw_shape


def _diagnostic_text(diagnostics: Mapping[str, Any]) -> str:
    return "Silhouette {0:.3f}  |  Linear CV {1:.1f}%".format(
        float(diagnostics["silhouette"]),
        float(diagnostics["linear_cv_accuracy_mean"]) * 100,
    )


def _scatter(
    ax: Any,
    samples: Sequence[RequestSample],
    coordinates: np.ndarray,
    title: str,
    diagnostics: Mapping[str, Any],
    show_legend: bool,
) -> None:
    labels = np.asarray([sample.dataset for sample in samples])
    for capability in CAPABILITY_ORDER:
        mask = labels == capability
        points = coordinates[mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            alpha=0.68,
            color=COLORS[capability],
            edgecolors="none",
            label=DISPLAY_NAMES[capability],
        )
        centroid = points.mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            DISPLAY_NAMES[capability],
            color=COLORS[capability],
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    ax.set_title("{0}\n{1}".format(title, _diagnostic_text(diagnostics)), fontsize=13, weight="bold", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, loc="best", fontsize=9, markerscale=1.4)


def _render_standalone(
    path: Path,
    samples: Sequence[RequestSample],
    coordinates: np.ndarray,
    title: str,
    diagnostics: Mapping[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    _scatter(ax, samples, coordinates, title, diagnostics, show_legend=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_comparison(
    path: Path,
    samples: Sequence[RequestSample],
    tfidf_coordinates: np.ndarray,
    qwen_coordinates: np.ndarray,
    tfidf_diagnostics: Mapping[str, Any],
    qwen_diagnostics: Mapping[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    _scatter(axes[0], samples, tfidf_coordinates, "TF-IDF + t-SNE", tfidf_diagnostics, False)
    _scatter(axes[1], samples, qwen_coordinates, "Qwen2.5-1.5B Embedding + t-SNE", qwen_diagnostics, True)
    fig.suptitle("Six Capability Input Clusters", fontsize=16, weight="bold", y=1.01)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_coordinates(
    path: Path,
    samples: Sequence[RequestSample],
    coordinates: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "example_id", "x", "y", "text"])
        writer.writeheader()
        for sample, point in zip(samples, coordinates):
            writer.writerow(
                {
                    "dataset": sample.dataset,
                    "split": sample.split,
                    "example_id": sample.example_id,
                    "x": "{0:.8f}".format(float(point[0])),
                    "y": "{0:.8f}".format(float(point[1])),
                    "text": sample.text,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-capability", type=int, default=120)
    parser.add_argument("--model", default="/data/huggingface/model/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="artifacts/analysis/capability_embedding_comparison")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--recompute-model", action="store_true")
    args = parser.parse_args()
    if args.samples_per_capability <= 0:
        parser.error("--samples-per-capability must be positive")

    _configure_font()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "qwen1.5b_embeddings.npz"
    if args.recompute_model and cache_path.exists():
        cache_path.unlink()

    samples = _balanced_samples(args)
    texts = [sample.text for sample in samples]
    labels = [sample.dataset for sample in samples]
    print("Collected {0} balanced inputs ({1} per capability)".format(len(samples), args.samples_per_capability), flush=True)

    tfidf_vectors, tfidf_reduced = embed_tfidf(texts, seed=args.seed)
    qwen_reduced, qwen_raw_shape = _qwen_reduced(
        texts,
        model_path=args.model,
        cache_path=cache_path,
        seed=args.seed,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print("Projecting TF-IDF with t-SNE", flush=True)
    tfidf_coordinates = project_embeddings(tfidf_reduced, "tsne", args.seed, args.perplexity)
    print("Projecting Qwen embeddings with t-SNE", flush=True)
    qwen_coordinates = project_embeddings(qwen_reduced, "tsne", args.seed, args.perplexity)
    tfidf_diagnostics = cluster_diagnostics(tfidf_reduced, labels, args.seed)
    qwen_diagnostics = cluster_diagnostics(qwen_reduced, labels, args.seed)

    _render_standalone(output_dir / "tfidf_tsne.png", samples, tfidf_coordinates, "TF-IDF + t-SNE", tfidf_diagnostics)
    _render_standalone(output_dir / "qwen1.5b_tsne.png", samples, qwen_coordinates, "Qwen2.5-1.5B Embedding + t-SNE", qwen_diagnostics)
    _render_comparison(output_dir / "tfidf_vs_qwen1.5b.png", samples, tfidf_coordinates, qwen_coordinates, tfidf_diagnostics, qwen_diagnostics)
    _write_coordinates(output_dir / "tfidf_coordinates.csv", samples, tfidf_coordinates)
    _write_coordinates(output_dir / "qwen1.5b_coordinates.csv", samples, qwen_coordinates)

    result: Dict[str, Any] = {
        "samples": len(samples),
        "samples_per_capability": args.samples_per_capability,
        "capabilities": CAPABILITY_ORDER,
        "split": args.split,
        "seed": args.seed,
        "projection": {"method": "tsne", "perplexity": args.perplexity},
        "tfidf": {"vector_shape": list(tfidf_vectors.shape), "reduced_shape": list(tfidf_reduced.shape), "diagnostics": tfidf_diagnostics},
        "qwen1.5b": {"model": args.model, "pooling": "attention_mask_mean_last_hidden_state", "max_length": args.max_length, "raw_shape": list(qwen_raw_shape), "reduced_shape": list(qwen_reduced.shape), "diagnostics": qwen_diagnostics},
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
