#!/usr/bin/env python3
"""Visualize centers and 2D slices of the semantic routing regions."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

from idn.routing.router import SemanticEmbedder
from idn.routing.space import RoutingSpace, _collect_configured_texts


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


def _embed(
    texts: Sequence[str],
    model_path: str,
    cache_path: Path,
    max_length: int,
    batch_size: int,
    recompute: bool,
) -> np.ndarray:
    if recompute and cache_path.exists():
        cache_path.unlink()
    if cache_path.exists():
        with np.load(str(cache_path), allow_pickle=False) as cached:
            embeddings = np.asarray(cached["embeddings"], dtype=np.float64)
        if len(embeddings) != len(texts):
            raise ValueError("cached routing embeddings do not match sampled inputs")
        print("Loaded cached embeddings: {0}".format(cache_path), flush=True)
        return embeddings
    embedder = SemanticEmbedder(
        model_path=model_path,
        dtype="bfloat16",
        device="cuda",
        local_files_only=True,
        max_length=max_length,
        batch_size=batch_size,
    ).load()
    embeddings = embedder.embed(texts, progress=lambda message: print(message, flush=True))
    np.savez_compressed(str(cache_path), embeddings=np.asarray(embeddings, dtype=np.float32))
    del embedder
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return np.asarray(embeddings, dtype=np.float64)


def _distances(space: RoutingSpace, projected: np.ndarray) -> np.ndarray:
    columns = []
    for centroid, precision in zip(space.centroids, space.precisions):
        differences = projected - centroid
        squared = np.einsum("ni,ij,nj->n", differences, precision, differences)
        columns.append(np.sqrt(np.maximum(squared, 0.0)))
    return np.stack(columns, axis=1)


def _center_slice_ellipse(space: RoutingSpace, class_index: int) -> Tuple[float, float, float]:
    precision_2d = space.precisions[class_index, :2, :2]
    covariance_2d = np.linalg.pinv(precision_2d, hermitian=True)
    values, vectors = np.linalg.eigh(covariance_2d)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * space.ood_threshold * np.sqrt(values)
    return float(width), float(height), angle


def _grid(
    space: RoutingSpace,
    projected: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.vstack([projected[:, :2], space.centroids[:, :2]])
    low = np.quantile(values, 0.01, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    margin = np.maximum((high - low) * 0.12, 0.5)
    xs = np.linspace(low[0] - margin[0], high[0] + margin[0], size)
    ys = np.linspace(low[1] - margin[1], high[1] + margin[1], size)
    xx, yy = np.meshgrid(xs, ys)
    slice_points = np.zeros((size * size, space.dimensions), dtype=np.float64)
    slice_points[:, 0] = xx.ravel()
    slice_points[:, 1] = yy.ravel()
    distances = _distances(space, slice_points)
    regions = np.argmin(distances, axis=1).reshape(size, size)
    min_distances = np.min(distances, axis=1).reshape(size, size)
    return xx, yy, regions, min_distances


def _scatter_samples(
    ax: Any,
    space: RoutingSpace,
    projected: np.ndarray,
    labels: Sequence[str],
    accepted: np.ndarray,
) -> None:
    labels_array = np.asarray(labels)
    for index, capability in enumerate(space.capabilities):
        mask = labels_array == capability
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=16,
            alpha=0.55,
            color=COLORS[capability],
            edgecolors="none",
            label=DISPLAY_NAMES[capability],
        )
        rejected = mask & ~accepted
        if np.any(rejected):
            ax.scatter(
                projected[rejected, 0],
                projected[rejected, 1],
                s=26,
                marker="x",
                linewidths=0.8,
                color=COLORS[capability],
            )
        center = space.centroids[index, :2]
        ax.scatter(center[0], center[1], marker="*", s=190, color=COLORS[capability], edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(center[0], center[1], DISPLAY_NAMES[capability], fontsize=9.5, weight="bold", color=COLORS[capability], ha="center", va="bottom")


def _render(
    path: Path,
    space: RoutingSpace,
    projected: np.ndarray,
    labels: Sequence[str],
    predictions: Sequence[str],
    accepted: np.ndarray,
    grid_size: int,
) -> None:
    explained = list(space.metadata.get("pca_explained_variance_ratio", []))
    pc1 = float(explained[0]) * 100 if len(explained) > 0 else float("nan")
    pc2 = float(explained[1]) * 100 if len(explained) > 1 else float("nan")
    xx, yy, regions, min_distances = _grid(space, projected, grid_size)
    cmap = ListedColormap([COLORS[name] for name in space.capabilities])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), sharex=True, sharey=True)

    _scatter_samples(axes[0], space, projected, labels, accepted)
    for index, capability in enumerate(space.capabilities):
        width, height, angle = _center_slice_ellipse(space, index)
        center = space.centroids[index, :2]
        axes[0].add_patch(
            Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor=COLORS[capability],
                alpha=0.9,
            )
        )
    axes[0].set_title("样本、类中心与中心截面", fontsize=13, weight="bold", pad=12)
    axes[0].legend(frameon=False, fontsize=8.5, loc="best", markerscale=1.3)

    axes[1].pcolormesh(xx, yy, regions, cmap=cmap, shading="auto", alpha=0.18, vmin=-0.5, vmax=len(space.capabilities) - 0.5)
    contour = axes[1].contour(xx, yy, min_distances, levels=[space.ood_threshold], colors="#222222", linewidths=1.4, linestyles="--")
    if contour.allsegs[0]:
        axes[1].clabel(contour, fmt={space.ood_threshold: "OOD threshold"}, inline=True, fontsize=8)
    _scatter_samples(axes[1], space, projected, labels, accepted)
    axes[1].set_title("最近类决策区域与 OOD 边界", fontsize=13, weight="bold", pad=12)

    for ax in axes:
        ax.set_xlabel("Routing PCA 1 ({0:.1f}%)".format(pc1), fontsize=11)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Routing PCA 2 ({0:.1f}%)".format(pc2), fontsize=11)
    accuracy = np.mean(np.asarray(predictions) == np.asarray(labels)) * 100
    coverage = np.mean(accepted) * 100
    fig.suptitle(
        "语义路由空间二维切片（64D → PCA1/PCA2）\n样本路由准确率 {0:.1f}%  |  OOD 覆盖率 {1:.1f}%".format(accuracy, coverage),
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "虚线椭圆：穿过各类中心的阈值截面；背景与黑虚线：其余 PCA 维固定为全局均值时的决策切片。",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(w_pad=2.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_points(
    path: Path,
    projected: np.ndarray,
    labels: Sequence[str],
    predictions: Sequence[str],
    accepted: np.ndarray,
    distances: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "prediction", "accepted", "pca1", "pca2", "minimum_distance"])
        writer.writeheader()
        for point, label, prediction, keep, row_distances in zip(projected, labels, predictions, accepted, distances):
            writer.writerow(
                {
                    "label": label,
                    "prediction": prediction,
                    "accepted": int(keep),
                    "pca1": "{0:.8f}".format(float(point[0])),
                    "pca2": "{0:.8f}".format(float(point[1])),
                    "minimum_distance": "{0:.8f}".format(float(np.min(row_distances))),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="artifacts/routing/space.npz")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--samples-per-capability", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/analysis/routing_regions")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    _configure_font()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    space = RoutingSpace.load(Path(args.space))
    texts, labels, counts = _collect_configured_texts(
        config_dir=Path(args.config_dir),
        machine_path=args.machine,
        split_kind="train",
        samples_per_capability=args.samples_per_capability,
        seed=int(space.metadata.get("seed", 13)),
        max_chars=2000,
    )
    embeddings = _embed(
        texts,
        model_path=str(space.metadata["model_path"]),
        cache_path=output_dir / "routing_input_embeddings.npz",
        max_length=int(space.metadata.get("max_length", 512)),
        batch_size=args.batch_size,
        recompute=args.recompute,
    )
    projected = space.project(embeddings)
    distances = _distances(space, projected)
    predicted_indices = np.argmin(distances, axis=1)
    predictions = [space.capabilities[index] for index in predicted_indices]
    accepted = np.min(distances, axis=1) <= space.ood_threshold
    output = output_dir / "routing_centers_and_regions.png"
    _render(output, space, projected, labels, predictions, accepted, args.grid_size)
    _write_points(output_dir / "routing_points.csv", projected, labels, predictions, accepted, distances)
    result: Dict[str, Any] = {
        "space": args.space,
        "dimensions": space.dimensions,
        "embedding_size": space.embedding_size,
        "capabilities": list(space.capabilities),
        "samples": len(labels),
        "samples_by_capability": counts,
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        "coverage": float(np.mean(accepted)),
        "ood_threshold": float(space.ood_threshold),
        "slice": "PCA dimensions 3-64 fixed at global projected mean (zero)",
        "output": str(output),
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
