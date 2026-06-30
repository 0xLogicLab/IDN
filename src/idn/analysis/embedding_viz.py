"""Visualize request embeddings across configured capability datasets."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from idn.config import (
    DEFAULT_CAPABILITY_DIR,
    DEFAULT_MACHINE_CONFIG,
    CapabilityConfig,
    discover_capability_configs,
    load_resolved_config,
)


RAW_PARQUET_NAMES = {
    "cnn_dailymail": "cnn_dailymail",
    "gsm8k": "gsm8k",
    "mbpp": "mbpp_sanitized",
    "squad": "squad",
}

TARGET_LIKE_KEYS = {
    "answer",
    "answers",
    "code",
    "highlights",
    "label",
    "solution",
    "target",
}


@dataclass(frozen=True)
class RequestSample:
    dataset: str
    split: str
    example_id: str
    text: str


def visualize_dataset_embeddings(
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: Path = DEFAULT_MACHINE_CONFIG,
    split: Optional[str] = "train",
    samples_per_dataset: int = 300,
    output_dir: Path = Path("artifacts/analysis"),
    embedding: str = "tfidf",
    projection: str = "tsne",
    seed: int = 13,
    max_chars: int = 2000,
    perplexity: float = 30.0,
) -> Dict[str, Any]:
    """Sample each configured dataset, embed request text, and render a 2D plot."""

    if embedding != "tfidf":
        raise ValueError("Only embedding='tfidf' is currently supported")
    if projection not in ("pca", "tsne"):
        raise ValueError("projection must be 'pca' or 'tsne'")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_request_samples(
        config_dir=config_dir,
        machine_path=machine_path,
        split=split,
        samples_per_dataset=samples_per_dataset,
        seed=seed,
        max_chars=max_chars,
    )
    if len(samples) < 2:
        raise ValueError("Need at least two request samples to visualize embeddings")

    vectors, reduced = embed_tfidf([sample.text for sample in samples], seed=seed)
    coordinates = project_embeddings(
        reduced=reduced,
        projection=projection,
        seed=seed,
        perplexity=perplexity,
    )
    diagnostics = cluster_diagnostics(reduced, [sample.dataset for sample in samples], seed)

    stem = "dataset_embeddings_{embedding}_{projection}_n{count}_seed{seed}".format(
        embedding=embedding,
        projection=projection,
        count=len(samples),
        seed=seed,
    )
    csv_path = output_dir / "{0}.csv".format(stem)
    json_path = output_dir / "{0}.json".format(stem)
    plot_path = output_dir / "{0}.png".format(stem)

    write_coordinates_csv(csv_path, samples, coordinates)
    render_plot(plot_path, samples, coordinates, diagnostics, projection)

    result = {
        "samples": len(samples),
        "datasets": sorted({sample.dataset for sample in samples}),
        "samples_per_dataset": samples_per_dataset,
        "embedding": embedding,
        "projection": projection,
        "seed": seed,
        "max_chars": max_chars,
        "vector_shape": list(vectors.shape),
        "diagnostics": diagnostics,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plot": str(plot_path),
        },
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def collect_request_samples(
    config_dir: Path,
    machine_path: Path,
    split: Optional[str],
    samples_per_dataset: int,
    seed: int,
    max_chars: int,
) -> List[RequestSample]:
    samples: List[RequestSample] = []
    for config_path in discover_capability_configs(config_dir):
        resolved = load_resolved_config(config_path, machine_path)
        capability = resolved.capability
        actual_split = split or capability.train_split
        rows = load_dataset_rows(
            capability=capability,
            split=actual_split,
            limit=samples_per_dataset,
            seed=seed,
        )
        for index, row in enumerate(rows):
            text = request_text_from_row(capability.id, row)
            text = normalize_request_text(text, max_chars=max_chars)
            if not text:
                continue
            samples.append(
                RequestSample(
                    dataset=capability.id,
                    split=actual_split,
                    example_id=str(row.get("id", row.get("idx", index))),
                    text=text,
                )
            )
    return samples


def load_dataset_rows(
    capability: CapabilityConfig,
    split: str,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    if capability.id in RAW_PARQUET_NAMES:
        return load_rows_from_parquet(capability, split, limit)
    try:
        return load_rows_from_disk(capability.dataset_path, split, limit, seed)
    except Exception:
        return load_rows_from_parquet(capability, split, limit)


def load_rows_from_disk(
    dataset_path: str,
    split: str,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required for load_from_disk") from exc

    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    elif split not in ("train", "all"):
        raise KeyError("Single split dataset cannot provide split: {0}".format(split))

    count = min(limit, len(dataset))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=count, replace=False)
    return [dict(dataset[int(index)]) for index in sorted(indices)]


def load_rows_from_parquet(
    capability: CapabilityConfig,
    split: str,
    limit: int,
) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet fallback") from exc

    files = parquet_files_for_capability(capability, split)
    rows: List[Dict[str, Any]] = []
    columns = preferred_columns(capability.id)
    for path in files:
        parquet_file = pq.ParquetFile(path)
        existing_columns = [
            column for column in columns if column in parquet_file.schema.names
        ] or None
        for batch in parquet_file.iter_batches(
            batch_size=min(1024, max(1, limit - len(rows))),
            columns=existing_columns,
        ):
            rows.extend(batch.to_pylist())
            if len(rows) >= limit:
                return rows[:limit]
    return rows[:limit]


def parquet_files_for_capability(capability: CapabilityConfig, split: str) -> List[Path]:
    dataset_root = Path(capability.dataset_path).parent
    raw_name = RAW_PARQUET_NAMES.get(capability.id, Path(capability.dataset_path).name)
    raw_dir = dataset_root / "raw_hf_parquet" / raw_name
    if not raw_dir.exists():
        raise FileNotFoundError("No raw parquet fallback dir: {0}".format(raw_dir))
    files = sorted(raw_dir.glob("{0}*.parquet".format(split)))
    if not files:
        raise FileNotFoundError(
            "No parquet files for {0} split {1}".format(capability.id, split)
        )
    return files


def preferred_columns(capability_id: str) -> List[str]:
    return {
        "cnn_dailymail": ["id", "article", "highlights"],
        "cola": ["idx", "sentence", "label"],
        "gsm8k": ["question", "answer"],
        "mbpp": ["task_id", "prompt", "text", "test_list", "code"],
        "squad": ["id", "title", "context", "question", "answers"],
        "sst2": ["idx", "sentence", "label"],
    }.get(capability_id, [])


def request_text_from_row(capability_id: str, row: Mapping[str, Any]) -> str:
    """Extract the user-facing request text for a known capability row."""

    if capability_id in ("cola", "sst2"):
        return str(row.get("sentence", ""))
    if capability_id == "gsm8k":
        return str(row.get("question", ""))
    if capability_id == "squad":
        return join_parts(
            [
                ("Question", row.get("question")),
                ("Context", row.get("context")),
            ]
        )
    if capability_id == "cnn_dailymail":
        return str(row.get("article", ""))
    if capability_id == "mbpp":
        tests = row.get("test_list")
        if isinstance(tests, np.ndarray):
            tests = tests.tolist()
        if isinstance(tests, (list, tuple)):
            tests = "\n".join(str(item) for item in tests)
        return join_parts(
            [
                ("Prompt", row.get("prompt", row.get("text"))),
                ("Tests", tests),
            ]
        )
    return generic_request_text(row)


def generic_request_text(row: Mapping[str, Any]) -> str:
    parts = []
    for key, value in row.items():
        if key.lower() in TARGET_LIKE_KEYS:
            continue
        if isinstance(value, (str, int, float)):
            parts.append(("{0}".format(key), value))
    return join_parts(parts)


def join_parts(parts: Sequence[Tuple[str, Any]]) -> str:
    rendered = []
    for label, value in parts:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            rendered.append("{0}: {1}".format(label, text))
    return "\n\n".join(rendered)


def normalize_request_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0]
    return text


def embed_tfidf(texts: Sequence[str], seed: int) -> Tuple[Any, np.ndarray]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_df=0.95,
        max_features=20000,
        min_df=2 if len(texts) >= 30 else 1,
        ngram_range=(1, 2),
        norm="l2",
        strip_accents="unicode",
    )
    vectors = vectorizer.fit_transform(texts)
    max_components = max(2, min(50, vectors.shape[0] - 1, vectors.shape[1] - 1))
    if max_components >= 2:
        reduced = TruncatedSVD(n_components=max_components, random_state=seed).fit_transform(
            vectors
        )
    else:
        reduced = vectors.toarray()
    return vectors, np.asarray(reduced)


def project_embeddings(
    reduced: np.ndarray,
    projection: str,
    seed: int,
    perplexity: float,
) -> np.ndarray:
    if reduced.shape[0] < 3:
        return pad_2d(reduced)
    if projection == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(reduced)

    from sklearn.manifold import TSNE

    usable_perplexity = min(float(perplexity), max(2.0, (reduced.shape[0] - 1) / 3.0))
    usable_perplexity = min(usable_perplexity, reduced.shape[0] - 1.0)
    return TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        perplexity=usable_perplexity,
        random_state=seed,
    ).fit_transform(reduced)


def pad_2d(values: np.ndarray) -> np.ndarray:
    if values.shape[1] >= 2:
        return values[:, :2]
    return np.column_stack([values[:, 0], np.zeros(values.shape[0])])


def cluster_diagnostics(
    reduced: Sequence[Sequence[float]],
    labels: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    reduced_array = np.asarray(reduced)
    label_list = list(labels)
    encoded = LabelEncoder().fit_transform(label_list)
    counts = {label: label_list.count(label) for label in sorted(set(label_list))}
    diagnostics: Dict[str, Any] = {"counts": counts}
    if len(set(label_list)) > 1 and min(counts.values()) >= 2 and len(label_list) > len(counts):
        diagnostics["silhouette"] = float(silhouette_score(reduced_array, encoded))
    else:
        diagnostics["silhouette"] = None

    min_count = min(counts.values()) if counts else 0
    if len(counts) > 1 and min_count >= 3:
        splits = min(5, min_count)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            multi_class="auto",
            random_state=seed,
        )
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        scores = cross_val_score(model, reduced_array, encoded, cv=cv)
        diagnostics["linear_cv_accuracy_mean"] = float(np.mean(scores))
        diagnostics["linear_cv_accuracy_std"] = float(np.std(scores))
        diagnostics["linear_cv_splits"] = int(splits)
    else:
        diagnostics["linear_cv_accuracy_mean"] = None
        diagnostics["linear_cv_accuracy_std"] = None
        diagnostics["linear_cv_splits"] = 0
    return diagnostics


def write_coordinates_csv(
    path: Path,
    samples: Sequence[RequestSample],
    coordinates: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "split", "example_id", "x", "y", "text"],
        )
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


def render_plot(
    path: Path,
    samples: Sequence[RequestSample],
    coordinates: np.ndarray,
    diagnostics: Mapping[str, Any],
    projection: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({sample.dataset for sample in samples})
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    for index, dataset in enumerate(datasets):
        mask = np.array([sample.dataset == dataset for sample in samples])
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=12,
            alpha=0.72,
            label=dataset,
            color=colors(index % 10),
            linewidths=0,
        )
        centroid = coordinates[mask].mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            dataset,
            fontsize=9,
            weight="bold",
            color=colors(index % 10),
        )

    title = "Request Embedding Projection by Dataset ({0})".format(projection.upper())
    subtitle = "silhouette={0}, linear-CV={1}".format(
        format_optional_float(diagnostics.get("silhouette")),
        format_optional_float(diagnostics.get("linear_cv_accuracy_mean")),
    )
    ax.set_title("{0}\n{1}".format(title, subtitle), fontsize=12)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return "{0:.3f}".format(float(value))
