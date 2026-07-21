#!/usr/bin/env python3
"""Measure and plot geographic request locality in WildChat-1M."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, load_from_disk
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm


DEFAULT_DATASET = Path("/data/huggingface/dataset/wildchat_1m")
DEFAULT_OUTPUT_DIR = Path("artifacts/wildchat_locality")
COUNTRY_DISPLAY_NAMES = {
    "United States": "United States",
    "United Kingdom": "United Kingdom",
    "The Netherlands": "Netherlands",
    "DR Congo": "DR Congo",
    "Türkiye": "Turkiye",
}


def _configure_font() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_columns(dataset_path: Path) -> Tuple[List[str | None], List[str | None], np.ndarray]:
    loaded = load_from_disk(str(dataset_path))
    dataset: Dataset
    if isinstance(loaded, DatasetDict):
        dataset = loaded["train"]
    else:
        dataset = loaded
    table = dataset.data.table.select(["country", "language", "timestamp"])
    countries = table.column("country").to_pylist()
    languages = table.column("language").to_pylist()
    timestamp_chunks = [
        chunk.cast(pa.int64()).to_numpy(zero_copy_only=False)
        for chunk in table.column("timestamp").chunks
    ]
    timestamps = np.concatenate(timestamp_chunks)
    return countries, languages, timestamps


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _count_pairs(
    countries: Sequence[str | None],
    languages: Sequence[str | None],
    mask: np.ndarray | None = None,
) -> Tuple[Counter[str], Counter[str], Counter[Tuple[str, str]]]:
    country_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    joint_counts: Counter[Tuple[str, str]] = Counter()
    include: Iterable[bool]
    include = mask if mask is not None else (True for _ in countries)
    for raw_country, raw_language, keep in zip(countries, languages, include):
        if not keep:
            continue
        country = _clean(raw_country)
        language = _clean(raw_language)
        if country is None or language is None:
            continue
        country_counts[country] += 1
        language_counts[language] += 1
        joint_counts[(country, language)] += 1
    return country_counts, language_counts, joint_counts


def _explained_language_entropy(
    country_counts: Mapping[str, int],
    language_counts: Mapping[str, int],
    joint_counts: Mapping[Tuple[str, str], int],
) -> Tuple[float, float, float]:
    total = float(sum(joint_counts.values()))
    language_entropy = -sum(
        (count / total) * math.log(count / total)
        for count in language_counts.values()
        if count
    )
    mutual_information = 0.0
    for (country, language), count in joint_counts.items():
        probability = count / total
        independent_probability = (country_counts[country] / total) * (
            language_counts[language] / total
        )
        mutual_information += probability * math.log(probability / independent_probability)
    explained = mutual_information / language_entropy if language_entropy else 0.0
    return mutual_information, language_entropy, explained


def _top_k_coverage(
    train_country: Mapping[str, int],
    train_language: Mapping[str, int],
    train_joint: Mapping[Tuple[str, str], int],
    test_country: Mapping[str, int],
    test_joint: Mapping[Tuple[str, str], int],
    max_k: int,
    min_train_requests: int,
    min_test_requests: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    eligible = sorted(
        country
        for country in test_country
        if train_country.get(country, 0) >= min_train_requests
        and test_country[country] >= min_test_requests
    )
    if not eligible:
        raise ValueError("no countries meet the train/test request thresholds")

    by_country: Dict[str, Counter[str]] = {country: Counter() for country in eligible}
    for (country, language), count in train_joint.items():
        if country in by_country:
            by_country[country][language] += count
    local_rankings = {
        country: [language for language, _ in counts.most_common(max_k)]
        for country, counts in by_country.items()
    }
    global_ranking = [language for language, _ in Counter(train_language).most_common(max_k)]
    test_total = sum(test_country[country] for country in eligible)

    rows: List[Dict[str, Any]] = []
    for k in range(1, max_k + 1):
        global_set = set(global_ranking[:k])
        local_sets = {country: set(ranking[:k]) for country, ranking in local_rankings.items()}
        global_hits = 0
        local_hits = 0
        for (country, language), count in test_joint.items():
            if country not in local_sets:
                continue
            global_hits += count * int(language in global_set)
            local_hits += count * int(language in local_sets[country])
        rows.append(
            {
                "k": k,
                "global_coverage": global_hits / test_total,
                "country_local_coverage": local_hits / test_total,
                "gain_pp": (local_hits - global_hits) * 100.0 / test_total,
            }
        )
    return rows, eligible


def _heatmap_data(
    country_counts: Mapping[str, int],
    language_counts: Mapping[str, int],
    joint_counts: Mapping[Tuple[str, str], int],
    num_countries: int,
    num_languages: int,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    countries = [country for country, _ in Counter(country_counts).most_common(num_countries)]
    languages = [language for language, _ in Counter(language_counts).most_common(num_languages)]
    total = float(sum(joint_counts.values()))
    shares = np.zeros((len(countries), len(languages)), dtype=float)
    log_lifts = np.zeros_like(shares)
    for row, country in enumerate(countries):
        for column, language in enumerate(languages):
            count = joint_counts.get((country, language), 0)
            shares[row, column] = count / country_counts[country]
            global_share = language_counts[language] / total
            # A half-observation avoids negative infinity for empty cells.
            smoothed_share = (count + 0.5) / (country_counts[country] + 0.5)
            log_lifts[row, column] = math.log2(smoothed_share / global_share)
    return countries, languages, shares, log_lifts


def _plot(
    output_stem: Path,
    countries: Sequence[str],
    languages: Sequence[str],
    shares: np.ndarray,
    log_lifts: np.ndarray,
    coverage_rows: Sequence[Mapping[str, Any]],
    explained_entropy: float,
    num_requests: int,
    cutoff: datetime,
    eligible_countries: int,
) -> None:
    _configure_font()
    blue = "#2b6f9f"
    orange = "#d46b45"
    fig, (ax_heatmap, ax_coverage) = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.0),
        gridspec_kw={"width_ratios": [1.75, 1.0]},
    )

    clipped_lifts = np.clip(log_lifts, -3.0, 3.0)
    image = ax_heatmap.imshow(
        clipped_lifts,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0),
        aspect="auto",
    )
    ax_heatmap.set_xticks(np.arange(len(languages)), languages, rotation=43, ha="right")
    ax_heatmap.set_yticks(
        np.arange(len(countries)),
        [COUNTRY_DISPLAY_NAMES.get(country, country) for country in countries],
    )
    ax_heatmap.tick_params(length=0)
    ax_heatmap.spines[:].set_visible(False)
    ax_heatmap.set_title(
        "A  Language demand differs by country",
        loc="left",
        fontsize=13,
        weight="bold",
        pad=12,
    )
    ax_heatmap.set_xlabel("Requested language capability")
    for row in range(len(countries)):
        for column in range(len(languages)):
            share = shares[row, column] * 100.0
            if share < 0.45:
                continue
            color = "white" if abs(clipped_lifts[row, column]) >= 1.75 else "#202020"
            ax_heatmap.text(
                column,
                row,
                f"{share:.0f}" if share >= 10 else f"{share:.1f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.034, pad=0.025)
    colorbar.set_label("Demand lift vs. global mix (log2)")
    colorbar.set_ticks(
        [-3, -2, -1, 0, 1, 2, 3],
        labels=["1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x"],
    )
    ax_heatmap.text(
        0.0,
        -0.19,
        "Cell labels: share of requests from that country (%)",
        transform=ax_heatmap.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    k = np.asarray([row["k"] for row in coverage_rows], dtype=int)
    global_coverage = np.asarray([row["global_coverage"] for row in coverage_rows]) * 100.0
    local_coverage = np.asarray([row["country_local_coverage"] for row in coverage_rows]) * 100.0
    ax_coverage.fill_between(k, global_coverage, local_coverage, color=orange, alpha=0.12)
    ax_coverage.plot(k, local_coverage, marker="o", color=orange, linewidth=2.6, label="Country-specific catalog")
    ax_coverage.plot(k, global_coverage, marker="s", color=blue, linewidth=2.6, label="Same global catalog")
    ax_coverage.set_xticks(k)
    ax_coverage.set_ylim(45, 101)
    ax_coverage.set_xlabel("Language capabilities stored per region (k)")
    ax_coverage.set_ylabel("Held-out request coverage (%)")
    ax_coverage.grid(axis="y", alpha=0.22)
    ax_coverage.spines[["top", "right"]].set_visible(False)
    ax_coverage.legend(frameon=False, loc="lower right")
    ax_coverage.set_title(
        "B  Small local catalogs cover more future demand",
        loc="left",
        fontsize=13,
        weight="bold",
        pad=12,
    )
    best_row = max(coverage_rows, key=lambda row: row["gain_pp"])
    best_k = int(best_row["k"])
    best_local = float(best_row["country_local_coverage"]) * 100.0
    ax_coverage.annotate(
        f"+{best_row['gain_pp']:.1f} pp at k={best_k}",
        xy=(best_k, best_local),
        xytext=(min(best_k + 1.1, k[-1] - 1.1), max(best_local - 13.0, 55.0)),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9.5,
        color="#333333",
    )
    ax_coverage.text(
        0.02,
        -0.18,
        f"Chronological split: test requests after {cutoff:%Y-%m-%d}\n"
        f"{eligible_countries} countries; equal per-region catalog size",
        transform=ax_coverage.transAxes,
        fontsize=8.3,
        color="#555555",
        va="top",
        clip_on=False,
    )

    fig.suptitle(
        "WildChat-1M exhibits exploitable regional locality",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        f"{num_requests:,} geolocated conversations; country explains "
        f"{explained_entropy * 100:.1f}% of request-language entropy.",
        ha="center",
        fontsize=9.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96), w_pad=3.0)
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_gain(
    output_stem: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    cutoff: datetime,
    eligible_countries: int,
) -> None:
    """Plot the held-out coverage result as coverage and gain panels."""
    _configure_font()
    blue = "#2b6f9f"
    orange = "#d46b45"
    k = np.asarray([row["k"] for row in coverage_rows], dtype=int)
    global_coverage = np.asarray([row["global_coverage"] for row in coverage_rows]) * 100.0
    local_coverage = np.asarray([row["country_local_coverage"] for row in coverage_rows]) * 100.0
    gain = np.asarray([row["gain_pp"] for row in coverage_rows])

    fig, (ax_coverage, ax_gain) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax_coverage.fill_between(k, global_coverage, local_coverage, color=orange, alpha=0.12)
    ax_coverage.plot(
        k,
        local_coverage,
        marker="o",
        color=orange,
        linewidth=2.6,
        label="Country-specific catalog",
    )
    ax_coverage.plot(
        k,
        global_coverage,
        marker="s",
        color=blue,
        linewidth=2.6,
        label="Same global catalog",
    )
    ax_coverage.set_xticks(k)
    ax_coverage.set_ylim(60, 101)
    ax_coverage.set_xlabel("Language capabilities stored per region (k)")
    ax_coverage.set_ylabel("Held-out request coverage (%)")
    ax_coverage.set_title("A  Future request coverage", loc="left", fontsize=13, weight="bold", pad=10)
    ax_coverage.grid(axis="y", alpha=0.22)
    ax_coverage.spines[["top", "right"]].set_visible(False)
    ax_coverage.legend(frameon=False, loc="lower right")

    colors = [blue if value < 0 else orange for value in gain]
    bars = ax_gain.bar(k, gain, color=colors, width=0.68)
    ax_gain.axhline(0.0, color="#555555", linewidth=1.1)
    ax_gain.set_xticks(k)
    ax_gain.set_ylim(min(-6.0, float(gain.min()) - 2.0), float(gain.max()) + 4.0)
    ax_gain.set_xlabel("Language capabilities stored per region (k)")
    ax_gain.set_ylabel("Locality gain (percentage points)")
    ax_gain.set_title("B  Benefit of regional placement", loc="left", fontsize=13, weight="bold", pad=10)
    ax_gain.grid(axis="y", alpha=0.22)
    ax_gain.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, gain):
        vertical_alignment = "bottom" if value >= 0 else "top"
        offset = 0.55 if value >= 0 else -0.55
        ax_gain.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:+.1f}",
            ha="center",
            va=vertical_alignment,
            fontsize=9,
            color="#303030",
        )
    ax_gain.text(
        0.98,
        0.05,
        "Orange: local placement helps\nBlue: global placement helps",
        transform=ax_gain.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    fig.suptitle(
        "Regional placement helps once each region stores 2+ capabilities",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.015,
        f"Chronological holdout after {cutoff:%Y-%m-%d}; {eligible_countries} countries; "
        "equal per-region catalog size.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95), w_pad=3.0)
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    coverage_rows: Sequence[Mapping[str, Any]],
    countries: Sequence[str],
    languages: Sequence[str],
    shares: np.ndarray,
    log_lifts: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    with (output_dir / "heldout_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)
    with (output_dir / "country_language_locality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["country", "language", "country_share", "log2_lift"],
        )
        writer.writeheader()
        for row, country in enumerate(countries):
            for column, language in enumerate(languages):
                writer.writerow(
                    {
                        "country": country,
                        "language": language,
                        "country_share": float(shares[row, column]),
                        "log2_lift": float(log_lifts[row, column]),
                    }
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--num-countries", type=int, default=16)
    parser.add_argument("--num-languages", type=int, default=12)
    parser.add_argument("--min-train-requests", type=int, default=500)
    parser.add_argument("--min-test-requests", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between zero and one")
    countries, languages, timestamps = _load_columns(args.dataset)
    valid_timestamps = timestamps[timestamps != np.iinfo(np.int64).min]
    cutoff_us = int(np.quantile(valid_timestamps, args.train_fraction, method="higher"))
    train_mask = timestamps < cutoff_us
    test_mask = timestamps >= cutoff_us

    full_country, full_language, full_joint = _count_pairs(countries, languages)
    train_country, train_language, train_joint = _count_pairs(countries, languages, train_mask)
    test_country, _, test_joint = _count_pairs(countries, languages, test_mask)
    mutual_information, language_entropy, explained_entropy = _explained_language_entropy(
        full_country, full_language, full_joint
    )
    coverage_rows, eligible = _top_k_coverage(
        train_country,
        train_language,
        train_joint,
        test_country,
        test_joint,
        args.max_k,
        args.min_train_requests,
        args.min_test_requests,
    )
    heatmap_countries, heatmap_languages, shares, log_lifts = _heatmap_data(
        full_country,
        full_language,
        full_joint,
        args.num_countries,
        args.num_languages,
    )
    cutoff = datetime.fromtimestamp(cutoff_us / 1_000_000, tz=timezone.utc)
    summary: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "total_conversations": len(countries),
        "geolocated_labeled_conversations": sum(full_joint.values()),
        "countries": len(full_country),
        "languages": len(full_language),
        "train_fraction": args.train_fraction,
        "cutoff_utc": cutoff.isoformat(),
        "train_requests": sum(train_joint.values()),
        "test_requests": sum(test_joint.values()),
        "eligible_countries": len(eligible),
        "eligible_country_names": eligible,
        "mutual_information_nats": mutual_information,
        "language_entropy_nats": language_entropy,
        "explained_language_entropy": explained_entropy,
        "coverage": coverage_rows,
    }
    _write_outputs(
        args.output_dir,
        summary,
        coverage_rows,
        heatmap_countries,
        heatmap_languages,
        shares,
        log_lifts,
    )
    _plot(
        args.output_dir / "wildchat_regional_locality",
        heatmap_countries,
        heatmap_languages,
        shares,
        log_lifts,
        coverage_rows,
        explained_entropy,
        sum(full_joint.values()),
        cutoff,
        len(eligible),
    )
    _plot_coverage_gain(
        args.output_dir / "wildchat_locality_coverage_gain",
        coverage_rows,
        cutoff,
        len(eligible),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
