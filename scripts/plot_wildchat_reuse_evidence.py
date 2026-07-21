#!/usr/bin/env python3
"""Measure and plot cross-user request reuse evidence in WildChat-1M."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict, load_from_disk
from matplotlib import font_manager


DEFAULT_DATASET = Path("/data/huggingface/dataset/wildchat_1m")
DEFAULT_REGIONAL_SUMMARY = Path("artifacts/wildchat_locality/summary.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/wildchat_reuse")
WINDOWS: Sequence[Tuple[str, int | None]] = (
    ("1 hour", 60 * 60),
    ("1 day", 24 * 60 * 60),
    ("7 days", 7 * 24 * 60 * 60),
    ("30 days", 30 * 24 * 60 * 60),
    ("Any earlier", None),
)
WHITESPACE = re.compile(r"\s+")


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
            "font.size": 9.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_dataset(path: Path) -> Dataset:
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        return loaded["train"]
    return loaded


def _prompt_key(text: str) -> Tuple[bytes, int]:
    normalized = WHITESPACE.sub(
        " ", unicodedata.normalize("NFKC", text).casefold()
    ).strip()
    key = hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()
    return key, len(normalized)


def _measure_reuse(
    dataset: Dataset,
    min_prompt_chars: int,
    batch_size: int,
) -> Dict[str, Any]:
    table = dataset.data.table.select(
        ["timestamp", "hashed_ip", "country", "redacted", "conversation"]
    )
    # Values are count, first user, shared-across-users flag, latest event, and
    # latest event from a different user than the latest event.
    states: MutableMapping[bytes, List[Any]] = {}
    country_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    unique_users = set()
    eligible_requests = 0
    same_country_reuse = 0
    different_country_reuse = 0
    last_timestamp = -1

    for record_batch in table.to_batches(max_chunksize=batch_size):
        timestamps = record_batch.column(0).cast("int64").to_pylist()
        users = record_batch.column(1).to_pylist()
        countries = record_batch.column(2).to_pylist()
        redacted = record_batch.column(3).to_pylist()
        first_turn = pc.list_element(record_batch.column(4), 0)
        roles = pc.struct_field(first_turn, "role").to_pylist()
        texts = pc.struct_field(first_turn, "content").to_pylist()

        for timestamp, user, country, is_redacted, role, text in zip(
            timestamps, users, countries, redacted, roles, texts
        ):
            if (
                timestamp is None
                or not user
                or not text
                or is_redacted
                or role != "user"
            ):
                continue
            key, prompt_length = _prompt_key(str(text))
            if prompt_length < min_prompt_chars:
                continue
            if timestamp < last_timestamp:
                raise ValueError("WildChat rows must be chronological for reuse windows")
            last_timestamp = timestamp
            eligible_requests += 1
            unique_users.add(str(user))
            if country:
                country_counts[str(country)] += 1

            state = states.get(key)
            prior: Tuple[int, str, str | None] | None = None
            if state is None:
                # count, first_user, cross_user, latest_ts/user/country,
                # second_latest_ts/user/country
                states[key] = [
                    1,
                    str(user),
                    False,
                    timestamp,
                    str(user),
                    country,
                    None,
                    None,
                    None,
                ]
                continue

            state[0] += 1
            if user != state[1]:
                state[2] = True
            if user != state[4]:
                prior = (state[3], state[4], state[5])
            elif state[7] is not None:
                prior = (state[6], state[7], state[8])

            if prior is not None:
                gap_seconds = (timestamp - prior[0]) / 1_000_000
                window_counts["Any earlier"] += 1
                for label, limit_seconds in WINDOWS:
                    if limit_seconds is not None and gap_seconds <= limit_seconds:
                        window_counts[label] += 1
                if country and prior[2] and country == prior[2]:
                    same_country_reuse += 1
                elif country and prior[2]:
                    different_country_reuse += 1

            if user == state[4]:
                state[3] = timestamp
                state[5] = country
            else:
                old_latest = state[3:6]
                state[3:6] = [timestamp, str(user), country]
                state[6:9] = old_latest

    shared_group_sizes = sorted(
        (int(state[0]) for state in states.values() if state[2]), reverse=True
    )
    shared_requests = sum(shared_group_sizes)
    geolocated_reuse = same_country_reuse + different_country_reuse
    geolocated_requests = sum(country_counts.values())
    random_same_country = sum(
        (count / geolocated_requests) ** 2 for count in country_counts.values()
    )
    return {
        "dataset_rows": len(dataset),
        "eligible_requests": eligible_requests,
        "unique_hashed_users": len(unique_users),
        "min_prompt_chars": min_prompt_chars,
        "normalization": "NFKC + casefold + whitespace collapse; BLAKE2b-128 key",
        "cross_user_duplicate_groups": len(shared_group_sizes),
        "requests_in_cross_user_duplicate_groups": shared_requests,
        "requests_after_excluding_top_100_groups": sum(shared_group_sizes[100:]),
        "top_100_group_request_share": (
            sum(shared_group_sizes[:100]) / shared_requests if shared_requests else 0.0
        ),
        "reuse_windows": {
            label: {
                "requests": window_counts[label],
                "share_of_eligible_requests": window_counts[label] / eligible_requests,
            }
            for label, _ in WINDOWS
        },
        "geographic_reuse": {
            "geolocated_cross_user_reuse_events": geolocated_reuse,
            "same_country_events": same_country_reuse,
            "different_country_events": different_country_reuse,
            "same_country_share": (
                same_country_reuse / geolocated_reuse if geolocated_reuse else 0.0
            ),
            "unconditioned_random_pair_same_country_share": random_same_country,
        },
    }


def _load_regional_summary(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("coverage"):
        raise ValueError(f"regional summary has no coverage rows: {path}")
    return summary


def _plot(
    output_stem: Path,
    reuse: Mapping[str, Any],
    regional: Mapping[str, Any],
) -> None:
    _configure_font()
    teal = "#167D7F"
    coral = "#D46745"
    blue = "#356B9A"
    gold = "#C79A32"
    ink = "#252A31"
    muted = "#687078"
    grid = "#D9DEE2"

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 5.25),
        gridspec_kw={"width_ratios": [1.12, 0.82, 1.38]},
    )
    ax_time, ax_geo, ax_coverage = axes

    labels = [label for label, _ in WINDOWS]
    counts = np.asarray(
        [reuse["reuse_windows"][label]["requests"] for label in labels]
    )
    rates = counts / int(reuse["eligible_requests"]) * 100.0
    positions = np.arange(len(labels))
    colors = ["#8CC4C2", "#63AAA9", "#3B9292", teal, ink]
    bars = ax_time.barh(positions, rates, color=colors, height=0.68)
    ax_time.set_yticks(positions, labels)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Share of substantive requests (%)")
    ax_time.set_xlim(0.0, max(4.15, float(rates.max()) + 0.65))
    ax_time.set_title(
        "A  Exact requests recur quickly",
        loc="left",
        fontsize=12.5,
        weight="bold",
        pad=11,
    )
    for bar, rate, count in zip(bars, rates, counts):
        ax_time.text(
            rate + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.2f}%  ({count:,})",
            va="center",
            fontsize=8.8,
            color=ink,
        )
    ax_time.grid(axis="x", color=grid, linewidth=0.8)
    ax_time.set_axisbelow(True)

    geographic = reuse["geographic_reuse"]
    geo_values = np.asarray(
        [
            geographic["same_country_share"],
            geographic["unconditioned_random_pair_same_country_share"],
        ]
    ) * 100.0
    geo_bars = ax_geo.bar(
        [0, 1],
        geo_values,
        color=[coral, gold],
        width=0.62,
    )
    ax_geo.set_xticks(
        [0, 1],
        ["Exact reuse\nevents", "Random request\npairs"],
    )
    ax_geo.set_ylabel("Same-country share (%)")
    ax_geo.set_ylim(0, 86)
    ax_geo.set_title(
        "B  Reuse is geographically local",
        loc="left",
        fontsize=12.5,
        weight="bold",
        pad=11,
    )
    for bar, value in zip(geo_bars, geo_values):
        ax_geo.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color=ink,
        )
    ax_geo.text(
        0.97,
        0.56,
        "Reference is unconditioned;\nlanguage partly explains the gap.",
        transform=ax_geo.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color=muted,
    )
    ax_geo.grid(axis="y", color=grid, linewidth=0.8)
    ax_geo.set_axisbelow(True)

    coverage = regional["coverage"]
    k = np.asarray([row["k"] for row in coverage], dtype=int)
    global_coverage = np.asarray([row["global_coverage"] for row in coverage]) * 100
    local_coverage = np.asarray(
        [row["country_local_coverage"] for row in coverage]
    ) * 100
    ax_coverage.fill_between(
        k, global_coverage, local_coverage, color=coral, alpha=0.12
    )
    ax_coverage.plot(
        k,
        local_coverage,
        color=coral,
        marker="o",
        linewidth=2.5,
        label="Country-specific catalog",
    )
    ax_coverage.plot(
        k,
        global_coverage,
        color=blue,
        marker="s",
        linewidth=2.5,
        label="Same global catalog",
    )
    ax_coverage.set_xticks(k)
    ax_coverage.set_ylim(64, 100.5)
    ax_coverage.set_xlabel("Language capabilities stored per region (k)")
    ax_coverage.set_ylabel("Held-out request coverage (%)")
    ax_coverage.set_title(
        "C  Local catalogs cover future demand",
        loc="left",
        fontsize=12.5,
        weight="bold",
        pad=11,
    )
    ax_coverage.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax_coverage.grid(axis="y", color=grid, linewidth=0.8)
    ax_coverage.set_axisbelow(True)
    best = max(coverage, key=lambda row: row["gain_pp"])
    best_k = int(best["k"])
    best_y = float(best["country_local_coverage"]) * 100
    ax_coverage.annotate(
        f"+{best['gain_pp']:.1f} pp at k={best_k}",
        xy=(best_k, best_y),
        xytext=(best_k + 0.7, best_y - 10),
        arrowprops={"arrowstyle": "->", "color": muted, "linewidth": 1.0},
        fontsize=9,
        color=ink,
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AAB1B7")
        ax.tick_params(color="#AAB1B7", labelcolor=ink)

    fig.suptitle(
        "WildChat-1M shows demand for reusable capabilities",
        fontsize=16,
        weight="bold",
        color=ink,
        y=0.99,
    )
    fig.text(
        0.5,
        0.935,
        "Exact cross-user recurrence provides a lower bound; regional language demand predicts future capability placement.",
        ha="center",
        fontsize=10,
        color=muted,
    )
    fig.text(
        0.5,
        0.012,
        f"First user request per conversation; non-redacted; >= {reuse['min_prompt_chars']} normalized characters; "
        f"n={reuse['eligible_requests']:,}. User identity is approximated by hashed IP. "
        f"Panel C: chronological 80/20 split across {regional['eligible_countries']} eligible countries.",
        ha="center",
        fontsize=8.2,
        color=muted,
    )
    fig.tight_layout(rect=(0.025, 0.07, 0.99, 0.91), w_pad=2.8)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--regional-summary", type=Path, default=DEFAULT_REGIONAL_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-prompt-chars", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.min_prompt_chars < 1:
        raise ValueError("--min-prompt-chars must be positive")
    dataset = _load_dataset(args.dataset)
    reuse = _measure_reuse(dataset, args.min_prompt_chars, args.batch_size)
    regional = _load_regional_summary(args.regional_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "reuse": reuse,
        "regional_summary": str(args.regional_summary),
        "regional": {
            "cutoff_utc": regional["cutoff_utc"],
            "eligible_countries": regional["eligible_countries"],
            "explained_language_entropy": regional["explained_language_entropy"],
            "coverage": regional["coverage"],
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    _plot(args.output_dir / "wildchat_reuse_evidence", reuse, regional)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
