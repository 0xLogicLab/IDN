#!/usr/bin/env python3
"""Build shared basis and residual LoRA adapters from capability specialists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idn.compose.shared_basis import SharedBasisError, build_shared_basis


CAPABILITIES = ("cnn_dailymail", "cola", "gsm8k", "mbpp", "squad", "sst2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose six capability LoRAs into shared bases and residuals."
    )
    parser.add_argument("--adapter-root", default="artifacts/adapters")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--gsm8k-adapter",
        default=None,
        help="Override the GSM8K adapter path while keeping other adapters under --adapter-root.",
    )
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--output", default="artifacts/shared_basis")
    parser.add_argument(
        "--output-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--override", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rank <= 0:
        raise SystemExit("--rank must be positive")
    adapter_paths = {
        capability: Path(args.adapter_root) / capability / "r{0}".format(args.rank)
        for capability in CAPABILITIES
    }
    if args.gsm8k_adapter:
        adapter_paths["gsm8k"] = Path(args.gsm8k_adapter)
    try:
        manifest = build_shared_basis(
            adapter_paths=adapter_paths,
            output_dir=Path(args.output),
            components=args.components,
            output_dtype=args.output_dtype,
            override=args.override,
        )
    except SharedBasisError as exc:
        raise SystemExit("shared-basis error: {0}".format(exc)) from exc
    print("Shared-basis decomposition complete")
    print("components: {0}".format(manifest["components"]))
    print("materialized_rank: {0}".format(manifest["materialized_rank"]))
    print(
        "retained_energy_ratio: {0:.6f}".format(
            manifest["decomposition"]["retained_energy_ratio"]
        )
    )
    print("manifest: {0}".format(Path(args.output) / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
