"""Command line entry point for the IDN experiment pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from idn.config import DEFAULT_CAPABILITY_DIR, DEFAULT_MACHINE_CONFIG, ConfigError
from idn.registry import iter_capabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idn")
    parser.add_argument(
        "--machine",
        default=str(DEFAULT_MACHINE_CONFIG),
        help="Machine YAML merged over capability config.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-capabilities",
        help="List configured capabilities and resolved local paths.",
    )
    list_parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CAPABILITY_DIR),
        help="Directory containing capability YAML files.",
    )
    list_parser.set_defaults(func=cmd_list_capabilities)

    for name in (
        "gen-teacher",
        "build-sft",
        "train-lora",
        "eval",
        "sweep",
        "export",
        "serve",
        "report",
    ):
        stub = subparsers.add_parser(name, help="{0} (not implemented in Phase 0)".format(name))
        stub.set_defaults(func=cmd_not_implemented)

    return parser


def cmd_list_capabilities(args: argparse.Namespace) -> int:
    rows = list(
        iter_capabilities(
            config_dir=Path(args.config_dir),
            machine_path=Path(args.machine),
        )
    )
    if not rows:
        print("No capabilities found.")
        return 0

    print(
        "{:<14} {:<16} {:<11} {:<42} {:<42}".format(
            "capability", "metric", "eval_split", "dataset_path", "student_model_path"
        )
    )
    print("-" * 132)
    for row in rows:
        print(
            "{:<14} {:<16} {:<11} {:<42} {:<42}".format(
                row.id,
                row.config.metric_name,
                row.config.eval_split,
                row.config.dataset_path,
                row.machine.student_model_path,
            )
        )
    print("\nteacher_model_path: {0}".format(rows[0].machine.teacher_model_path))
    print("machine: {0}".format(rows[0].machine.name))
    return 0


def cmd_not_implemented(args: argparse.Namespace) -> int:
    raise SystemExit(
        "Command '{0}' is planned but not implemented in Phase 0.".format(args.command)
    )


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        func = getattr(args, "func")
        return int(func(args))
    except ConfigError as exc:
        print("config error: {0}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

