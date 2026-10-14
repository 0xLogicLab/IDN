#!/usr/bin/env python3
"""Run a chat dataset through a user-supplied IDN runtime factory.

The factory is an import path such as ``my_deployment:create_runtime`` and must
return a configured ``IDNRuntime``. This keeps model/router setup deployment-
specific while sharing one benchmark implementation. Use ``--dataset-factory``
for a non-JSONL dataset adapter implementing the benchmark dataset contract.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from idn.capabilities.jsonl_chat import JsonlChatDataset
from idn.eval.runtime_benchmark import run_runtime_benchmark
from idn.framework.types import ResourceBudget


def _load_factory(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("runtime factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise TypeError("runtime factory is not callable: {0}".format(spec))
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None,
                        help="JSONL split path when using the built-in adapter")
    parser.add_argument("--dataset-factory", default=None,
                        help="optional module:function(args) returning a dataset adapter")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-id", default="wildchat")
    parser.add_argument("--runtime-factory", required=True,
                        help="Python import path module:function returning IDNRuntime")
    parser.add_argument("--quality-scorer", default=None,
                        help="optional module:function(example, prediction) -> score mapping")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-examples", type=int, default=4)
    parser.add_argument("--max-memory-gib", type=float, default=None)
    parser.add_argument("--max-latency-ms", type=float, default=None)
    parser.add_argument("--min-quality-lower-bound", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "outputs" / "idn_benchmark" / timestamp
    )
    if args.dataset_factory:
        dataset = _load_factory(args.dataset_factory)(args)
    else:
        if not args.data:
            parser.error("--data is required unless --dataset-factory is provided")
        dataset = JsonlChatDataset(
            split_files={args.split: args.data}, dataset_id=args.dataset_id
        )
    runtime = _load_factory(args.runtime_factory)()
    quality_scorer = (
        _load_factory(args.quality_scorer) if args.quality_scorer else None
    )
    run_runtime_benchmark(
        dataset=dataset,
        runtime=runtime,
        split=args.split,
        output_dir=output_dir,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        warmup_examples=args.warmup_examples,
        quality_scorer=quality_scorer,
        budget=ResourceBudget(
            max_memory_gib=args.max_memory_gib,
            max_latency_ms=args.max_latency_ms,
            min_quality_lower_bound=args.min_quality_lower_bound,
        ),
        metadata={
            "runtime_factory": args.runtime_factory,
            "dataset_factory": args.dataset_factory or "builtin_jsonl_chat",
            "quality_scorer": args.quality_scorer,
        },
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
