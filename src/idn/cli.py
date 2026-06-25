from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from idn.data import load_eval_examples
from idn.eval import char_similarity, summarize, token_f1
from idn.models import ChatModel, GenerationConfig
from idn.utils import write_json, write_jsonl


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_models(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.num_samples is not None:
        cfg["num_samples"] = args.num_samples
    if args.output_path is not None:
        cfg["output_path"] = args.output_path
    if args.summary_path is not None:
        cfg["summary_path"] = args.summary_path

    examples = load_eval_examples(
        dataset_name=cfg["dataset_name"],
        split=cfg.get("dataset_split", "train"),
        num_samples=int(cfg.get("num_samples", 20)),
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=int(cfg.get("max_new_tokens", 160)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
    )

    max_memory = cfg.get("max_memory")
    records: list[dict[str, Any]] = [
        {
            "request_id": ex.request_id,
            "source": ex.source,
            "prompt": ex.prompt,
            "reference": ex.reference,
        }
        for ex in examples
    ]

    print(f"Loading base model: {cfg['base_model']}")
    base = ChatModel(
        cfg["base_model"],
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        device_map=cfg.get("device_map", "auto"),
        max_memory=max_memory,
    )
    for record in tqdm(records, desc="Base model"):
        record["base"] = {"model": cfg["base_model"], **base.generate(record["prompt"], gen_cfg)}

    del base
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"Loading teacher model: {cfg['teacher_model']}")
    teacher = ChatModel(
        cfg["teacher_model"],
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        device_map=cfg.get("device_map", "auto"),
        max_memory=max_memory,
    )
    for record in tqdm(records, desc="Teacher model"):
        record["teacher"] = {"model": cfg["teacher_model"], **teacher.generate(record["prompt"], gen_cfg)}

    for record in records:
        metrics: dict[str, Any] = {
            "base_teacher_char_similarity": char_similarity(record["base"]["text"], record["teacher"]["text"]),
            "base_teacher_token_f1": token_f1(record["base"]["text"], record["teacher"]["text"]),
        }
        if record.get("reference"):
            metrics["base_reference_char_similarity"] = char_similarity(record["base"]["text"], record["reference"])
            metrics["teacher_reference_char_similarity"] = char_similarity(record["teacher"]["text"], record["reference"])
        record["metrics"] = metrics

    summary = summarize(records)
    summary.update(
        {
            "teacher_model": cfg["teacher_model"],
            "base_model": cfg["base_model"],
            "dataset_name": cfg["dataset_name"],
            "dataset_split": cfg.get("dataset_split", "train"),
            "generation": gen_cfg.__dict__,
        }
    )
    write_jsonl(cfg["output_path"], records)
    write_json(cfg["summary_path"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if cfg.get("force_exit", False):
        import os

        os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(prog="idn-cap")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval-models", help="Run teacher/base model comparison on a dataset")
    p_eval.add_argument("--config", default="configs/qwen25_eval.yaml")
    p_eval.add_argument("--num-samples", type=int)
    p_eval.add_argument("--output-path")
    p_eval.add_argument("--summary-path")
    p_eval.set_defaults(func=eval_models)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
