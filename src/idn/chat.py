"""Interactive chat CLI for local causal language models."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from idn.models.loader import ChatModel, ModelLoadError, validate_dtype


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idn-chat")
    parser.add_argument(
        "--model",
        required=True,
        help="Local model path to load.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter path.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="bfloat16 or float32. fp16 is intentionally rejected.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per assistant turn.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt. Use an empty string to omit it.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not include prior turns in each prompt.",
    )
    parser.add_argument(
        "--allow-hub",
        action="store_true",
        help="Allow loading from the Hugging Face Hub instead of local files only.",
    )
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_dtype(args.dtype)
        chat_model = ChatModel(
            model_path=args.model,
            adapter_path=args.adapter,
            dtype=args.dtype,
            device=args.device,
            local_files_only=not args.allow_hub,
        )
        print("Loading model: {0}".format(args.model))
        if args.adapter:
            print("Loading adapter: {0}".format(args.adapter))
        chat_model.load()
    except ModelLoadError as exc:
        print("model error: {0}".format(exc), file=sys.stderr)
        return 2

    print("Ready. Type /exit to quit, /reset to clear history.")
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user_text = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_text:
            continue
        command = user_text.lower()
        if command in ("/exit", "/quit"):
            return 0
        if command == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("History cleared.")
            continue

        prompt_messages = list(messages)
        prompt_messages.append({"role": "user", "content": user_text})
        print("assistant> ", end="", flush=True)
        chunks = []
        try:
            for chunk in chat_model.stream_generate(
                messages=prompt_messages,
                max_new_tokens=args.max_new_tokens,
            ):
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        except ModelLoadError as exc:
            print("")
            print("generation error: {0}".format(exc), file=sys.stderr)
            return 2
        print("")

        response = "".join(chunks).strip()
        if not args.no_history:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": response})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
