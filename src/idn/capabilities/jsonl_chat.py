"""Generic JSONL conversation adapter for WildChat-like corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from idn.capabilities.base import Example, PromptDataset


class JsonlChatDataset:
    """Turn conversation JSONL files into prompt/target examples.

    Each split file must contain one JSON object per line. Records may store a
    conversation in ``messages`` or ``conversation``. If message objects have
    a truthy ``loss`` field, only those assistant messages become targets;
    otherwise every assistant turn with a preceding user message is emitted.
    """

    def __init__(
        self,
        *,
        split_files: Mapping[str, Union[str, Path]],
        dataset_id: str = "jsonl_chat",
        capability_field: str = "capability",
        capability_map: Optional[Mapping[str, str]] = None,
        capability_filter: Optional[Sequence[str]] = None,
    ) -> None:
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")
        if not split_files:
            raise ValueError("split_files cannot be empty")
        self.id = dataset_id
        self.split_files = {name: Path(path) for name, path in split_files.items()}
        self.capability_field = capability_field
        self.capability_map = dict(capability_map or {})
        self.capability_filter = (
            set(str(item) for item in capability_filter)
            if capability_filter is not None
            else None
        )

    def load_split(self, split: str, limit: Optional[int] = None) -> List[Example]:
        if split not in self.split_files:
            raise ValueError(
                "unknown split {0}; available: {1}".format(
                    split, ", ".join(sorted(self.split_files))
                )
            )
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided")

        path = self.split_files[split]
        if not path.is_file():
            raise FileNotFoundError("conversation split does not exist: {0}".format(path))

        examples: List[Example] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid JSON at {0}:{1}: {2}".format(path, line_number, exc)
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(
                        "expected a JSON object at {0}:{1}".format(path, line_number)
                    )
                examples.extend(self._expand_record(row, line_number))
                if limit is not None and len(examples) >= limit:
                    return examples[:limit]
        return examples

    def format_prompt(self, ex: Example) -> List[Dict[str, str]]:
        messages = ex.prompt_inputs.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("conversation example has no prompt messages")
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]

    def target_text(self, ex: Example, teacher_text: Optional[str] = None) -> str:
        if teacher_text is not None and teacher_text.strip():
            return teacher_text.strip()
        if ex.gold_target is None:
            raise ValueError("conversation example has no assistant target")
        return ex.gold_target

    def parse_prediction(self, raw_text: str) -> str:
        return str(raw_text).strip()

    def verify(self, raw_text: str, parsed: Any) -> bool:
        # This checks only that a response exists; it is not a quality metric.
        return bool(str(raw_text).strip() and str(parsed).strip())

    def _expand_record(self, row: Mapping[str, Any], line_number: int) -> List[Example]:
        raw_messages = row.get("messages", row.get("conversation"))
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid conversation JSON at line {0}: {1}".format(line_number, exc)
                ) from exc
        if not isinstance(raw_messages, list):
            return []

        messages = []
        for item in raw_messages:
            normalized = _normalize_message(item)
            if normalized is not None:
                messages.append(normalized)

        raw_capability = row.get(self.capability_field, self.id)
        capability = self.capability_map.get(str(raw_capability), str(raw_capability))
        if self.capability_filter is not None and capability not in self.capability_filter:
            return []

        has_loss_flags = any("loss" in item for item in raw_messages if isinstance(item, Mapping))
        target_indices = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "assistant"
            and message["content"].strip()
            and (not has_loss_flags or message.get("loss", False))
        ]

        record_id = str(row.get("id", row.get("conversation_id", line_number)))
        examples = []
        for turn_index, target_index in enumerate(target_indices):
            context = messages[:target_index]
            if not any(message["role"] == "user" for message in context):
                continue
            target = messages[target_index]["content"].strip()
            prompt_messages = [
                {"role": message["role"], "content": message["content"]}
                for message in context
                if message["role"] in ("system", "developer", "user", "assistant")
            ]
            if not prompt_messages:
                continue
            examples.append(
                Example(
                    id="{0}-turn-{1}".format(record_id, turn_index),
                    prompt_inputs={
                        "messages": prompt_messages,
                        "capability": capability,
                        "source_line": line_number,
                    },
                    reference={"text": target, "capability": capability},
                    gold_target=target,
                )
            )
        return examples


def _normalize_message(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    role = value.get("role") or value.get("from") or value.get("speaker")
    content = value.get("content") or value.get("value") or value.get("text")
    if role is None or content is None:
        return None
    role = str(role).lower()
    role = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
    }.get(role, role)
    if role not in ("system", "developer", "user", "assistant", "tool"):
        return None
    return {
        "role": role,
        "content": str(content),
        "loss": bool(value.get("loss", False)),
    }
