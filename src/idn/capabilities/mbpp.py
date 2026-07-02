"""MBPP Python programming capability."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from idn.capabilities.base import Example
from idn.capabilities.classification import DatasetLoadError
from idn.config import CapabilityConfig


CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.I | re.S)


@dataclass
class MBPPSpec:
    """CapabilitySpec implementation for MBPP."""

    config: CapabilityConfig
    _pass_cache: Dict[Tuple[str, Tuple[str, ...], str], bool] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def metric_name(self) -> str:
        return self.config.metric_name

    @property
    def higher_is_better(self) -> bool:
        return self.config.higher_is_better

    def load_split(self, split: str, limit: Optional[int] = None) -> List[Example]:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise DatasetLoadError(
                "datasets is required to load capability data"
            ) from exc

        dataset_dict = load_from_disk(self.config.dataset_path)
        dataset = dataset_dict[split]
        examples = []
        for index, row in enumerate(dataset):
            prompt = str(row.get("prompt", row.get("text", "")))
            tests = _string_list(row.get("test_list", []))
            setup_code = str(row.get("test_setup_code", "") or "")
            examples.append(
                Example(
                    id=str(row.get("task_id", row.get("id", index))),
                    prompt_inputs={
                        "prompt": prompt,
                        "tests": tests,
                        "test_setup_code": setup_code,
                    },
                    reference={
                        "code": str(row.get("code", "") or ""),
                        "tests": tests,
                        "test_setup_code": setup_code,
                    },
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def format_prompt(self, ex: Example) -> List[Dict[str, str]]:
        tests = "\n".join(ex.prompt_inputs.get("tests", []))
        content = (
            "{template}\n\nPrompt:\n{prompt}\n\nTests:\n{tests}\n\n"
            "Return only Python code."
        ).format(
            template=self.config.prompt_template,
            prompt=ex.prompt_inputs["prompt"],
            tests=tests,
        )
        return [
            {
                "role": "system",
                "content": "You write correct, minimal Python solutions.",
            },
            {"role": "user", "content": content},
        ]

    def parse_prediction(self, raw_text: str) -> str:
        return extract_python_code(raw_text)

    def target_text(self, ex: Example, teacher_text: Optional[str] = None) -> str:
        if teacher_text:
            code = self.parse_prediction(teacher_text)
            if code:
                return code
        return str(ex.reference.get("code", ""))

    def metric(self, preds: List[Any], refs: List[Any]) -> Dict[str, float]:
        if len(preds) != len(refs):
            raise ValueError("preds and refs must have the same length")
        if not refs:
            return {"pass@1": 0.0}
        passed = [
            self._passes_tests(
                str(pred or ""),
                _string_list(ref.get("tests", []) if isinstance(ref, dict) else []),
                str(ref.get("test_setup_code", "") if isinstance(ref, dict) else ""),
            )
            for pred, ref in zip(preds, refs)
        ]
        return {"pass@1": sum(int(item) for item in passed) / len(passed)}

    def verify(self, raw_text: str, parsed: Any) -> bool:
        return bool(parsed)

    def _passes_tests(self, code: str, tests: List[str], setup_code: str) -> bool:
        key = (code, tuple(tests), setup_code)
        if key not in self._pass_cache:
            self._pass_cache[key] = run_python_tests(code, tests, setup_code)
        return self._pass_cache[key]


def extract_python_code(raw_text: Any) -> str:
    if raw_text is None:
        return ""
    text = str(raw_text).strip()
    matches = CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    text = re.sub(r"^(python code|solution|answer)\s*:\s*", "", text, flags=re.I).strip()
    return text.strip("` \n\t")


def run_python_tests(
    code: str,
    tests: List[str],
    setup_code: str = "",
    timeout: float = 5.0,
) -> bool:
    if not code.strip() or not tests:
        return False
    script = "\n".join(
        [
            "import math",
            "import re",
            "import itertools",
            "import functools",
            "import collections",
            "import heapq",
            "import bisect",
            "import string",
            setup_code,
            code,
            "",
            "\n".join(tests),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="idn_mbpp_") as tmpdir:
        path = os.path.join(tmpdir, "candidate.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(script)
        return _run_candidate(path=path, cwd=tmpdir, timeout=timeout)


def _run_candidate(path: str, cwd: str, timeout: float) -> bool:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, path],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout) == 0
        except subprocess.TimeoutExpired:
            _kill_process(process)
            return False
    except OSError:
        return False


def _kill_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, 9)
    except OSError:
        process.kill()
    process.wait()


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _string_list(parsed)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        return _string_list(value.tolist())
    return [str(value)]
