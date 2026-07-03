#!/usr/bin/env bash
set -euo pipefail

# Edit this path for the machine you are preparing.
# TARGET_ROOT="/mnt/8T/whl/huggingface"
TARGET_ROOT="/data/huggingface"

MODEL_DIR="${TARGET_ROOT}/model"
DATASET_DIR="${TARGET_ROOT}/dataset"
RAW_DATASET_DIR="${DATASET_DIR}/raw_hf_parquet"

# The codebase reads datasets with datasets.load_from_disk(...). Keep this on
# unless you only want raw Hugging Face repo snapshots.
CONVERT_DATASETS_TO_LOAD_FROM_DISK=1

# Set to 1 to rebuild existing load_from_disk dataset directories.
OVERWRITE_DATASETS=0

HF_BIN="${HF_BIN:-hf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

download_model() {
  local repo_id="$1"
  local local_name="$2"
  local target_dir="${MODEL_DIR}/${local_name}"

  mkdir -p "${target_dir}"
  echo "Downloading model ${repo_id} -> ${target_dir}"
  "${HF_BIN}" download "${repo_id}" \
    --repo-type model \
    --local-dir "${target_dir}"
}

download_dataset_repo() {
  local repo_id="$1"
  local local_name="$2"
  local target_dir="${RAW_DATASET_DIR}/${local_name}"

  mkdir -p "${target_dir}"
  echo "Downloading dataset repo ${repo_id} -> ${target_dir}"
  "${HF_BIN}" download "${repo_id}" \
    --repo-type dataset \
    --local-dir "${target_dir}"
}

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Required command not found: ${name}" >&2
    exit 1
  fi
}

main() {
  require_command "${HF_BIN}"
  require_command "${PYTHON_BIN}"

  mkdir -p "${MODEL_DIR}" "${DATASET_DIR}" "${RAW_DATASET_DIR}"

  download_model "Qwen/Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct"
  download_model "Qwen/Qwen2.5-1.5B-Instruct" "Qwen2.5-1.5B-Instruct"
  download_model "QWen/Qwen3.5-0.8B" "Qwen3.5-0.8B"
  download_model "QWen/Qwen3.5-2B" "Qwen3.5-2B"
  download_model "QWen/Qwen3.5-9B" "Qwen3.5-9B"

  download_dataset_repo "openai/gsm8k" "gsm8k"
  download_dataset_repo "nyu-mll/glue" "glue"
  download_dataset_repo "rajpurkar/squad" "squad"
  download_dataset_repo "abisee/cnn_dailymail" "cnn_dailymail"
  download_dataset_repo "google-research-datasets/mbpp" "mbpp_sanitized"

  if [[ "${CONVERT_DATASETS_TO_LOAD_FROM_DISK}" == "1" ]]; then
    echo "Converting datasets to datasets.load_from_disk layout under ${DATASET_DIR}"
    OVERWRITE_DATASETS="${OVERWRITE_DATASETS}" "${PYTHON_BIN}" - "${DATASET_DIR}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Python package 'datasets' is required for conversion. "
        "Install it or set CONVERT_DATASETS_TO_LOAD_FROM_DISK=0."
    ) from exc

dataset_dir = Path(sys.argv[1])
overwrite = os.environ.get("OVERWRITE_DATASETS") == "1"

specs = [
    ("openai/gsm8k", "main", "gsm8k"),
    ("nyu-mll/glue", "cola", "glue_cola"),
    ("nyu-mll/glue", "sst2", "glue_sst2"),
    ("rajpurkar/squad", None, "squad"),
    ("abisee/cnn_dailymail", "3.0.0", "cnn_dailymail"),
    ("google-research-datasets/mbpp", "sanitized", "mbpp"),
]

for repo_id, config_name, local_name in specs:
    target = dataset_dir / local_name
    if target.exists() and not overwrite:
        print(f"Keeping existing dataset {target}")
        continue

    tmp_target = dataset_dir / f".{local_name}.tmp"
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    if target.exists():
        shutil.rmtree(target)

    label = repo_id if config_name is None else f"{repo_id}/{config_name}"
    print(f"Preparing {label} -> {target}")
    if config_name is None:
        dataset = load_dataset(repo_id)
    else:
        dataset = load_dataset(repo_id, config_name)
    dataset.save_to_disk(str(tmp_target))
    tmp_target.rename(target)
PY
  fi

  echo "Done."
  echo "Models: ${MODEL_DIR}"
  echo "Datasets: ${DATASET_DIR}"
  echo "Raw dataset snapshots: ${RAW_DATASET_DIR}"
}

main "$@"
