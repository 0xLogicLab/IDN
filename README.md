# Intelligence Delivery Network (IDN)

IDN is a research prototype for turning broad capabilities in large language
models into smaller, deployable services. Instead of asking one large model to
handle every request, IDN explores how to extract task specialists, identify the
capability a request needs, and run that capability on a smaller model close to
where it is needed.

The long-term vision is an **Intelligence Delivery Network**: a distributed AI
serving architecture that places and routes intelligence in much the same way
that a CDN places and routes content. This repository contains the experimental
model pipeline behind that vision.

> [!IMPORTANT]
> IDN is an active research project, not a production serving system. The code
> currently targets local Qwen2.5 checkpoints and six text benchmarks. Results
> below show that capability specialization is promising, while robust routing
> and composition remain open problems.

## What is implemented

- **Capability specialists:** supervised LoRA training and evaluation for six
  language tasks.
- **Semantic routing:** a PCA and shrinkage-regularized Mahalanobis router built
  from base-model hidden states, with confidence and out-of-distribution signals.
- **Adapter composition:** top-1 selection, probability-weighted LoRA mixing,
  and an experimental shared-basis representation.
- **Selective fallback:** calibrated policies that serve a request locally only
  when a capability package is certified, and otherwise select a larger-model
  fallback in offline evaluations.
- **Model extraction:** LOCATE, CARVE, HEAL, and CERTIFY stages for scoring model
  units, pruning a Qwen2 model, distilling a recovery adapter, and validating the
  resulting package.
- **Analysis tools:** capability embedding plots, activation visualizations,
  cross-capability transfer experiments, and reproducible JSON reports.

The main experimental paths are:

```text
request -> semantic router -> LoRA specialist or mixture -> local response
                           +-> confidence/certificate gate -> larger-model fallback

Qwen2 model -> LOCATE -> CARVE -> HEAL -> CERTIFY -> certified experiment artifact
```

## Current results

The table below summarizes the original completed Qwen2.5 experiment runs. The
macro score is an unweighted mean across six task-native metrics: ROUGE-L,
accuracy, exact match, and pass@1. It is useful for within-table comparison, but
it is not a universal accuracy measure.

| Evaluation | 1.5B base | 1.5B + specialist LoRA | 7B base |
| --- | ---: | ---: | ---: |
| Standalone task evaluations | 0.6451 | 0.6933 | 0.7135 |
| Balanced 1,000-request stream, oracle specialist selection | 0.6224 | 0.6849 | 0.6939 |

Five of six standalone specialists improve over the 1.5B base. With an oracle
capability label, the specialist system comes within 0.0090 of the 7B model on
the mixed stream. The current learned router does not yet preserve that gain:
weighted routing scores 0.6141, versus 0.6224 for the 1.5B base. Most of the gap
comes from confusing extractive QA with summarization, and wrong adapters can
cause severe negative transfer. These results motivate the certificate gate and
fallback path rather than unconditional local execution.

## Supported capabilities

| Capability | Dataset | Primary metric |
| --- | --- | --- |
| Summarization | CNN/DailyMail | ROUGE-L |
| Grammatical acceptability | CoLA | Accuracy (MCC also reported) |
| Arithmetic reasoning | GSM8K | Exact match |
| Python code generation | MBPP | pass@1 |
| Extractive question answering | SQuAD | Exact match (F1 also reported) |
| Sentiment classification | SST-2 | Accuracy |

## Quick start

### 1. Install IDN

Python 3.8 or newer is required. Python 3.10 and a CUDA GPU with bfloat16 support
are recommended for model workflows. The reference experiments used 24 GB and
80 GB NVIDIA GPUs; memory needs depend on the model and batch size.

```bash
git clone https://github.com/0xLogicLab/IDN.git
cd IDN
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The pipeline deliberately loads models and datasets from local storage rather
than downloading them during a run. At minimum, arrange the assets like this:

```text
/path/to/huggingface/
+-- model/
|   +-- Qwen2.5-1.5B-Instruct/
|   +-- Qwen2.5-7B-Instruct/
+-- dataset/
    +-- cnn_dailymail/
    +-- glue_cola/
    +-- glue_sst2/
    +-- gsm8k/
    +-- mbpp/
    +-- squad/
```

Dataset directories must use the layout produced by
`datasets.DatasetDict.save_to_disk()`. The helper at
[`scripts/download_hf_assets.sh`](scripts/download_hf_assets.sh) downloads and
converts the reference assets. Review its `TARGET_ROOT` and asset list before
running it because the complete download is large.

### 2. Describe your machine

Machine configurations are intentionally local and ignored by Git. Create
`configs/machine/local.yaml` with paths for your system:

```yaml
name: local
model_dir: /path/to/huggingface/model
data_dir: /path/to/huggingface/dataset
device: cuda
train_batch_size: 1
eval_batch_size: 1
local_files_only: true
```

Pass this file to every command with `--machine`:

```bash
idn --machine configs/machine/local.yaml list-capabilities
idn --machine configs/machine/local.yaml list_datasets
```

Use `list_datasets --no-inspect` to inspect configuration without loading the
datasets.

### 3. Evaluate a local model

Start with a small evaluation to verify the installation and asset paths:

```bash
idn --machine configs/machine/local.yaml eval \
  --config configs/capabilities/sst2.yaml \
  --model Qwen2.5-1.5B-Instruct \
  --limit 20 \
  --output artifacts/eval/sst2_smoke.json
```

Model names passed to `--model` are resolved relative to `model_dir`. Evaluation
artifacts include aggregate metrics, run metadata, and per-example predictions.

## Common workflows

Train a rank-16 LoRA specialist:

```bash
idn --machine configs/machine/local.yaml train-lora \
  --config configs/capabilities/sst2.yaml \
  --model Qwen2.5-1.5B-Instruct \
  --rank 16 \
  --output artifacts/adapters/sst2/r16
```

Evaluate the adapter:

```bash
idn --machine configs/machine/local.yaml eval \
  --config configs/capabilities/sst2.yaml \
  --model Qwen2.5-1.5B-Instruct \
  --adapter artifacts/adapters/sst2/r16 \
  --output artifacts/eval/sst2_r16.json
```

Fit the semantic router and route one request:

```bash
idn --machine configs/machine/local.yaml fit-routing \
  --model Qwen2.5-1.5B-Instruct

idn --machine configs/machine/local.yaml route \
  --input "Write Python code to sort a list"
```

Run the extraction pipeline one stage at a time:

```bash
# Score capability-relevant Qwen2 units.
idn --machine configs/machine/local.yaml locate \
  --config configs/capabilities/sst2.yaml \
  --model Qwen2.5-1.5B-Instruct

# See all CARVE, HEAL, and CERTIFY options before launching expensive jobs.
idn carve --help
idn heal --help
idn certify --help
idn gate-eval --help
```

Inspect a base model or adapter interactively:

```bash
idn-chat --model /path/to/huggingface/model/Qwen2.5-1.5B-Instruct
idn-chat --model /path/to/huggingface/model/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/adapters/sst2/r16
```

Run `idn --help` or `idn <command> --help` for the complete CLI. The `scripts/`
directory contains the exact multi-stage experiment drivers used for routing,
shared-basis composition, selective fallback, and plotting. Read a script before
running it: some reference paths and batch sizes are specific to the original
experiment machines. The `gen-teacher`, `build-sft`, `export`, `serve`, and
`report` commands are currently placeholders.

## Repository layout

```text
configs/capabilities/  Task prompts, splits, metrics, and quality tolerances
configs/train/         Default LoRA training configuration
scripts/               Experiment, evaluation, and plotting entry points
src/idn/analysis/      Embedding and activation analysis
src/idn/capabilities/  Dataset adapters, prompts, parsers, and metrics
src/idn/extraction/    LOCATE, CARVE, HEAL, and CERTIFY implementations
src/idn/routing/       Semantic routing space and inference
src/idn/serving/       Certificate gate and selective-serving logic
src/idn/training/      LoRA training pipeline
```

Generated models, datasets, and experiment artifacts are excluded from Git
because of their size. Commands write new outputs under `artifacts/` by default.

## Adding a capability

A new capability needs both a YAML description under `configs/capabilities/`
and a Python `CapabilitySpec` that loads examples, formats prompts, parses model
outputs, and computes metrics. Register that spec in
[`src/idn/registry.py`](src/idn/registry.py). Existing capability modules are the
best templates for classification, generation, and extractive QA tasks.

## Contributing

Issues and focused pull requests are welcome, especially for reproducibility,
new capability adapters, routing calibration, extraction quality, and support
for additional local model families. Please include the command used, relevant
hardware and software versions, and a small reproducible result when reporting
experimental behavior.

## Citation

If this repository supports your research, please cite it as:

```bibtex
@misc{idn2026,
  title  = {Intelligence Delivery Network: Toward an Internet Architecture for the AI Age},
  author = {Hanling Wang and Qing Li and Dan Zhao and Yuhong Song and Xingchi Chen and Teng Gao and Peiyuan Zong and Dirk Kutscher and Zhuyun Qi and Yue Yu and Yong Jiang},
  year   = {2026},
  url    = {https://github.com/0xLogicLab/IDN}
}
```

## License

Unless otherwise noted, IDN source code is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Project-authored
documentation, papers, and figures are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Models, datasets, and other third-party assets are not relicensed by this
project. Their respective upstream licenses and terms continue to apply. Files
derived from third-party projects retain their original notices and license
requirements.
