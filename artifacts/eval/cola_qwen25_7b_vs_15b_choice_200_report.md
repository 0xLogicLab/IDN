# Local Qwen Big vs Small Accuracy Reference — CoLA Dev

Generated: 2026-06-25
Project: IDN capability extraction

## Dataset

- Dataset: CoLA in-domain dev set
- Source: https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/cola_public/tokenized/in_domain_dev.tsv
- Train few-shot source: https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/cola_public/tokenized/in_domain_train.tsv
- Local dev file: `data/cola/in_domain_dev.tsv`
- Local train file: `data/cola/in_domain_train.tsv`
- Task: binary English grammatical acceptability classification
- Label mapping: `A = acceptable`, `B = unacceptable`
- Evaluation subset: first 200 dev examples
- Prompting: 3-shot per label from train set
- Scoring: single-step next-token logit score over A/B label tokens

## Models

- Small model: `/home/sig/models/Qwen2.5-1.5B-Instruct`
- Big model: `/home/sig/models/Qwen2.5-7B-Instruct`
- Hardware: RTX 3090 24GB
- Runtime: local HuggingFace Transformers, fp16, device_map=auto

## Results

| Role | Model | Accuracy | Correct / Total | Mean latency | P50 latency | P95 latency |
|---|---|---:|---:|---:|---:|---:|
| Small | Qwen2.5-1.5B-Instruct | 71.0% | 142 / 200 | 32.1 ms | 27.7 ms | 29.1 ms |
| Big | Qwen2.5-7B-Instruct | 67.5% | 135 / 200 | 27.0 ms | 26.8 ms | 28.4 ms |

Confusion matrices:

| Role | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Small | 125 | 17 | 53 | 5 |
| Big | 73 | 62 | 8 | 57 |

## Interpretation for IDN model distribution

On this CoLA slice, Qwen2.5-1.5B-Instruct is more aggressive toward `A/acceptable`, giving higher overall accuracy on the first 200 examples. Qwen2.5-7B-Instruct is more conservative toward `B/unacceptable`, giving fewer false positives and more false negatives.

Routing reference:

- For short grammar/acceptability classification, route default traffic to Qwen2.5-1.5B-Instruct.
- Use Qwen2.5-7B-Instruct as fallback when false positives are more costly than false negatives.
- Add domain-specific IDN/networking classification and reasoning datasets before broad distribution policy.

## Artifacts

- Choice-scoring runner: `src/idn/eval_qwen_choice_accuracy.py`
- Generative runner: `src/idn/eval_qwen_accuracy.py`
- Config note: `configs/cola_qwen25_local_accuracy.yaml`
- Per-example records: `artifacts/eval/cola_qwen25_7b_vs_15b_choice_200.jsonl`
- Summary JSON: `artifacts/eval/cola_qwen25_7b_vs_15b_choice_200_summary.json`
- Report: `artifacts/eval/cola_qwen25_7b_vs_15b_choice_200_report.md`

## Reproduction

```bash
/home/sig/anaconda3/bin/python src/idn/eval_qwen_choice_accuracy.py \
  --num-samples 200 \
  --few-shot-per-label 3 \
  --small-model /home/sig/models/Qwen2.5-1.5B-Instruct \
  --big-model /home/sig/models/Qwen2.5-7B-Instruct \
  --output-path artifacts/eval/cola_qwen25_7b_vs_15b_choice_200.jsonl \
  --summary-path artifacts/eval/cola_qwen25_7b_vs_15b_choice_200_summary.json
```
