# Big vs Small Model Accuracy Reference — CoLA Dev

Generated: 2026-06-25
Project: IDN capability extraction

## Dataset

- Dataset: CoLA in-domain dev set
- Source: https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/cola_public/tokenized/in_domain_dev.tsv
- Local file: `data/cola/in_domain_dev.tsv`
- Task: binary English grammatical acceptability classification
- Label mapping: `A = acceptable`, `B = unacceptable`
- Evaluation subset: first 200 dev examples
- Decoding: temperature 0, max_tokens 4

This is a real public benchmark dataset. The previous smoke/sample data only had 5 handcrafted records and was too small for model capability reference.

## Models

- Small model: `gpt-4o-mini`
- Big model: `gpt-4o`
- Provider: OpenAI Chat Completions API

## Results

| Role | Model | Accuracy | Correct / Total | Invalid predictions | Mean latency | P50 latency | P95 latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Small | gpt-4o-mini | 82.0% | 164 / 200 | 0 | 1390 ms | 1038 ms | 4786 ms |
| Big | gpt-4o | 79.0% | 158 / 200 | 0 | 1016 ms | 889 ms | 1337 ms |

Confusion matrices:

| Role | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Small | 109 | 55 | 15 | 21 |
| Big | 113 | 45 | 25 | 17 |

There were 26 prediction disagreements between the two models.

## Interpretation for model distribution

On this quick CoLA slice, `gpt-4o-mini` achieved slightly higher accuracy than `gpt-4o`, while `gpt-4o` had lower latency in this API run. The dataset is grammar acceptability, so it should be treated as a language understanding reference point, not a universal capability score.

Recommended use in IDN routing:

- Use small model for default grammatical / short classification requests.
- Trigger big model fallback when confidence estimation, answer verification, or task metadata suggests high-risk ambiguity.
- Add at least two more datasets before using this as a broad distribution policy: one reasoning dataset and one domain-specific IDN/networking dataset.

## Artifacts

- Per-example records: `artifacts/eval/cola_gpt4o_vs_gpt4omini_200.jsonl`
- Summary JSON: `artifacts/eval/cola_gpt4o_vs_gpt4omini_200_summary.json`
- Runner script: `src/idn/eval_accuracy.py`
- Config note: `configs/cola_openrouter_accuracy.yaml`

## Reproduction

```bash
bash -ic '/home/sig/anaconda3/bin/python src/idn/eval_accuracy.py \
  --num-samples 200 \
  --big-model gpt-4o \
  --small-model gpt-4o-mini \
  --output-path artifacts/eval/cola_gpt4o_vs_gpt4omini_200.jsonl \
  --summary-path artifacts/eval/cola_gpt4o_vs_gpt4omini_200_summary.json'
```
