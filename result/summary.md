# Qwen2.5 IDN Experiment Summary

Last updated: 2026-07-17

## Evaluation conventions

The six capabilities use different native metrics, so there is no single literal
"accuracy" that is valid for every task. The primary metrics are ROUGE-L for
CNN/DailyMail, accuracy for CoLA and SST-2, exact match for GSM8K and SQuAD, and
pass@1 for MBPP. `macro_capability_score` is the unweighted mean of these six
primary metrics and gives every capability equal importance.

The standalone evaluations requested up to 1,000 held-out examples per capability.
MBPP contains only 257 available evaluation examples and SST-2 contains 872, so
those runs use their complete smaller splits. Each of the 1.5B base, 1.5B LoRA,
and 7B base result sets therefore contains 5,129 predictions rather than 6,000.

## Standalone base and specialist results

| Capability | Primary metric | Samples | 1.5B base | 7B base | 1.5B + selected LoRA | LoRA vs 1.5B |
|---|---:|---:|---:|---:|---:|---:|
| CNN/DailyMail | ROUGE-L | 1,000 | 0.1734 | 0.1801 | 0.1999 | +0.0265 |
| CoLA | Accuracy | 1,000 | 0.7840 | 0.8200 | 0.8730 | +0.0890 |
| GSM8K | Exact match | 1,000 | 0.6930 | 0.8780 | 0.6880 | -0.0050 |
| MBPP | pass@1 | 257 | 0.5253 | 0.7626 | 0.5292 | +0.0039 |
| SQuAD | Exact match | 1,000 | 0.7760 | 0.7170 | 0.9030 | +0.1270 |
| SST-2 | Accuracy | 872 | 0.9186 | 0.9232 | 0.9667 | +0.0482 |
| **Equal-capability macro** | Mixed native metrics | 6 capabilities | **0.6451** | **0.7135** | **0.6933** | **+0.0483** |

The 7B base has an equal-capability macro advantage of 0.0684 over the 1.5B base
and 0.0202 over the selected 1.5B specialist set. Its largest gains over the 1.5B
base are GSM8K (+0.1850 exact match) and MBPP (+0.2374 pass@1). The 7B result is
only slightly higher on CNN/DailyMail, CoLA, and SST-2, and its measured SQuAD
exact match is lower by 0.0590.

Secondary metrics are CoLA MCC 0.4769 / 0.5955 / 0.6974 and SQuAD F1 0.8568 /
0.8445 / 0.9350 for the 1.5B base, 7B base, and 1.5B LoRA respectively. Five
specialists improve their primary metric over the 1.5B base, although the MBPP gain
is negligible. The selected GSM8K result uses the answer-weighted adapter scaled
to 0.1. Over 1,000 examples it improves exact match from 0.5070 for the previous
full-strength specialist to 0.6880, finishing 0.0050 below the 1.5B base and
0.1900 below the 7B base.

## Mixed 1,000-example framework evaluation

The custom stream is balanced across all six capabilities: 167 examples each for
CNN/DailyMail, CoLA, GSM8K, and MBPP, and 166 each for SQuAD and SST-2. No training
split examples are included. Because some public datasets do not provide enough
unused labeled rows, 407 examples may overlap earlier evaluation runs; this is
evaluation overlap, not training leakage.

| System | Status | Macro capability score | Sample-weighted score | Routing accuracy | In-scope coverage |
|---|---|---:|---:|---:|---:|
| 1.5B base | Complete, 1,000/1,000 | 0.6224 | 0.6220 | N/A | N/A |
| 7B base | Complete, 1,000/1,000 | 0.6939 | 0.6938 | N/A | N/A |
| Oracle capability LoRA | Pending; no result artifact yet | - | - | 1.0000 by definition | N/A |
| Semantic ROUTE + weighted six-LoRA mix (scaled GSM8K) | Complete, 1,000/1,000 | 0.6141 | 0.6140 | 0.9010 | 0.5930 |
| Semantic ROUTE + four-basis/top-1-residual mix (scaled GSM8K) | Complete, 1,000/1,000 | 0.6148 | 0.6147 | 0.9010 | 0.5930 |

The completed 1.5B and 7B base, direct routed-LoRA, and shared-basis
per-capability results are:

| Capability | 1.5B base | 7B base | Routed LoRA | Shared basis | Basis vs 1.5B |
|---|---:|---:|---:|---:|---:|
| CNN/DailyMail ROUGE-L | 0.2098 | 0.2202 | 0.2394 | 0.2375 | +0.0277 |
| CoLA accuracy | 0.7784 | 0.7784 | 0.8443 | 0.8503 | +0.0719 |
| GSM8K exact match | 0.6527 | 0.8503 | 0.6467 | 0.6527 | 0.0000 |
| MBPP pass@1 | 0.5329 | 0.7725 | 0.5868 | 0.5749 | +0.0419 |
| SQuAD exact match | 0.6145 | 0.6024 | 0.4036 | 0.4096 | -0.2048 |
| SST-2 accuracy | 0.9458 | 0.9398 | 0.9639 | 0.9639 | +0.0181 |

On the mixed stream, the 7B base exceeds the 1.5B base by 0.0716 macro score
and 0.0717 sample-weighted score. Its largest gains are MBPP (+0.2395 pass@1)
and GSM8K (+0.1976 exact match). CNN/DailyMail improves by 0.0104 and CoLA is
unchanged, while SQuAD and SST-2 decrease by 0.0120 and 0.0060 respectively.

Replacing the full-strength GSM8K adapter raises the routed mixture macro score
from 0.5779 to 0.6141 and the shared-basis score from 0.5675 to 0.6148. The direct
mixture's GSM8K exact match increases by 0.2395; the basis mixture gains 0.2275 on
GSM8K and 0.0663 on SQuAD. The updated direct and basis mixtures remain 0.0082 and
0.0075 below the 1.5B base macro score, respectively, and well below the 7B base.

The shared-basis result is 0.0007 above direct mixing. It exactly matches the 1.5B
base on GSM8K and improves CNN/DailyMail, CoLA, MBPP, and SST-2, but SQuAD remains
0.2048 below the base. This is still primarily a routing problem: only 69 of 166
SQuAD requests route to SQuAD, while 96 route to CNN/DailyMail.

## Routing findings

The router fit report is strong on its original held-out set: 0.9967 routing
accuracy over 1,200 samples, 0.7742 coverage, 1.0000 selective accuracy, and 1.0000
rejection on 200 synthetic OOD probes. On the custom mixed stream, however,
routing accuracy falls to 0.9010 and coverage to 0.5930.

The mixed-stream mean route confidence is 0.9744 despite the SQuAD errors. This
indicates that the current distance-derived probabilities are overconfident under
the custom stream's distribution shift. Routing calibration and better separation
of extractive QA from summarization are required before routing confidence can be
used reliably as an acceptance gate.

## Shared-basis findings

The scaled-GSM8K evaluation uses four global shared directions plus one top-1
capability residual. Its result artifact records 0.9951 retained energy and rank-96
materialized adapters. Because the scaled GSM8K specialist targets only the four
attention projections, its 84 absent MLP LoRA modules are represented as exact
zero deltas in the common module space.

There is a provenance mismatch that must be resolved before reproducing this basis
result. The evaluation records manifest SHA-256 `d5fb8c24...`, while the manifest
currently present at `artifacts/shared_basis_gsm8k_s0.1/manifest.json` hashes to
`27041f9a...` and reports 0.9866 retained energy. The scores above describe the
manifest recorded in the completed evaluation, not the subsequently replaced file.

The current materialization uses four rank-96 basis adapters and six rank-96
residual adapters. It is a functional shared-space experiment, not yet a storage
compression result: the complete basis package is about 2.1 GB, compared with
roughly 216 MB for the six original rank-16 adapters. At inference, only the four
bases and the probability-scaled top-1 residual are active for each request.

## Other key findings

- The scaled GSM8K specialist closes nearly all of its deficit against the 1.5B
  base, while CoLA, SQuAD, and SST-2 specialists improve clearly. On the mixed
  stream, SQuAD routing is now the dominant measured failure rather than GSM8K
  specialist quality.
- Cross-capability tests show severe negative transfer from selecting the wrong
  adapter. For example, CoLA accuracy falls from a 0.74 base anchor to 0.28 with
  the SST-2 adapter, and SST-2 accuracy falls from 0.92 to 0.09 with the CoLA
  adapter. Correct routing or conservative rejection is therefore essential.
- The next decisive comparison is the missing oracle-LoRA result on the same
  1,000-example mixed stream. It will separate specialist-quality failures from
  routing failures; the completed basis result now quantifies the additional cost
  of shared-space truncation.

## Source artifacts

- Standalone base: `artifacts/eval/base_all_limit1000/`
- Standalone 7B base: `artifacts/eval/base_all_limit1000_7B/`
- Standalone specialists: `artifacts/eval/lora_all_limit1000/`
- Standalone scaled GSM8K: `artifacts/eval/gsm8k_r16_answer_weighted_s0.1_limit1000.json`
- Mixed dataset: `artifacts/eval/idn_mixed_1000.jsonl`
- Mixed base: `artifacts/eval/idn_mixed_oracle/base.json`
- Mixed 7B base: `artifacts/eval/idn_mixed_oracle/7b_base.json`
- Mixed oracle LoRA: `artifacts/eval/idn_mixed_oracle/oracle_lora.json` (pending)
- Routed weighted LoRA with scaled GSM8K: `artifacts/eval/idn_mixed_scaled_gsm8k/routed_lora_mix.json`
- Routed shared basis with scaled GSM8K: `artifacts/eval/idn_mixed_scaled_gsm8k/basis_lora_mix.json`
- Router report: `artifacts/routing/report.json`
- Shared-basis manifest currently on disk: `artifacts/shared_basis_gsm8k_s0.1/manifest.json`
- Phase 1 specialist report: `artifacts/reports/phase1_specialists.json`
