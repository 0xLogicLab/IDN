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

| Capability | Primary metric | Samples | 1.5B base | 7B base | 1.5B + capability LoRA | LoRA vs 1.5B |
|---|---:|---:|---:|---:|---:|---:|
| CNN/DailyMail | ROUGE-L | 1,000 | 0.1734 | 0.1801 | 0.1999 | +0.0265 |
| CoLA | Accuracy | 1,000 | 0.7840 | 0.8200 | 0.8730 | +0.0890 |
| GSM8K | Exact match | 1,000 | 0.6930 | 0.8780 | 0.5070 | -0.1860 |
| MBPP | pass@1 | 257 | 0.5253 | 0.7626 | 0.5292 | +0.0039 |
| SQuAD | Exact match | 1,000 | 0.7760 | 0.7170 | 0.9030 | +0.1270 |
| SST-2 | Accuracy | 872 | 0.9186 | 0.9232 | 0.9667 | +0.0482 |
| **Equal-capability macro** | Mixed native metrics | 6 capabilities | **0.6451** | **0.7135** | **0.6631** | **+0.0181** |

The 7B base has an equal-capability macro advantage of 0.0684 over the 1.5B base
and 0.0503 over the 1.5B specialist set. Its largest gains over the 1.5B base are
GSM8K (+0.1850 exact match) and MBPP (+0.2374 pass@1). The 7B result is only
slightly higher on CNN/DailyMail, CoLA, and SST-2, and its measured SQuAD exact
match is lower by 0.0590.

Secondary metrics are CoLA MCC 0.4769 / 0.5955 / 0.6974 and SQuAD F1 0.8568 /
0.8445 / 0.9350 for the 1.5B base, 7B base, and 1.5B LoRA respectively. Five
specialists improve their primary metric over the 1.5B base, although the MBPP gain
is negligible. The full-strength GSM8K LoRA remains the clear specialist failure:
its 1,000-example result loses 18.6 percentage points against the 1.5B base and
37.1 points against the 7B base. A separately calibrated 0.1-scaled GSM8K adapter
improves the 100-example checkpoint to 0.70, as detailed below.

### GSM8K scaled-LoRA checkpoint

| GSM8K system | Adapter scale | Samples | Exact match |
|---|---:|---:|---:|
| 1.5B base | N/A | 100 | 0.6900 |
| Previous conservative LoRA | 1.0 | 100 | 0.5300 |
| Cleaned-rationale/KL LoRA | 1.0 | 100 | 0.5800 |
| Answer-weighted LoRA | 0.1 | 100 | **0.7000** |

The scaled answer-weighted adapter improves by 0.17 over the previous conservative
LoRA and by 0.01 over the 1.5B base on this checkpoint. Its 1,000-example run was
interrupted after 22 examples, so 0.70 is a completed 100-example result and does
not replace the unscaled 1,000-example value in the main table.

## Mixed 1,000-example framework evaluation

The custom stream is balanced across all six capabilities: 167 examples each for
CNN/DailyMail, CoLA, GSM8K, and MBPP, and 166 each for SQuAD and SST-2. No training
split examples are included. Because some public datasets do not provide enough
unused labeled rows, 407 examples may overlap earlier evaluation runs; this is
evaluation overlap, not training leakage.

| System | Status | Macro capability score | Sample-weighted score | Routing accuracy | In-scope coverage |
|---|---|---:|---:|---:|---:|
| 1.5B base | Complete, 1,000/1,000 | 0.6224 | 0.6220 | N/A | N/A |
| Oracle capability LoRA | Pending; no result artifact yet | - | - | 1.0000 by definition | N/A |
| Semantic ROUTE + weighted six-LoRA mix | Complete, 1,000/1,000 | 0.5779 | 0.5777 | 0.9010 | 0.5890 |
| Semantic ROUTE + four-basis/top-1-residual mix | Complete, 1,000/1,000 | 0.5675 | 0.5673 | 0.9010 | 0.5890 |

The completed mixed base, direct routed-LoRA, and shared-basis per-capability results are:

| Capability | Mixed base | Routed LoRA | Shared basis | Basis vs base |
|---|---:|---:|---:|---:|
| CNN/DailyMail ROUGE-L | 0.2098 | 0.2376 | 0.2355 | +0.0257 |
| CoLA accuracy | 0.7784 | 0.8443 | 0.8443 | +0.0659 |
| GSM8K exact match | 0.6527 | 0.4072 | 0.4251 | -0.2275 |
| MBPP pass@1 | 0.5329 | 0.6048 | 0.5928 | +0.0599 |
| SQuAD exact match | 0.6145 | 0.4096 | 0.3434 | -0.2711 |
| SST-2 accuracy | 0.9458 | 0.9639 | 0.9639 | +0.0181 |

The direct routed mixture improves four capabilities but loses heavily on GSM8K
and SQuAD, reducing the macro score by 0.0445 relative to the base. The shared
basis improves over the base on four capabilities as well, but its macro score is
another 0.0104 below the direct routed mixture. Relative to direct routed mixing,
the basis recovers some GSM8K performance (+0.0180) but loses CNN/DailyMail
(-0.0021), MBPP (-0.0120), and especially SQuAD (-0.0663). The SQuAD degradation
is still primarily a routing problem: only 69 of 166 SQuAD requests route to
SQuAD, while 96 route to CNN/DailyMail; shared-space truncation compounds it.

## Routing findings

The router fit report is strong on its original held-out set: 0.9967 routing
accuracy over 1,200 samples, 0.7742 coverage, 1.0000 selective accuracy, and 1.0000
rejection on 200 synthetic OOD probes. On the custom mixed stream, however,
routing accuracy falls to 0.9010 and coverage to 0.5890.

The mixed-stream mean route confidence is 0.9744 despite the SQuAD errors. This
indicates that the current distance-derived probabilities are overconfident under
the custom stream's distribution shift. Routing calibration and better separation
of extractive QA from summarization are required before routing confidence can be
used reliably as an acceptance gate.

## Shared-basis findings

The six rank-16 LoRA deltas were decomposed into four global shared directions plus
one exact residual per capability. The four directions retain 0.9199 of total delta
energy. Algebraic reconstruction with each capability residual is exact before
artifact quantization.

The retained energy is not balanced across capabilities. Relative residual squared
norm is approximately 0.2% for CNN/DailyMail and effectively zero for CoLA, GSM8K,
and SST-2, but it is 99.98% for MBPP and 99.71% for SQuAD. Thus the four global
directions mostly capture CNN/DailyMail, CoLA, GSM8K, and SST-2; MBPP and SQuAD
remain almost entirely capability-specific. The completed basis evaluation
confirms the consequence: it is close to direct routed mixing on CNN/DailyMail,
CoLA, GSM8K, MBPP, and SST-2, but substantially worse on SQuAD.

The current materialization uses four rank-96 basis adapters and six rank-96
residual adapters. It is a functional shared-space experiment, not yet a storage
compression result: the complete basis package is about 2.1 GB, compared with
roughly 216 MB for the six original rank-16 adapters. At inference, only the four
bases and the probability-scaled top-1 residual are active for each request.

## Other key findings

- Earlier Phase 1 anchors show CoLA and SST-2 specialists exceeding both the 1.5B
  base and the measured 7B teacher, while GSM8K remains below both. The scientific
  bottleneck is specialist quality, not only routing or composition.
- Full-strength revised GSM8K recipes remain below the base: answer-weighted SFT
  reached 0.4074 on its first 27 examples, and cleaned-rationale SFT with base-logit
  KL preservation reached 0.58 on 100 examples. Applying a 0.1 adapter scale raises
  the answer-weighted checkpoint result to 0.70 on 100 examples; its full
  1,000-example result is still unavailable.
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
- GSM8K scaled LoRA checkpoint: `artifacts/eval/gsm8k_r16_answer_weighted_s0.1_limit100.json`
- Mixed dataset: `artifacts/eval/idn_mixed_1000.jsonl`
- Mixed base: `artifacts/eval/idn_mixed_oracle/base.json`
- Mixed oracle LoRA: `artifacts/eval/idn_mixed_oracle/oracle_lora.json` (pending)
- Routed weighted LoRA: `artifacts/eval/idn_mixed_routed_lora.json`
- Routed shared basis: `artifacts/eval/idn_mixed_shared_basis.json` (complete)
- Router report: `artifacts/routing/report.json`
- Shared-basis manifest: `artifacts/shared_basis/manifest.json`
- Phase 1 specialist report: `artifacts/reports/phase1_specialists.json`
