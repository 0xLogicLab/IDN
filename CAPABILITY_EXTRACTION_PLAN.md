# IDN Capability Extraction — Implementation & Validation Plan (V1)

> **Audience:** this document is written to be handed directly to a coding agent
> (Claude Code / Codex) to generate the implementation. Every module has a path,
> a responsibility, and a function/CLI contract. Read sections 1–4 for intent,
> then implement section 5 onward in phase order (section 11).

---

## 0. One-paragraph summary

We test the central IDN claim — *a small model can match or beat a large teacher
on a specific capability while using a tiny fraction of the parameters/compute* —
by treating each labeled dataset on disk (CoLA, SST-2, SQuAD, GSM8K, CNN/DM, MBPP)
as a **capability**. For each capability we distill the **Qwen2.5-7B-Instruct teacher**
into a **small student** (Qwen2.5-1.5B-Instruct, optionally Qwen2.5-1B / a 0.5B model)
using **LoRA**, sweep LoRA rank × target-modules × base-model-size, and produce an
**accuracy-per-parameter Pareto frontier**. The deliverable of V1 is: for each capability,
the smallest added-parameter footprint that reaches teacher-level accuracy, packaged as a
loadable IDN CapabilityPackage. GSM8K (math reasoning) is the flagship task.

---

## 1. Scientific claim and what V1 must prove

The paper frames capability extraction as **local conditional approximation**, not global
distillation. V1 reduces this to a measurable claim per capability `c`:

```
For capability c with eval metric M_c:
  exists a small base θ0 and a low-rank Δ (LoRA) such that
    M_c( F_{θ0+Δ} )  >=  M_c( F_teacher ) - ε      (quality)
  while
    params(Δ) / params(F_teacher)  <=  small budget  (efficiency)
```

V1 is **successful** if, for at least the flagship capability (GSM8K) and ≥2 others, the
distilled small+LoRA student lands on or above the teacher's metric while adding <2% of the
*student's* params and <0.5% of the *teacher's* params. The headline artifact is the
**frontier plot/table**: metric vs added-parameter-count across the sweep.

Non-goals for V1: multi-capability composition (shared basis / residual), online serving at
scale, PCA region discovery from unlabeled logs. Those are V2/V3 (section 14) and must not
block V1.

---

## 2. Environment (verified on this machine — do not assume, this is measured)

**Local machine (primary dev + light training):**
- 1× RTX 3090 (24 GB). Idle.
- Use the `control` conda env: `/home/sig/anaconda3/envs/control/bin/python`
  - Python 3.8.5, torch 2.0.0+cu117, transformers 4.46.3, peft 0.13.2, trl 0.11.4,
    datasets 2.21.0, accelerate 1.0.1, scikit-learn 1.0. **No bitsandbytes** (QLoRA
    unavailable locally unless installed).
- `torch.cuda.is_bf16_supported()` is **True**, but this torch/CUDA stack produced
  **fp16 NaN / `!`-repetition** in the prior run. **Rule: train and eval in `bfloat16`
  (preferred) or `float32` (fallback). Never fp16.** Make dtype a config field defaulting
  to `bfloat16`, with `float32` fallback documented.

**Models (local, absolute paths — load from disk, never from the hub):**
- Teacher: `/data/huggingface/model/Qwen2.5-7B-Instruct`
- Student (default): `/data/huggingface/model/Qwen2.5-1.5B-Instruct`
- A 1B and 0.5B Qwen may be added later; keep base-model path a config field.

**Datasets (local, HuggingFace `datasets.load_from_disk` format):**

| capability id | metric | path | splits (train/val/test) |
|---|---|---|---|
| `gsm8k` (flagship) | final-answer exact match | `/data/huggingface/dataset/gsm8k` | 7473 / – / 1319 |
| `cola` | accuracy + MCC | `/data/huggingface/dataset/glue_cola` | 8551 / 1043 / 1063(unlabeled) |
| `sst2` | accuracy | `/data/huggingface/dataset/glue_sst2` | 67349 / 872 / 1821(unlabeled) |
| `squad` | EM + F1 | `/data/huggingface/dataset/squad` | 87599 / 10570 / – |
| `cnn_dailymail` | ROUGE-L | `/data/huggingface/dataset/cnn_dailymail` | 287113 / 13368 / 11490 |
| `mbpp` | pass@1 (unit tests) | `/data/huggingface/dataset/mbpp` | 120 / 43 / 257 |

> GLUE `test` splits have hidden labels (label = -1). **Use `validation` as the held-out
> eval set** for CoLA/SST-2. For GSM8K use `test`. For SQuAD use `validation`. Cap eval
> sets (e.g. 500–1000 examples) via config for fast iteration; full eval for final numbers.

**Cloud machine (heavy training only, optional):**
- `ssh wanghl@172.22.11.235` (password in `info.md`), 1× H100 80GB.
- Use for: teacher generation over large train splits, 7B-as-student experiments, full sweeps.
- Sync with `rsync -av --exclude .git --exclude .venv --exclude artifacts ...`. Code must run
  identically on both; only paths/batch-size differ via config.

---

## 3. Design decisions locked for V1 (from review)

1. **Framing:** known labeled capabilities (each dataset = one capability). Region discovery
   from logs is deferred to V2.
2. **Primary metric:** accuracy-per-parameter frontier. Report the curve, not a single point.
3. **Flagship task:** GSM8K. Build and debug the whole pipeline on a cheap task (SST-2 or CoLA)
   first, then run flagship.
4. **Teacher role:** the 7B is both (a) the *reference bar* we must match/beat, and (b) the
   *label source* for distillation (teacher-generated targets), in addition to gold labels.
5. **Student adaptation:** LoRA only in V1 (full-FT and QLoRA are sweep options, not required).
6. **Routing/serving:** a minimal single-capability loader + verifier + teacher-fallback is in
   scope (section 9) but the headline result is offline extraction quality, not serving throughput.

---

## 4. The experiment pipeline (V1 closed loop)

```
capability spec (dataset adapter + metric + prompt template)
  -> [optional] teacher generation: F_teacher(x) over train split  -> distillation targets
  -> build SFT dataset: {prompt, target}  (target = gold and/or teacher output)
  -> train student LoRA  (sweep: base model × rank × target modules × dtype)
  -> evaluate: base student / student+LoRA / teacher  on held-out eval split
  -> compute frontier: metric vs added-params; risk gap to teacher; latency; VRAM; disk
  -> export CapabilityPackage (descriptor + adapter + verifier + fallback + metrics + lineage)
  -> [optional] serve: load base+adapter, verifier, fallback to teacher
  -> report: per-capability table + frontier plot
```

Each arrow is a CLI subcommand (section 8). Every stage writes artifacts to disk so stages
are independently re-runnable and cacheable.

---

## 5. Repository structure to generate

```
/home/sig/files/IDN/
  pyproject.toml                 # package "idn", entry point idn = idn.cli:main
  README.md
  configs/
    capabilities/
      gsm8k.yaml                 # one file per capability (spec + prompt + metric + eval cap)
      cola.yaml
      sst2.yaml
      squad.yaml
      cnn_dailymail.yaml
      mbpp.yaml
    train/
      lora_default.yaml          # default LoRA hyperparams + dtype=bfloat16
    sweep/
      frontier_v1.yaml           # the rank × target-modules × base-model grid
    machine/
      local_3090.yaml            # paths, batch sizes, eval caps for the 3090
      cloud_h100.yaml            # H100 overrides
  src/idn/
    __init__.py
    cli.py                       # argparse/typer dispatch to all subcommands
    config.py                    # dataclasses + YAML load/merge (machine overrides capability)
    registry.py                  # capability registry: id -> spec, package discovery
    capabilities/
      base.py                    # CapabilitySpec protocol: load_split, format_prompt,
                                  #   parse_prediction, metric(preds, refs) -> dict
      gsm8k.py                   # final-answer extraction (#### parsing + regex), EM metric
      cola.py                    # acceptable? yes/no -> label, accuracy + MCC
      sst2.py                    # positive/negative -> label, accuracy
      squad.py                   # span answer, SQuAD EM + F1 (official-style)
      cnn_dailymail.py           # summary, ROUGE-L
      mbpp.py                    # code, run unit tests in subprocess sandbox, pass@1
    data/
      build_sft.py               # capability + (gold|teacher) -> SFT jsonl {prompt,target,meta}
      collator.py                # causal LM collator with prompt-masking (-100 on prompt tokens)
    teacher/
      generate.py                # batched F_teacher over a split -> cached jsonl
    models/
      loader.py                  # ChatModel: load base, optional attach LoRA adapter, generate
      lora.py                    # build LoraConfig from train config; param accounting
    training/
      train_lora.py              # SFTTrainer/Trainer wrapper; saves adapter + training_meta
    eval/
      runner.py                  # run a model over eval split, collect predictions + latency
      metrics.py                 # frontier math: risk_gap, added_params, coverage, frontier()
      report.py                  # render markdown table + matplotlib frontier plot
    package/
      schema.py                  # descriptor / lineage / metrics JSON schemas (pydantic/dataclass)
      exporter.py                # write CapabilityPackage dir
    serving/
      capability.py              # load package: base+adapter+verifier+fallback
      verifier.py                # schema/format/confidence verifiers per capability
      server.py                  # minimal FastAPI: /generate with receipt (optional V1)
    sweep/
      run_frontier.py            # orchestrate the grid, collect rows, call report
  tests/
    test_capabilities.py         # metric correctness on tiny fixtures (esp. gsm8k parse, squad f1)
    test_build_sft.py            # prompt masking, target formatting
    test_metrics.py              # frontier/risk_gap math on synthetic numbers
    test_loader_smoke.py         # load 1.5B, generate 5 tokens (gpu-gated, skip if no cuda)
    test_package_roundtrip.py    # export then re-load a package
  artifacts/                     # gitignored: features, adapters, eval, packages, reports
```

**Conventions the agent must follow:**
- All model/dataset paths come from config; **no hub downloads**, everything `local_files_only`.
- Set `os.environ` guards (`USE_TF=0`, `TRANSFORMERS_NO_TF=1`, `TOKENIZERS_PARALLELISM=false`)
  at import top of any entrypoint, matching the prior working setup.
- Default dtype `bfloat16`; expose `--dtype` everywhere; document fp16 is forbidden.
- Greedy decoding (temperature 0) for all eval to make metrics deterministic.
- Every artifact carries a `lineage` blob (base model, teacher, data hash, config hash, git sha,
  timestamp).

---

## 6. The `CapabilitySpec` contract (the heart of the abstraction)

Every capability implements one class. This is the single extension point that makes adding a
new capability cheap and keeps train/eval/serve generic.

```python
# src/idn/capabilities/base.py
from typing import Protocol, Any
from dataclasses import dataclass

@dataclass
class Example:
    id: str
    prompt_inputs: dict[str, Any]   # raw fields needed to render the prompt
    reference: Any                  # gold answer used by metric()

class CapabilitySpec(Protocol):
    id: str                         # "gsm8k", "cola", ...
    metric_name: str                # "exact_match", "accuracy", "mcc", "rougeL", "pass@1"
    higher_is_better: bool

    def load_split(self, split: str, limit: int | None) -> list[Example]: ...
    # render the chat prompt (messages) for one example; same template for base/student/teacher
    def format_prompt(self, ex: Example) -> list[dict]: ...   # [{role, content}, ...]
    # turn raw decoded text into a normalized prediction comparable to reference
    def parse_prediction(self, raw_text: str) -> Any: ...
    # the supervised target string the student should learn to produce (for build_sft)
    def target_text(self, ex: Example, teacher_text: str | None) -> str: ...
    # aggregate metric over the eval set -> {metric_name: value, ...}
    def metric(self, preds: list[Any], refs: list[Any]) -> dict[str, float]: ...
    # verifier used at serve time to decide local-accept vs fallback (format/confidence)
    def verify(self, raw_text: str, parsed: Any) -> bool: ...
```

**Per-capability notes the implementer must get right:**

- **gsm8k:** prompt asks for step-by-step reasoning then `#### <number>`. `parse_prediction`
  extracts the last integer after `####` (fallback: last number in text). Reference is the gold
  final number parsed from the `answer` field's `####`. Metric = exact match on the number.
  `target_text` with teacher = the teacher's full chain-of-thought ending in `#### n` (this is
  the distillation signal that matters for math); with gold-only = the dataset's reference
  solution. **Allow CoT in the target** — this is the capability.
- **cola:** prompt: "Is this sentence grammatically acceptable? Answer yes or no." Parse
  yes/no→1/0. Metrics: accuracy **and** Matthews corr coef (sklearn). Target = "yes"/"no"
  (+ optional 1-line justification, but keep label parseable).
- **sst2:** prompt: positive/negative sentiment. Parse→1/0. Metric accuracy.
- **squad:** prompt: context+question, answer with the exact span. Parse = trimmed text.
  Metric = official SQuAD EM + token-F1 (normalize: lowercase, strip punctuation/articles).
  Handle multiple gold answers (max over references).
- **cnn_dailymail:** prompt: summarize the article. Metric ROUGE-L (use `rouge_score` if
  available; otherwise implement LCS-based ROUGE-L F1 — add dep if needed).
- **mbpp:** prompt: function description + test list signature. Parse = extracted code block.
  Metric = pass@1 by executing the provided `test_list` assertions in a **subprocess with
  timeout** (resource-limited; never exec in-process). This is the highest-effort capability;
  it can ship last.

Registry (`registry.py`) maps `id -> CapabilitySpec` instance and is the only place that
imports concrete capabilities, so train/eval/serve code stays capability-agnostic.

---

## 7. Teacher generation (distillation target source)

`teacher/generate.py`: given a capability + split + limit, run the 7B teacher with the
capability's own prompt template, greedy decode, and cache results to
`artifacts/teacher/{capability}/{split}.jsonl` as `{id, teacher_text, latency_ms}`.

- Batch with left-padding; cap `max_new_tokens` per capability (gsm8k ~512 for CoT, cola/sst2
  ~8, squad ~64, summary ~160, mbpp ~512).
- 7B in bf16 fits the 3090 (~15–16 GB) for inference; if OOM, allow CPU-offload via
  `device_map="auto"` + `max_memory`, or run this stage on the H100.
- Idempotent: skip ids already cached. This stage is the expensive one — cache aggressively.
- Output is consumed by `build_sft.py` to form distillation targets. Also reused as the
  **teacher reference predictions** for eval (don't regenerate).

`data/build_sft.py`: combine capability examples + (gold and/or teacher cache) into
`artifacts/sft/{capability}/{train|eval}.jsonl` rows `{id, prompt_messages, target_text, meta}`.
Config flag `target_source: gold | teacher | teacher_then_gold` selects supervision. For gsm8k
default to `teacher` (CoT); for classification default to `gold`.

---

## 8. CLI surface (`idn ...`)

All commands take `--config` (capability yaml) and `--machine` (machine yaml, merged over it).

```bash
# 0. sanity: list capabilities and resolved paths
idn list-capabilities

# 1. teacher generation (cache distillation targets + teacher reference preds)
idn gen-teacher   --config configs/capabilities/gsm8k.yaml --split train --limit 2000
idn gen-teacher   --config configs/capabilities/gsm8k.yaml --split test  --limit 500

# 2. build SFT dataset
idn build-sft     --config configs/capabilities/gsm8k.yaml --target-source teacher

# 3. train a single student LoRA
idn train-lora    --config configs/capabilities/gsm8k.yaml \
                  --train configs/train/lora_default.yaml \
                  --base /data/huggingface/model/Qwen2.5-1.5B-Instruct \
                  --lora-r 16 --lora-target qkvo \
                  --output artifacts/adapters/gsm8k_qwen15b_r16_qkvo

# 4. evaluate any of: base / student+adapter / teacher
idn eval          --config configs/capabilities/gsm8k.yaml --split test --limit 500 \
                  --base /data/huggingface/model/Qwen2.5-1.5B-Instruct \
                  --adapter artifacts/adapters/gsm8k_qwen15b_r16_qkvo \
                  --output artifacts/eval/gsm8k_qwen15b_r16_qkvo.json
idn eval          --config configs/capabilities/gsm8k.yaml --split test --limit 500 \
                  --base /data/huggingface/model/Qwen2.5-7B-Instruct \
                  --output artifacts/eval/gsm8k_teacher.json     # the bar

# 5. run the whole frontier sweep (section 10) and emit the report
idn sweep         --config configs/capabilities/gsm8k.yaml \
                  --sweep configs/sweep/frontier_v1.yaml \
                  --machine configs/machine/local_3090.yaml

# 6. export a capability package from the best adapter
idn export        --config configs/capabilities/gsm8k.yaml \
                  --adapter artifacts/adapters/gsm8k_qwen15b_r16_qkvo \
                  --eval artifacts/eval/gsm8k_qwen15b_r16_qkvo.json \
                  --output artifacts/packages/gsm8k_v1

# 7. (optional) serve + fallback
idn serve         --package artifacts/packages/gsm8k_v1 \
                  --teacher /data/huggingface/model/Qwen2.5-7B-Instruct --port 8000

# 8. render report across all eval jsons for a capability
idn report        --config configs/capabilities/gsm8k.yaml \
                  --eval-glob 'artifacts/eval/gsm8k_*.json' \
                  --output artifacts/reports/gsm8k_frontier.md
```

---

## 9. Training & serving details

**Training (`training/train_lora.py`):**
- Use `peft.LoraConfig` + HF `Trainer` (or `trl.SFTTrainer`). Prior run confirmed this stack
  works in fp32; prefer **bf16** now that we know fp16 is broken here.
- **Prompt masking is mandatory:** label = -100 on prompt tokens, real ids only on the target
  span. The prior collator masked nothing (`labels = input_ids`); fix this — it materially
  affects quality. Implement in `data/collator.py` and unit-test it.
- Defaults (`configs/train/lora_default.yaml`): `lora_alpha = 2*r`, `lora_dropout = 0.05`,
  `lr = 2e-4`, cosine schedule, `warmup_ratio = 0.03`, `epochs = 2` (configurable; gsm8k may
  need 3), `max_length = 1024` (gsm8k 1536 for CoT), grad checkpointing on, `batch_size`
  + `grad_accum` from machine config (3090: bs 1–2, accum 8–16).
- Save: adapter (`adapter_model.safetensors` + `adapter_config.json`), tokenizer, and
  `training_metadata.json` (loss curve, trainable %, config hash, wall-clock, peak VRAM).
- **Param accounting (`models/lora.py`):** record `trainable_params`, `total_base_params`,
  `added_params = trainable_params`, ratios vs student and vs teacher. This feeds the frontier.

**Serving (`serving/`, optional for V1 sign-off):**
- `capability.py` loads base once, attaches the package adapter, holds the verifier + a teacher
  client for fallback.
- Accept logic (V1, single capability — no routing needed yet):
  `accept = verifier.verify(raw, parsed) AND confidence_ok`. On reject → call teacher.
  `confidence_ok` for classification = label token prob ≥ τ; for generation = format/verifier
  pass (no logprob gate required in V1).
- `/generate` returns an **execution receipt**: `{answer, served_by: local|teacher,
  capability_id, verifier_pass, latency_ms}` (mirrors the paper's Execution Receipt descriptor).
- **Security note:** the FastAPI server is unauthenticated by default. Bind to `127.0.0.1` and
  state in the README that it is a local research endpoint, not production. Do not add network
  exposure without auth.

---

## 10. The frontier sweep (the headline experiment)

`configs/sweep/frontier_v1.yaml` defines a grid; `sweep/run_frontier.py` runs train+eval for
each cell, collects one row per cell, and calls the report.

**Grid axes (start small, expand):**
- `base_model`: [Qwen2.5-1.5B-Instruct] → add [Qwen2.5-1B, Qwen2.5-0.5B] later.
- `lora_r`: [4, 8, 16, 32, 64]  (added params scale ~linearly with r).
- `lora_target`: [`qv` (q,v only), `qkvo` (attn), `qkvo+mlp` (attn + gate/up/down)].
- (optional) `target_source`: [gold, teacher].

**Fixed baselines computed once per capability:**
- `base_student` (no adapter) — lower bound.
- `teacher` (7B) — the bar.
- (optional) `full_ft_student` — upper bound on what the small model's capacity allows.

**Each sweep row records:**
```
{capability, base_model, lora_r, lora_target, target_source,
 metric_value, teacher_metric, base_metric,
 added_params, added_params_pct_student, added_params_pct_teacher,
 adapter_disk_mb, train_wall_clock_s, peak_train_vram_gb,
 eval_p50_ms, eval_p95_ms, eval_vram_gb}
```

**Outputs (`eval/report.py`):**
1. Markdown table sorted by added_params.
2. Frontier plot: x = added_params (log), y = metric; horizontal lines for base & teacher;
   mark the **Pareto-optimal** points (min params at/above teacher).
3. A one-line verdict per capability: smallest config reaching `teacher - ε`.

---

## 11. Metrics & definitions (`eval/metrics.py`)

```
risk_gap          = teacher_metric - student_metric        (lower = closer; ≤0 means beat teacher)
added_params      = LoRA trainable params (or full-FT delta count)
param_efficiency  = metric_value / log10(added_params)     (for ranking; report raw too)
coverage          = fraction of eval accepted locally (serve mode)
selective_metric  = metric on the accepted subset only     (serve mode)
fallback_ratio    = 1 - coverage
pareto_frontier(rows) -> subset where no other row has both >= metric and <= added_params
```

ε defaults: gsm8k/squad 2 metric points; cola/sst2 1 point; summary 1 ROUGE-L point. Configurable.

---

## 12. CapabilityPackage format (`package/`)

Export the best adapter per capability as an IDN-aligned package (maps to paper's capability
descriptor `C=⟨name,task,quality,latency,security,resource,lineage⟩`):

```
artifacts/packages/{capability}_v1/
  descriptor.json      # name, task, quality{metric,value,risk_gap,eval_set}, latency{p50,p95},
                       #   security{privacy_label, verifier}, resource{base_model, adapter_type,
                       #   added_params, disk_mb, eval_vram_gb}
  adapter/             # adapter_config.json + adapter_model.safetensors + tokenizer
  verifier.json        # capability id + thresholds the serve-time verifier uses
  fallback_policy.json # {on: [verifier_fail, low_confidence], action: "teacher"}
  metrics.json         # full eval dump: base/student/teacher + frontier row
  lineage.json         # teacher path, base path, dataset path+hash, sft hash, train config hash,
                       #   git sha, created_at
```

`package/schema.py` defines/validates these; `test_package_roundtrip.py` exports then reloads.

---

## 13. Implementation phases (build in this order — each ends in a runnable check)

**Phase 0 — skeleton.** `pyproject.toml`, package, `cli.py` dispatch, `config.py` (YAML +
machine-merge), `registry.py`. `idn list-capabilities` works. `pytest` runs (empty-ish).
*Check: CLI help + list-capabilities print resolved paths.*

**Phase 1 — capability layer on the cheap task.** Implement `CapabilitySpec` for **sst2**
(easiest) and **cola**, plus `metrics.py` core. Unit-test parse + metric on fixtures.
*Check: `test_capabilities.py` green; metric on 20 hand-labeled rows is sane.*

**Phase 2 — models + eval (no training yet).** `models/loader.py` (load base, generate),
`eval/runner.py`. Run `idn eval` for **base student** and **teacher** on sst2 val (limit 200).
*Check: two eval jsons with believable accuracy; teacher > base.*

**Phase 3 — teacher gen + SFT build + training.** `teacher/generate.py`, `data/build_sft.py`,
`data/collator.py` (with prompt masking + its unit test), `training/train_lora.py`,
`models/lora.py` param accounting. Train one sst2 LoRA, eval it.
*Check: student+LoRA ≥ base student on sst2; trainable% logged; no fp16 used.*

**Phase 4 — frontier sweep + report.** `sweep/run_frontier.py`, `eval/report.py`. Run a small
grid (r ∈ {8,32}, target ∈ {qv,qkvo}) on sst2.
*Check: markdown table + frontier plot produced; Pareto points marked.*

**Phase 5 — flagship GSM8K.** Implement `gsm8k.py` (CoT target, `####` parsing). Generate
teacher CoT on 2000 train + 500 test. Build SFT (teacher targets). Run the full
frontier_v1 grid on the 3090 (offload teacher gen to H100 if slow).
*Check: frontier table for gsm8k; identify smallest config reaching teacher − ε.*

**Phase 6 — packaging + serving + remaining capabilities.** `package/` export, `serving/`
loader+verifier+fallback, optional FastAPI. Add `squad`, `cnn_dailymail`, then `mbpp` (sandboxed
exec) as time permits.
*Check: `idn export` produces a valid package; `idn serve` answers locally and falls back;
`test_package_roundtrip` green.*

**Phase 7 — final report.** Aggregate per-capability frontiers into one `artifacts/reports/`
summary answering the section-1 claim. Note where the small model does and does not reach the
teacher, and the cheapest config that does.

---

## 14. V1 acceptance criteria

1. End-to-end pipeline runs from `gen-teacher` → `report` for ≥3 capabilities incl. gsm8k.
2. Prompt-masking collator implemented and unit-tested.
3. All training/eval in bf16 or fp32 (no fp16); dtype recorded in lineage.
4. Frontier report exists per capability: metric vs added-params, with base + teacher lines and
   Pareto points.
5. For gsm8k and ≥2 others, report the smallest LoRA config that reaches `teacher_metric − ε`
   (or an honest statement that none does, with the closest config and the gap).
6. ≥1 CapabilityPackage exported and reloadable; serve path returns a receipt and falls back.
7. Tests green: capabilities metrics, collator masking, frontier math, package roundtrip.
8. Report explicitly states added_params as % of student and % of teacher for the chosen configs.

---

## 15. Risks & mitigations (learned from the prior run + this stack)

- **fp16 NaN / `!` repetition on this CUDA stack.** → bf16 default, fp32 fallback; assert dtype
  != fp16 in code with a clear error.
- **No prompt masking last time → weak learning.** → mask prompt tokens; unit-test it.
- **Synthetic-task overfit (prior benchmark was hand-written IDN JSON).** → V1 uses real labeled
  datasets with standard metrics, so numbers are externally comparable.
- **GSM8K may not move with tiny LoRA / short training.** → distill teacher **CoT** (not just
  the final number), allow r up to 64 and qkvo+mlp targets, 3 epochs; if still stuck, that is a
  *finding* (small-model math capacity limit) — report it, don't fake it.
- **Teacher generation slow on 3090.** → cache aggressively, cap train-target count (1–2k is
  enough for SFT), offload heavy generation to the H100.
- **MBPP code execution is unsafe/slow.** → subprocess + timeout + restricted env; ship it last.
- **3090 OOM with 7B + student simultaneously.** → never hold both in memory; eval teacher and
  student in separate processes/runs; load one model at a time.

---

## 16. V2 / V3 (explicitly out of V1 scope, design for them but don't build yet)

- **V2 — multi-capability composition:** Top-T adapter mixing across the per-capability adapters
  built in V1; measure boundary-sample gain and adapter conflict (cosine/sign). Reuses
  `package/` adapters as inputs.
- **V2 — region discovery from logs:** the PCA-routing-space + clustering path from the original
  `capability_extraction_implementation_plan.md` (features → PCA → KMeans → Mahalanobis routing).
  Layer it on top of the V1 extraction/packaging machinery to discover capabilities from
  unlabeled traffic instead of using dataset labels.
- **V3 — shared weight basis + sparse coefficients + residual:** factor the V1/V2 adapters into a
  shared basis for cheap composition and deployment, per PPT slides 4–6.
- **Serving at scale:** real router across multiple capability packages, state/KV caching,
  trust/lineage signing — the remaining IDN paper components.

These are intentionally deferred so V1 delivers the measurable core claim fast.

---

## 17. First concrete action for the coding agent

1. Implement Phase 0 + Phase 1 + Phase 2 against **sst2** (cheapest path to a real number).
2. Run: `idn eval` base student vs teacher on sst2 (limit 200) and confirm the gap is real.
3. Only then proceed to training (Phase 3) and the flagship (Phase 5).
Do not implement all six capabilities up front — get one full vertical slice working end to end,
then widen.

