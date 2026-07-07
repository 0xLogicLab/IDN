# IDN Capability Carving — Implementation & Validation Plan (V3 - White-Box Extraction)

> **Audience:** this document is written to be handed directly to a coding agent
> (Claude Code / Codex) to generate the implementation. Every module has a path,
> a responsibility, and a function/CLI contract.

> **Status:** V3 supersedes the V2 "compression toolbox" plan. V2's five techniques
> (task vector, sparse routing, layer pruning, mixed-precision quant, QLoRA) are demoted
> to **baselines**. The core of V3 is a single white-box pipeline:
> **locate → carve → heal → certify → deliver**.

---

## 0. Executive summary — from black-box compression to white-box carving

**What we have (empirical, in this repo):**
1. The 6 capabilities (CoLA, SST-2, SQuAD, GSM8K, CNN/DM, MBPP) **cluster in representation
   space** (`artifacts/analysis/dataset_embeddings_*`: linear CV accuracy ≈ 0.91) and show
   capability-contrasting activation structure (`visualize-activations` dashboards).
2. The 7B teacher significantly outperforms the 1.5B student
   (`artifacts/eval/`: GSM8K exact_match **0.873 vs 0.688** on the full test set).
3. Working infrastructure: capability registry, eval runner, model loader (+LoRA attach),
   activation/embedding analysis. No extraction code exists yet.

**Why V2 was wrong as a thesis:** every V2 technique treats the model as a black box and
buys size with accuracy ("within ε"). That is compression, not intelligence delivery, and
its novelty is low. The user-facing requirement is *teacher-level accuracy at the edge*,
which black-box compression cannot promise.

**The V3 thesis (the research claim):**

> An inference capability of a large LLM is a **causally localizable sub-computation**.
> It can be **located** with attribution-based circuit discovery (no training),
> **carved out** as a structurally repacked dense sub-model (no training),
> **healed** cheaply using the teacher's own activations as supervision (no labeled data),
> and **certified** for fidelity against the teacher (measurable guarantee).
> The certified artifact is a first-class **network object** that IDN delivers to the edge
> the way a CDN delivers content objects.

**Positioning vs prior art (know what is taken):**
- *Circuit extraction for standalone inference* exists at GPT-2 scale on toy tasks
  (AAAI'25, arXiv:2412.15750) — masking only, no healing, no generative reasoning, no system.
- *Attribution patching* (EAP, EAP-IG) exists as a discovery method — never used as the
  front end of a deployment pipeline at 7B+.
- *GraftLLM/SkillPacks* transfer capabilities as distilled parameter deltas — black-box,
  training-heavy.

**Our claimed novelty:**
1. First causal (attribution-based) capability extraction at 7B+ on **generative reasoning**
   (GSM8K, MBPP), with physical repacking (dense smaller model, real speedup).
2. **Carve-then-heal**: healing a carved sub-network with teacher activation matching on
   *unlabeled* domain prompts — vs training a foreign student (distillation) or not healing
   at all (circuit extraction). Hypothesis: same-geometry initialization makes healing
   converge orders of magnitude cheaper than distillation.
3. **Capability-as-object**: fidelity certificate + out-of-circuit detection + escalation
   valve + delta delivery. This is the IDN abstraction, and it is indifferent to how well
   any single capability carves.

**Either outcome publishes:** if reasoning carves to ≤50% with ≤2pt fidelity loss, we have
the method paper. If reasoning refuses to carve below 80%, the **extractability spectrum**
across the 6 capabilities (classification carves, reasoning doesn't, and the mechanistic
reason why) is itself the finding, and IDN falls back to delivering larger-but-certified
artifacts.

**Flagship tasks:** GSM8K (math reasoning), MBPP (code generation) — hardest cases.
**Sanity tasks:** SST-2, CoLA — where carving should clearly work; use them to debug the
pipeline before touching the flagships.

---

## 1. The core research questions

```
Given:
  - Teacher F_T (Qwen2.5-7B-Instruct) with strong performance on capability c
  - Unlabeled domain prompts X_c (no gold labels required)
  - Edge constraints: memory < M_edge, latency < L_edge

RQ1 (Locate):   Does capability c correspond to a sparse causal sub-graph of F_T?
                Measure: attribution mass concentration; faithfulness of top-k% sub-graph.

RQ2 (Carve):    Can the sub-graph be physically repacked into a dense model C_c with
                size(C_c) ≤ ρ · size(F_T) for ρ ∈ {0.3, 0.5, 0.7}?

RQ3 (Heal):     Can teacher-activation matching on X_c restore C_c to
                M_c(C_c) ≥ M_c(F_T) − ε at a training cost ≪ distilling a foreign
                student to the same quality?

RQ4 (Certify):  Can we attach a per-artifact fidelity certificate and a per-query
                out-of-circuit detector such that (edge answers) ∪ (escalated answers)
                matches teacher accuracy?

RQ5 (Spectrum): How does extractability (min ρ at fixed ε) vary across the 6
                capabilities, and does it correlate with the representation-space
                clustering we already measured?
```

**V3 success = concrete answers to RQ1–RQ4 on GSM8K + MBPP, and the RQ5 spectrum table.**

---

## 2. The pipeline — five stages, each a module with a contract

### 2.1 Stage LOCATE — causal attribution of the capability sub-graph

**Goal:** a causal importance score for every prunable unit of the teacher, per capability.

**Prunable unit granularity (structured, so carving is physically repackable):**
- attention heads (respecting Qwen2.5 **GQA**: KV heads are shared — prune query heads
  freely, prune a KV head only when all its group's query heads are pruned)
- FFN intermediate channels (rows of `gate_proj`/`up_proj`, columns of `down_proj`)
- whole layers (as a coarse fallback and for comparison with V2-style layer pruning)

**Method (in order of preference; implement 1 first, 2 as upgrade):**
1. **Gradient×activation attribution (EAP-style first-order):** for each unit u,
   score(u) = |Σ_x ∂L_c/∂a_u · a_u| where L_c is the task loss on the answer tokens
   (teacher-forced) and a_u the unit's activation. One forward+backward per example;
   200–500 examples per capability. Cheap enough for the 3090 with batch 1 + bf16 +
   gradient checkpointing; use H100 if OOM.
2. **EAP-IG (integrated gradients over a corruption path):** interpolate between clean
   embeddings and corrupted embeddings (shuffled/other-capability prompt) in m=5 steps;
   more faithful, ~5× cost.
3. **Validation on a subsample — activation patching (ground truth):** for the top-N and
   a random-N units, zero/mean-ablate and measure metric drop on 50 examples. Report
   rank correlation between attribution scores and ablation impact. This is the evidence
   that LOCATE is causal, not correlational (the key upgrade over the existing
   `visualize-activations`, which stays as exploratory evidence).

**Outputs:**
- `artifacts/carving/{capability}/locate/scores.safetensors` — per-unit scores
- `locate/report.json` — concentration stats (Gini/top-k% mass), ablation-validation
  rank correlation, config (n examples, method, seed)

**Modules:**
- `src/idn/extraction/units.py` — enumerate prunable units for a loaded model; GQA-aware
  head grouping; maps unit ↔ weight slices. **Everything else depends on this; write it
  first, with tests.**
- `src/idn/extraction/locate.py` — hooks, attribution passes, score aggregation
- `src/idn/extraction/ablate.py` — patching/ablation validation harness

### 2.2 Stage CARVE — threshold and physically repack

**Goal:** a dense, smaller, standalone HF-loadable model containing only the sub-graph.

**Method:**
1. Given scores + keep-ratio ρ (per unit type or global), select kept units.
   Enforce structural floors: ≥1 query head group per layer or drop the whole layer;
   FFN floor (e.g. keep ≥12.5% channels) or drop the layer's MLP.
2. **Repack**: slice q/k/v/o projections to kept heads (GQA-consistent), slice FFN
   matrices to kept channels, drop removed layers from the module list. Emit a new
   `config.json` (per-layer head counts / intermediate sizes need a small custom
   modeling wrapper — see `pruned_qwen.py`) + `safetensors`.
3. Record a **carve manifest**: exact unit ids kept, hash of teacher weights, ρ,
   scores version. This is what makes delta delivery and reproducibility work.

**Outputs:** `artifacts/carving/{capability}/carve_r{rho}/` — model dir + `manifest.json`.

**Modules:**
- `src/idn/extraction/carve.py` — selection + repacking
- `src/idn/models/pruned_qwen.py` — modeling wrapper that loads heterogeneous per-layer
  shapes (subclass Qwen2 modeling with per-layer head/channel counts from config)
- Immediately evaluable with the existing `idn eval` via a new `--artifact` flag.

**Expected (hypotheses to test, not promises):** carved-unhealed quality drops sharply
below ρ≈0.7; that drop is what HEAL must recover.

### 2.3 Stage HEAL — teacher-activation self-supervised repair

**Goal:** recover teacher-level in-domain quality without labeled data or long training.

**Method:**
1. Corpus: unlabeled domain prompts (the capability's train split *inputs only*, plus
   optionally teacher-generated prompts). **No gold labels anywhere in the loss.**
2. Loss = α·KL(student logits ‖ teacher logits on teacher-forced teacher outputs)
   + β·Σ_l MSE(student hidden_l, teacher hidden_l) over kept layers (project/align via
   the carve manifest's index maps; hidden dim is unchanged — only heads/channels went
   away, so residual stream aligns 1:1; for dropped layers match the next kept layer).
3. Optimizer scope (cheapest that works, escalate in order): (a) LoRA on kept projections,
   (b) only the seams (projections adjacent to removed units), (c) full carved model.
4. Budget cap: ≤ 2k steps, batch ≤ 8, bf16. Teacher runs inference-only (no grad) —
   7B teacher + ≤0.7·7B student fits the H100; on the 3090 precompute teacher
   logits/hiddens to disk for the corpus and train student-only.
5. **Efficiency claim experiment (RQ3):** matched-budget comparison — same loss, same
   corpus, same steps, applied to (i) carved sub-model init vs (ii) foreign 1.5B student.
   Plot quality vs training FLOPs. This chart is the core evidence for novelty claim #2.

**Outputs:** `artifacts/carving/{capability}/heal_r{rho}/` — healed weights (or adapter)
+ `training_log.json`.

**Modules:**
- `src/idn/extraction/heal.py` — activation-matching trainer (plain torch loop or HF
  Trainer; must run on py3.8/torch 2.0 constraints below)
- `src/idn/extraction/teacher_cache.py` — precompute + memory-map teacher logits/hiddens

### 2.4 Stage CERTIFY — fidelity certificate + out-of-circuit detection

**Goal:** turn "usually as good" into a measurable, shippable guarantee.

**Certificate (per artifact, computed once):**
- `agreement`: task-metric agreement with teacher on a held-out probe set (≥300 examples)
  — both absolute metric and per-example match rate
- `output_fidelity`: mean KL(student‖teacher) on probe outputs
- `faithfulness`: cosine similarity of intermediate hidden states on kept layers
  (does it reproduce the computation, not just the answers?)
- `scope`: embedding-space description of D_c (centroid + covariance of probe prompts,
  reusing the TF-IDF/embedding machinery in `analysis/embedding_viz.py`)

**Per-query out-of-circuit detector (runtime, cheap):**
- Mahalanobis distance of the query's pooled hidden state (early kept layer) to the
  D_c scope distribution; threshold calibrated so that in-domain FPR ≤ 5%.
- Optional second signal: student's own answer-token entropy.
- Decision: serve locally / **escalate to origin**. Escalation is the safety valve backing
  the certificate — not the core design. Report: end-to-end accuracy of
  (served ∪ escalated) vs teacher, and escalation rate (the "cache miss rate" of IDN).

**Modules:**
- `src/idn/extraction/certify.py` — certificate computation, probe-set management
- `src/idn/serving/gate.py` — runtime detector + escalation decision (pure function;
  actual serving loop can stay offline/simulated in V3)

### 2.5 Stage DELIVER — capability packaging (the IDN object)

**Goal:** the artifact + metadata as a self-contained deployable unit.

**CapabilityPackage layout:**
```
package/
  descriptor.json      # name, task, quality (certificate), latency, resource, lineage,
                       # scope, escalation policy — the paper's capability descriptor C
  manifest.json        # carve manifest (units kept, teacher hash, rho)
  weights/             # EITHER full carved+healed weights
  delta/               # OR mask + healed-weight diff vs the shared teacher base
                       # (edge nodes holding the base download only the delta)
```
- `src/idn/package/builder.py` — build/package/validate; size accounting for full vs delta
- Delta mode is the CDN story: incremental delivery of a capability ≈ size of healed
  diffs (target: ≤2GB), not 14GB.

---

## 3. Experimental design

**Per capability c ∈ {SST-2, CoLA (sanity) → GSM8K, MBPP (flagship) → SQuAD, CNN/DM}:**

**Baselines (V2 techniques, demoted; implement only the cheap ones):**
1. Full 7B teacher bf16 — quality bar (GSM8K already measured: 0.873)
2. 1.5B base — floor (GSM8K: 0.688)
3. 1.5B distilled with the SAME heal loss + budget — the critical comparison for RQ3
4. Magnitude-pruned 7B at matching ρ (same repacker, magnitude scores instead of causal)
   — isolates the value of LOCATE
5. Random-carve at matching ρ — sanity control
6. (optional) int4 GPTQ 7B — the pure-quantization point on the frontier

**Main sweep:** ρ ∈ {0.3, 0.5, 0.7} × {unhealed, healed}. 6 artifacts per capability.

**Metrics per artifact:** capability metric, certificate fields, disk GB, VRAM GB,
p50/p95 latency + throughput (existing eval runner already logs latency), heal cost
(FLOPs, wall-clock, steps).

**Headline figures:**
1. **Frontier plot** per flagship: size vs metric; causal-carve should dominate
   magnitude/random at equal ρ, and healed-carve should reach teacher−ε.
2. **Heal-efficiency plot**: quality vs training budget, carved-init vs foreign-1.5B.
3. **Extractability spectrum table**: min ρ at ε per capability + attribution
   concentration + clustering separability → the RQ5 correlation.
4. **End-to-end IDN row**: edge artifact + gate + escalation → accuracy = teacher±0,
   escalation rate e%, effective throughput gain.

---

## 4. Environment

**Local (dev + LOCATE + CARVE + small heals):**
- 1× RTX 3090 (24 GB).
- Envs: `control` conda env (`/home/sig/anaconda3/envs/control/bin/python`, py3.8.5,
  torch 2.0.0+cu117, transformers 4.46.3, peft 0.13.2) and `.venv_qwen35` (used by
  `scripts/cmd.sh`). Verify which env runs before writing training code; keep the
  trainer to plain-torch features available in torch 2.0.
- **Dtype rule:** bf16 or fp32 only (fp16 NaNs on this stack).
- 7B backward passes on 24GB: batch 1, grad checkpointing, no optimizer states needed
  for LOCATE (grads w.r.t. activations only, discard param grads).

**Cloud (full heals + teacher-cache generation + sweeps):**
- `ssh wanghl@172.22.11.235`, 1× H100 80GB.

**Models (relative to machine `model_dir`, loaded from disk):**
- Teacher: `Qwen2.5-7B-Instruct`  ·  Reference student: `Qwen2.5-1.5B-Instruct`

**Datasets:** the 6 capabilities under `configs/capabilities/*.yaml` (already working).

---

## 5. Repository structure (additions; V1 infra is kept as-is)

```
src/idn/extraction/            # NEW — the carving pipeline
  __init__.py
  units.py                     # prunable-unit enumeration, GQA groups, unit↔slice maps
  locate.py                    # grad×act and EAP-IG attribution
  ablate.py                    # patching validation of scores
  carve.py                     # selection + physical repacking
  heal.py                      # activation-matching trainer
  teacher_cache.py             # precompute teacher logits/hiddens to disk
  certify.py                   # certificate + probe sets
src/idn/models/
  pruned_qwen.py               # loads per-layer heterogeneous head/channel counts
src/idn/serving/
  gate.py                      # out-of-circuit detector + escalation decision
src/idn/package/
  builder.py                   # CapabilityPackage build/validate, full vs delta
configs/carving/
  locate.yaml  carve.yaml  heal.yaml  certify.yaml
artifacts/carving/{capability}/{locate,carve_r*,heal_r*,certificate,package}/
tests/
  test_units.py                # slice maps round-trip; GQA constraints
  test_carve_repack.py         # carved model loads, forward-pass parity when rho=1.0
  test_heal_loss.py            # loss decreases on tiny fixture
  test_certify.py              # certificate fields computed; gate thresholding
```

**Critical invariant test:** `carve(rho=1.0)` must reproduce the teacher's logits exactly
(repacking is a no-op at full keep). This validates `units.py` + `carve.py` end to end
before any science happens.

---

## 6. CLI surface (additions)

```bash
# Stage 1: attribute
idn locate --config configs/capabilities/gsm8k.yaml --model Qwen2.5-7B-Instruct \
  --method gradxact --num-examples 300 --output artifacts/carving/gsm8k/locate

# Validate attribution causally (subsample ablation)
idn locate-validate --scores artifacts/carving/gsm8k/locate --top-n 50 --random-n 50

# Stage 2: carve + repack
idn carve --scores artifacts/carving/gsm8k/locate --keep-ratio 0.5 \
  --output artifacts/carving/gsm8k/carve_r0.5

# Stage 3: heal (teacher self-supervision, no labels)
idn heal --carved artifacts/carving/gsm8k/carve_r0.5 --teacher Qwen2.5-7B-Instruct \
  --config configs/carving/heal.yaml --output artifacts/carving/gsm8k/heal_r0.5

# Stage 4: certify
idn certify --artifact artifacts/carving/gsm8k/heal_r0.5 --teacher Qwen2.5-7B-Instruct \
  --probe-size 300 --output artifacts/carving/gsm8k/certificate_r0.5.json

# Stage 5: package
idn package --artifact ... --certificate ... --mode delta --output .../package

# Eval any artifact through the existing runner
idn eval --config configs/capabilities/gsm8k.yaml --artifact artifacts/carving/gsm8k/heal_r0.5

# Sweep + report
idn sweep-carve --config configs/capabilities/gsm8k.yaml --keep-ratios 0.3,0.5,0.7 \
  --baselines magnitude,random --output artifacts/reports/gsm8k_carving_frontier.md
```

---

## 7. Implementation phases

**Phase 0 — infra checks (mostly done).** Eval + configs work; GSM8K baselines exist.
Remaining: MBPP + SST-2 baselines for 7B and 1.5B. *Gate: baseline JSONs exist for the
3 tasks used below.*

**Phase 1 — units + repack skeleton.** `units.py`, `carve.py`, `pruned_qwen.py`,
`idn carve`, tests. *Gate: rho=1.0 logit-parity test passes on Qwen2.5-1.5B (cheap),
then on 7B.*

**Phase 2 — LOCATE on SST-2 (sanity task).** `locate.py` grad×act on 1.5B first, then 7B.
`ablate.py` validation. *Gate: attribution-vs-ablation rank correlation clearly beats
random; carve@0.5 with causal scores beats magnitude scores on SST-2 unhealed.*

**Phase 3 — CARVE+eval GSM8K.** Run locate+carve on 7B/GSM8K at ρ∈{0.3,0.5,0.7};
eval unhealed. *Gate: the degradation curve exists — this is RQ1/RQ2 evidence
regardless of how good it looks.*

**Phase 4 — HEAL.** `teacher_cache.py` + `heal.py`; heal GSM8K@0.5 on H100.
Matched-budget foreign-1.5B comparison. *Gate: healed@0.5 within 5pts of 0.873 AND
carved-init beats foreign-init at equal budget — if the first fails but the second
holds, raise ρ and continue; if both fail, pivot to the spectrum finding (see §9).*

**Phase 5 — CERTIFY + gate.** `certify.py`, `serving/gate.py`; end-to-end
(edge+escalation) accuracy on a mixed in/out-of-domain query stream.
*Gate: combined accuracy ≥ teacher − 1pt with escalation rate reported.*

**Phase 6 — MBPP + baselines + frontier.** Repeat 3–5 on MBPP; run magnitude/random/
1.5B-distill baselines; generate the three headline figures. *Gate: figures exist.*

**Phase 7 — spectrum + packaging.** Remaining capabilities at coarse settings for the
RQ5 table; `package/builder.py` with delta mode; final report.

**Do not build all stages up front.** Phases 1–2 on the sanity task prove the machinery;
GSM8K@ρ=0.5 healed is the single most important experiment in the plan.

---

## 8. V3 acceptance criteria

1. **rho=1.0 parity test green** (repacking correctness) + unit tests green.
2. **LOCATE validated causally** on ≥2 capabilities (rank correlation vs ablation).
3. **Frontier plots** for GSM8K + MBPP: causal carve vs magnitude vs random vs 1.5B-distill,
   unhealed and healed.
4. **Heal-efficiency plot** (carved-init vs foreign-init at matched budget) for ≥1 flagship.
5. **≥1 certified artifact** per flagship: ≤8GB VRAM, certificate attached, end-to-end
   (with escalation) accuracy ≥ teacher − ε (ε from capability YAML).
6. **Extractability spectrum table** across ≥4 capabilities.
7. **≥1 CapabilityPackage in delta mode** with measured delivery size.

---

## 9. Risks and honest failure modes

- **Reasoning is diffuse (the central gamble).** GSM8K may not carve below ρ≈0.8.
  → The spectrum result is the publication then: which capabilities localize, which
  don't, and the mechanistic correlate (attribution concentration, clustering
  separability). IDN's abstraction is indifferent to ρ; a certified 0.8-carve is still
  a smaller, faster, deliverable object.
- **Attribution noise at 7B.** First-order EAP is known to be imperfect. → Always
  validate with ablation subsamples; upgrade to EAP-IG; ensemble over seeds/examples.
- **GQA structural coupling limits head pruning.** → `units.py` treats a KV group as
  partially prunable (query heads) and fully prunable (whole group); measure both.
- **Heal changes the computation instead of repairing it** (student drifts from the
  circuit, faithfulness drops while accuracy holds). → That weakens the "extraction"
  claim; report faithfulness alongside accuracy; keep β (hidden-matching) high enough.
- **torch 2.0/py3.8 limits** (no modern quant libs, older HF). → The core pipeline needs
  only plain torch + transformers. Quant baseline is optional; run modern-lib steps on
  the H100 env if needed.
- **Latency wins need dense repacking, not masks.** → That's why CARVE repacks physically;
  measure real p50/p95 with the existing runner, and report honestly if per-layer
  heterogeneity hurts kernel efficiency.

---

## 10. Connection to the IDN paper architecture

The pipeline instantiates the paper's **Intelligence Capability Abstraction**:

```
C = ⟨name, task, quality, latency, security, resource, lineage, scope, escalation⟩
```

- **quality** = the CERTIFY certificate (agreement, output_fidelity, faithfulness)
- **scope** = the D_c descriptor used by the gate — this is what makes routing
  *capability-aware* rather than load-aware
- **resource/lineage** = carve manifest + teacher hash → content-addressed artifacts,
  **delta delivery**, and versioned invalidation when the origin model updates
- **escalation** = the origin fallback policy — IDN's "cache miss" semantics

Deployment Controller places packages; Service Router uses `scope` + gate signals;
State/Cache Manager handles delta distribution and invalidation. V4 adds composition
(union of carve masks / adapter mixing for multi-capability nodes) and online routing.

---

## 11. First concrete actions for the coding agent

1. **Finish Phase 0:** run `idn eval` for 7B and 1.5B on MBPP and SST-2 (limits per YAML).
2. **Write `units.py` + tests** against Qwen2.5-1.5B (same architecture family as 7B,
   cheap to iterate): unit enumeration, GQA groups, unit↔slice maps.
3. **Write `carve.py` + `pruned_qwen.py`** and make the **rho=1.0 logit-parity test** pass.
4. **Implement `locate.py` (grad×act)** and run on SST-2/1.5B; validate with `ablate.py`.
5. **Only then** touch the 7B and GSM8K. Do not implement HEAL until Phase 3's
   degradation curve exists — its shape determines how much healing capacity to build.
