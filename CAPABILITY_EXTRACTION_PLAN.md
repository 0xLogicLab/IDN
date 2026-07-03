# IDN Capability Extraction — Implementation & Validation Plan (V2 - Extraction Focus)

> **Audience:** this document is written to be handed directly to a coding agent
> (Claude Code / Codex) to generate the implementation. Every module has a path,
> a responsibility, and a function/CLI contract.

> **Status:** UPDATED based on empirical findings that capabilities cluster in representation
> space and 7B significantly outperforms 1.5B across all tasks. The goal shifts from "train
> a small model" to "extract and isolate specific capabilities from the 7B for efficient
> edge deployment."

---

## 0. Executive summary — the paradigm shift

**What we learned:**
1. The 6 capabilities (CoLA, SST-2, SQuAD, GSM8K, CNN/DM, MBPP) **cluster in representation space**
   — math/code/summarization separate from classification/QA tasks.
2. The 7B teacher **significantly outperforms** the 1.5B student across all capabilities.
3. Training a 1.5B+LoRA to match 7B on all tasks simultaneously is hard and misses the IDN point.

**The updated goal (true IDN capability extraction):**

Extract the 7B's capability for **one specific task** (e.g., code generation on MBPP) into a
**deployable artifact** that:
- Matches the 7B's task-specific performance (within ε)
- Deploys in <50% of the 7B's memory/disk footprint
- Can run on edge nodes with limited resources (12–16GB GPUs)
- Does NOT need to handle other capabilities (intentionally specialized)

This artifact is an **IDN capability unit** — the smallest deployable intelligence that preserves
teacher-level performance on one capability and can be delivered to edge nodes that need only
that specific service.

**Flagship tasks:** GSM8K (math reasoning), MBPP (code generation) — the hardest capabilities
where the extraction techniques will be most tested.

---

## 1. The core research question

```
Given:
  - Teacher model F_7B with strong performance on capability c
  - Capability c is localized/separable (clustering evidence)
  - Edge deployment constraints: memory < M_edge, latency < L_edge

Can we extract a capability artifact C_c from F_7B such that:
  1. Quality:       M_c(C_c) >= M_c(F_7B) - ε
  2. Efficiency:    size(C_c) <= 0.5 × size(F_7B)
  3. Deployability: C_c runs in M_edge memory with L_edge latency
  4. Isolation:     C_c need not preserve performance on other capabilities
```

**V2 success = answering YES with concrete artifacts for ≥2 capabilities.**

---

## 2. Five extraction techniques to implement and compare

Each technique produces a capability artifact from the 7B. We compare them on the
**capability-preservation vs deployment-size frontier**.

### 2.1 Task Vector Extraction (inspired by task arithmetic / editing models)

**Core idea:** the 7B's capability for task `c` can be represented as a parameter delta
`ΔW_c = W_7B_finetuned_on_c - W_7B_base`. Extract and apply only that delta.

**Method:**
1. Take the pretrained 7B base (before instruction tuning, or use the instruct model as "base").
2. Fine-tune a copy on capability `c` only (e.g., GSM8K) for a few epochs.
3. Compute `ΔW_c = W_finetuned - W_base` for each layer.
4. The capability artifact = `W_base + λ·ΔW_c` where λ is a scaling factor (default 1.0).
5. **Compression:** store ΔW_c in low-rank (SVD) or quantized form (int8/int4).

**Deployment:** load base + compressed ΔW_c. If base is shared across capabilities, only the
small ΔW_c needs to be delivered per capability.

**Expected footprint:** base (14GB bf16) + ΔW_c (0.5–2GB depending on compression) = 14.5–16GB.
If base is pre-loaded, incremental delivery = 0.5–2GB.

**Implementation modules:**
- `src/idn/extraction/task_vector.py`: fine-tune on task, compute delta, SVD compress delta
- `src/idn/extraction/task_vector_apply.py`: load base + apply task vector at inference

### 2.2 Sparse Activation Routing (identify and use only capability-relevant neurons)

**Core idea:** for capability `c`, only a subset of neurons/attention heads are active/critical.
Route through only those, zeroing or skipping the rest.

**Method:**
1. **Neuron importance scoring:** run the 7B on capability `c` examples, collect activations.
   Score each neuron by: (a) activation magnitude, (b) gradient w.r.t. task loss, (c) ablation
   impact (zero it out, measure metric drop).
2. **Select top-k% neurons** per layer (e.g., keep 40% of FFN neurons, 60% of attention heads).
3. **Sparse routing mask:** at inference, multiply activations by a binary mask, or skip
   computation for masked-out neurons entirely.
4. **Optional: prune and repack** the weights to save memory (structured pruning).

**Deployment:** full model structure but with ~40–60% of weights zeroed/pruned. If pruned and
repacked, can achieve 50–70% memory reduction.

**Expected footprint:** 6–10GB (vs 14GB for full 7B bf16).

**Implementation modules:**
- `src/idn/extraction/neuron_importance.py`: score neurons via activation stats + gradients
- `src/idn/extraction/sparse_mask.py`: select top-k, export mask
- `src/idn/extraction/sparse_model.py`: load model + apply mask at inference (or prune weights)

### 2.3 Knowledge Localization + Layer Pruning (find which layers encode the capability)

**Core idea:** not all 28–32 layers of the 7B are equally important for capability `c`. Some
layers are generic (early layers = tokenization, late layers = output formatting); the
capability-specific knowledge may concentrate in the middle layers.

**Method:**
1. **Layer attribution:** run the 7B on capability `c`, measure each layer's contribution via:
   - Integrated gradients w.r.t. task loss
   - Ablation: freeze or remove layer `l`, measure metric drop
   - Probing: train a linear probe on each layer's hidden states to predict task success
2. **Identify critical layers** (e.g., layers 10–20 for math, 5–15 for code).
3. **Prune non-critical layers** or replace them with lightweight pass-through / identity.
4. The capability artifact = pruned 7B with ~50% fewer layers.

**Deployment:** a shallower model (e.g., 14–16 layers instead of 28) with the same width.

**Expected footprint:** ~50% of full 7B = 7–8GB. Inference speed ~2× faster.

**Implementation modules:**
- `src/idn/extraction/layer_attribution.py`: score layer importance via gradients/ablation/probing
- `src/idn/extraction/layer_pruning.py`: remove or skip non-critical layers, export pruned model
- `src/idn/models/pruned_loader.py`: load and run pruned architecture

### 2.4 Capability-Specific Mixed-Precision Quantization (aggressive quant on non-critical parts)

**Core idea:** not all weights are equally sensitive for capability `c`. Quantize non-critical
parts aggressively (int4/int3), preserve critical parts in higher precision (fp16/int8).

**Method:**
1. **Sensitivity analysis:** for each weight matrix (or even per-channel), measure how quantization
   affects capability `c` metric. Use Hessian trace, or empirical: quantize, measure drop.
2. **Assign precision budgets:** critical = fp16 or int8, non-critical = int4 or int3.
3. **Mixed-precision quantization:** store weights in variable bit-widths. Use llm.int8-style
   outlier preservation or GPTQ/AWQ for structured quantization.

**Deployment:** entire model fits in 4–6GB with careful mixed-precision. May need custom
CUDA kernels or use `bitsandbytes` / `autoawq` / `auto-gptq`.

**Expected footprint:** 4–6GB (vs 14GB bf16), ~2–3× inference speedup on compatible hardware.

**Implementation modules:**
- `src/idn/extraction/sensitivity.py`: per-weight or per-layer sensitivity scoring
- `src/idn/extraction/mixed_precision_quant.py`: quantize with per-part precision budgets
- `src/idn/models/mixed_prec_loader.py`: load and infer with mixed-precision weights

**Dependencies:** may require `bitsandbytes`, `auto-gptq`, or `autoawq` (install on demand).

### 2.5 LoRA Distillation into a Quantized/Pruned 7B Base (hybrid)

**Core idea:** combine techniques — start with a compressed 7B base (quantized to int4), then
train a task-specific LoRA on top. The base is shared, the LoRA is per-capability.

**Method:**
1. Quantize the full 7B to int4 (using GPTQ or AWQ) → base artifact ~3.5–4GB.
2. Train a LoRA (rank 16–64) on capability `c` using the quantized base.
3. The capability artifact = quantized base + LoRA adapter (50–200MB).

**Deployment:** load quantized base once, hot-swap LoRA adapters per capability. Incremental
delivery per capability = ~100MB.

**Expected footprint per capability:** 4GB (shared base) + 0.1GB (LoRA) = 4.1GB total, but
amortized if base is reused.

**Implementation modules:**
- `src/idn/extraction/quantize_base.py`: GPTQ/AWQ quantization of 7B to int4
- `src/idn/training/train_lora_on_quantized.py`: train LoRA on top of quantized base
- `src/idn/models/qlora_loader.py`: load quantized base + attach LoRA (similar to QLoRA)

**Dependencies:** `auto-gptq` or `autoawq`, `bitsandbytes` (if using NF4 like QLoRA).

---

## 3. Experimental design — the frontier comparison

For each capability `c` ∈ {GSM8K, MBPP, SQuAD, CNN/DM, CoLA, SST-2}:

**Baselines (fixed per capability):**
1. **Full 7B teacher (bf16):** the quality bar. Size = 14GB, latency = baseline.
2. **1.5B student base:** the deployment-size lower bound. Quality << 7B (from your findings).
3. **1.5B + LoRA (rank 64, all modules):** best-effort small model. Likely still << 7B quality.

**Extraction techniques (one artifact per technique per capability):**
1. Task vector (SVD-compressed)
2. Sparse activation routing (40%, 60% sparsity)
3. Layer pruning (keep 50%, 70% of layers)
4. Mixed-precision quantization (int4 non-critical, int8 critical)
5. Quantized base + LoRA (int4 base + rank 32 LoRA)

**Metrics collected per artifact:**
- **Quality:** capability metric (EM for GSM8K, pass@1 for MBPP, etc.)
- **Deployment size:** disk footprint (GB), VRAM usage (GB)
- **Latency:** p50/p95 inference time on 512-token prompts (3090 GPU)
- **Throughput:** requests/sec under load
- **Quality gap:** `M_c(artifact) - M_c(7B)` (negative = worse)

**Frontier plot:** X-axis = deployment size (GB), Y-axis = capability metric. Mark:
- Full 7B (top-right corner)
- 1.5B base (bottom-left)
- 1.5B+LoRA (left, hopefully higher than base)
- Each extraction artifact

**Success = identify ≥1 artifact per capability that:**
- Sits in the top-left zone (high quality, low size)
- Matches 7B within ε while using <8GB (fits on edge GPUs)

---

## 4. Environment (unchanged from V1)

**Local machine (primary dev + light extraction experiments):**
- 1× RTX 3090 (24 GB). Idle.
- Use the `control` conda env: `/home/sig/anaconda3/envs/control/bin/python`
  - Python 3.8.5, torch 2.0.0+cu117, transformers 4.46.3, peft 0.13.2, trl 0.11.4,
    datasets 2.21.0, accelerate 1.0.1, scikit-learn 1.0.
- **Add on-demand:** `auto-gptq`, `autoawq`, `bitsandbytes` (for quantization techniques).
- **Dtype rule:** bf16 or fp32 only (fp16 produces NaNs on this stack).

**Models (local, absolute paths):**
- Teacher: `/data/huggingface/model/Qwen2.5-7B-Instruct` (the extraction source)
- Reference student: `/data/huggingface/model/Qwen2.5-1.5B-Instruct` (baseline only)

**Datasets:** same 6 capabilities from V1 (paths unchanged, see V1 backup for details).

**Cloud machine (heavy extraction + full sweeps):**
- `ssh wanghl@172.22.11.235`, 1× H100 80GB.
- Use for: task vector fine-tuning (full 7B training), quantization (GPTQ calibration), large
  ablation sweeps.

---

## 5. Repository structure (additions to V1 structure)

```
/home/sig/files/IDN/
  ... (V1 structure: configs/, src/idn/capabilities/, data/, models/, training/, eval/, ...)
  
  src/idn/extraction/                    # NEW: the core extraction methods
    __init__.py
    task_vector.py                       # fine-tune on task, compute ΔW, SVD compress
    task_vector_apply.py                 # load base + apply task vector at inference
    neuron_importance.py                 # score neurons via activations + gradients
    sparse_mask.py                       # select top-k neurons, export mask
    sparse_model.py                      # load + apply sparse mask at inference
    layer_attribution.py                 # score layer importance (gradients/ablation/probing)
    layer_pruning.py                     # remove layers, export pruned model
    sensitivity.py                       # per-weight sensitivity for mixed-precision quant
    mixed_precision_quant.py             # quantize with per-part precision budgets
    quantize_base.py                     # GPTQ/AWQ full quantization of 7B
    utils.py                             # shared: SVD compress, load/save artifacts
  
  src/idn/models/
    pruned_loader.py                     # load pruned models (fewer layers)
    mixed_prec_loader.py                 # load mixed-precision quantized models
    qlora_loader.py                      # load quantized base + LoRA adapter
  
  configs/extraction/
    task_vector.yaml                     # hyperparams: finetune epochs, SVD rank, λ
    sparse_routing.yaml                  # sparsity levels, importance metric
    layer_pruning.yaml                   # layer selection strategy, ablation budget
    mixed_precision.yaml                 # sensitivity threshold, quant bits per tier
    qlora.yaml                           # base quantization method, LoRA rank
  
  artifacts/extraction/                  # extraction outputs per technique
    {capability}/
      task_vector/
        delta_compressed.safetensors     # SVD-compressed ΔW_c
        config.json                      # base model, λ, compression params
      sparse_routing/
        neuron_mask.pt                   # binary mask per layer
        importance_scores.json           # neuron scores for analysis
      layer_pruning/
        pruned_model/                    # model with removed layers
        layer_importance.json            # attribution scores
      mixed_precision/
        mixed_prec_model/                # weights in variable bitwidths
        precision_map.json               # per-layer/per-weight precision assignments
      qlora/
        quantized_base/                  # GPTQ/AWQ int4 base (shared)
        adapter/                         # task-specific LoRA
  
  tests/
    test_task_vector.py                  # apply task vector, check output shape
    test_sparse_mask.py                  # mask application, forward pass
    test_layer_pruning.py                # pruned model loads and runs
    test_mixed_prec.py                   # mixed-prec inference numerics
    test_qlora.py                        # load quantized base + adapter
```

---

## 6. CLI surface (additions to V1)

All V1 commands (gen-teacher, build-sft, train-lora, eval, export, serve, report) remain.
Add extraction-specific commands:

```bash
# Extract task vector for a capability
idn extract-task-vector \
  --config configs/capabilities/gsm8k.yaml \
  --base Qwen2.5-7B-Instruct \
  --output artifacts/extraction/gsm8k/task_vector \
  --finetune-epochs 2 --svd-rank 512 --lambda 1.0

# Score neuron importance for sparse routing
idn extract-sparse-mask \
  --config configs/capabilities/gsm8k.yaml \
  --model Qwen2.5-7B-Instruct \
  --output artifacts/extraction/gsm8k/sparse_routing \
  --sparsity 0.4 --importance-metric gradient

# Attribute and prune layers
idn extract-prune-layers \
  --config configs/capabilities/gsm8k.yaml \
  --model Qwen2.5-7B-Instruct \
  --output artifacts/extraction/gsm8k/layer_pruning \
  --keep-ratio 0.5 --attribution-method ablation

# Mixed-precision quantization
idn extract-mixed-precision \
  --config configs/capabilities/gsm8k.yaml \
  --model Qwen2.5-7B-Instruct \
  --output artifacts/extraction/gsm8k/mixed_precision \
  --sensitivity-threshold 0.01 --default-bits 4 --critical-bits 8

# Quantize base + train LoRA (QLoRA-style)
idn extract-qlora \
  --config configs/capabilities/gsm8k.yaml \
  --base Qwen2.5-7B-Instruct \
  --output artifacts/extraction/gsm8k/qlora \
  --quant-method gptq --lora-r 32

# Evaluate any extraction artifact (reuse eval command with artifact path)
idn eval \
  --config configs/capabilities/gsm8k.yaml \
  --artifact artifacts/extraction/gsm8k/task_vector \
  --artifact-type task_vector \
  --output artifacts/eval/gsm8k_task_vector.json

# Run the full extraction frontier sweep
idn sweep-extraction \
  --config configs/capabilities/gsm8k.yaml \
  --techniques task_vector,sparse_routing,layer_pruning,mixed_precision,qlora \
  --output artifacts/reports/gsm8k_extraction_frontier.md
```

---

## 7. Implementation phases (updated for extraction focus)

**Phase 0 — skeleton (reuse V1).** Package, CLI, config, registry, capabilities (at least
SST-2, CoLA, GSM8K, MBPP), eval infrastructure. *Check: eval baselines (7B, 1.5B, 1.5B+LoRA)
on GSM8K and confirm the gap.*

**Phase 1 — Task vector extraction.** Implement `extraction/task_vector.py` (fine-tune 7B on
task, compute delta, SVD compress), `task_vector_apply.py` (load + apply), eval harness
integration. Run on GSM8K.
*Check: task vector artifact ≤ 2GB, matches 7B within 5 pts on GSM8K.*

**Phase 2 — Sparse activation routing.** Implement `neuron_importance.py` (score via gradients
or activation magnitude on task examples), `sparse_mask.py` (select top-k), `sparse_model.py`
(masked inference). Try 40% and 60% sparsity on GSM8K.
*Check: 40% sparse model ≈ 8GB, quality gap ≤ 10 pts vs full 7B.*

**Phase 3 — Layer pruning.** Implement `layer_attribution.py` (integrated gradients or ablation),
`layer_pruning.py` (remove layers), `pruned_loader.py`. Prune to 50% and 70% of layers on GSM8K.
*Check: 50% pruned model ≈ 7GB, 2× faster inference, quality gap ≤ 15 pts.*

**Phase 4 — Mixed-precision quantization.** Implement `sensitivity.py` (measure per-weight impact),
`mixed_precision_quant.py` (int4 default, int8 sensitive parts), `mixed_prec_loader.py`. Run on
GSM8K. May require `auto-gptq` or custom kernels.
*Check: mixed-prec model ≈ 5GB, quality gap ≤ 8 pts, 2× speedup.*

**Phase 5 — QLoRA (quantized base + LoRA).** Implement `quantize_base.py` (GPTQ/AWQ int4 7B),
`train_lora_on_quantized.py`, `qlora_loader.py`. Quantize once, train LoRA per capability.
*Check: int4 base = 3.5GB, LoRA = 100MB, quality gap ≤ 5 pts on GSM8K.*

**Phase 6 — Frontier sweep + report.** Run all 5 techniques + baselines on GSM8K and MBPP (flagship).
Generate frontier plot (size vs metric). Identify Pareto-optimal artifacts.
*Check: report shows ≥1 technique per capability that fits in 8GB and matches 7B within ε.*

**Phase 7 — Packaging + multi-capability.** Extend to all 6 capabilities. Export best artifacts
as IDN CapabilityPackages with verifier + fallback. Final report with per-capability recommendations.

---

## 8. V2 acceptance criteria (updated)

1. **Baseline measurements exist** for all 6 capabilities: 7B teacher, 1.5B base, 1.5B+LoRA(rank64).
2. **≥3 extraction techniques implemented and working** (task vector, sparse routing, QLoRA minimum).
3. **Frontier plot generated** for GSM8K and MBPP showing size vs metric for all baselines + artifacts.
4. **≥1 artifact per capability** (GSM8K, MBPP minimum) that:
   - Fits in ≤8GB VRAM (edge-deployable)
   - Matches 7B within ε (GSM8K: 5pts, MBPP: 10pts pass@1, others: 2–5pts)
   - Is packaged with descriptor + verifier + fallback
5. **Report includes:** per-technique comparison table, frontier plot, deployment recommendations,
   limitations discussion (where extraction fails or degrades).
6. **Tests green** for: task vector round-trip, sparse mask application, pruned model forward pass,
   artifact packaging.

---

## 9. Expected outcomes and insights

**Best-case scenarios per technique:**
- **Task vector:** works best when capability is localized to a small parameter subspace; may fail
  if capability is diffuse or interacts with instruction-following heavily.
- **Sparse routing:** effective if capability uses a distinct subset of neurons; risky if task
  needs broad model capacity (e.g., long-context reasoning).
- **Layer pruning:** works if capability concentrates in middle layers; fails if early/late layers
  are critical (e.g., multilingual models).
- **Mixed-precision quant:** universal technique, safest; limited by how much you can compress
  without hurting the capability.
- **QLoRA:** most practical for multi-capability deployment (shared base); quality depends on how
  well LoRA can adapt the quantized base.

**Failure modes to document:**
- Extraction works on classification (CoLA, SST-2) but fails on generation (GSM8K, MBPP) → suggests
  capability is not cleanly isolatable, or requires full model capacity.
- All techniques degrade significantly → capability may be "emergent" from model scale, not extractable.
- One technique works, others don't → reveals structural properties of how the capability is encoded.

**The deliverable is the knowledge:** which extraction methods work for which capabilities, and
the concrete artifacts proving it.

---

## 10. Connection to the IDN paper architecture

These extracted capability artifacts map directly to the paper's **Intelligence Capability
Abstraction** component:

```
Capability descriptor C = ⟨name, task, quality, latency, security, resource, lineage⟩
```

- **name:** "gsm8k_math_reasoning"
- **task:** "solve grade-school math problems"
- **quality:** {metric: "exact_match", value: 0.78, teacher_gap: -0.03, eval_set: "gsm8k_test_500"}
- **latency:** {p50_ms: 450, p95_ms: 890}
- **security:** {privacy_label: "public", verifier: ["format", "confidence"]}
- **resource:** {method: "task_vector", base: "Qwen2.5-7B", vram_gb: 6.2, disk_mb: 1800}
- **lineage:** {teacher: "Qwen2.5-7B-Instruct", extraction_method: "task_vector_svd512", ...}

The extracted artifacts become the **deployable capability units** that IDN's Deployment Controller
places on edge nodes, Service Router directs requests to, and State/Cache Manager optimizes.

V3 will add multi-capability composition (adapter mixing, shared basis) and online routing — but
V2 must first prove that single-capability extraction is feasible and effective.

---

## 11. Risks and mitigations (updated)

- **Extraction may not work for all capabilities.** → That's a finding, not a failure. Document
  which capabilities are extractable and which aren't; this reveals model properties.
- **Quantization libraries (GPTQ, AWQ) may not install on py3.8 + torch 2.0.** → Try; if blocked,
  upgrade torch to 2.1+ in a separate env, or skip mixed-prec/QLoRA and focus on task vector +
  sparse routing.
- **Task vector fine-tuning is expensive (full 7B training).** → Run on H100; alternatively, use
  few-shot fine-tuning (200–500 examples, 1 epoch) as a cheaper approximation.
- **Sparse routing + pruning may require custom CUDA kernels for real speedup.** → For V2, measure
  size reduction (primary goal); latency improvement is secondary. Note in report if structured
  pruning would help but wasn't implemented.
- **Clustering observation may not translate to extractability.** → Clustering in representation
  space != capability is localized in parameters. The experiment tests this hypothesis.

---

## 12. First concrete actions for the coding agent

1. **Reuse V1 infrastructure:** capabilities, eval, baselines. Run: `idn eval` on 7B, 1.5B,
   1.5B+LoRA for GSM8K (limit 500). Confirm the gap. *Do not proceed until this is done.*
2. **Implement Phase 1 (task vector) for GSM8K:** fine-tune 7B on gsm8k train (1000 examples, 1–2
   epochs), compute ΔW, SVD compress (rank 512), apply at eval. Target: artifact ≤ 2GB, gap ≤ 5pts.
3. **Only after task vector works, implement Phase 5 (QLoRA):** quantize 7B to int4, train LoRA
   rank 32 on GSM8K. This is the most practical technique and should be validated early.
4. **Generate the first frontier plot:** 7B baseline, 1.5B+LoRA, task vector, QLoRA. Four points,
   one plot. This is the proof-of-concept that guides the rest.
5. **Then widen:** add sparse routing (Phase 2), evaluate on MBPP, expand to other capabilities.

**Do not implement all 5 techniques across all 6 capabilities up front.** Validate the hardest
one (GSM8K) with 2 techniques first, prove the concept, then scale.
