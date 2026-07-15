# IDN Hotspot Capability Packages — Implementation & Validation Plan (V4 - LoRA Basis Compilation)

> **Audience:** this document is written to be handed directly to a coding agent
> (Claude Code / Codex) to generate the implementation. Every module has a path,
> a responsibility, and a function/CLI contract.

> **Status:** V4 supersedes V3 (white-box carving) and V2 (compression toolbox).
> The technical route follows `局部能力提取方案_PPT大纲.md` and uses its notation.
> Core claim: the edge model does not globally imitate the teacher; it **routes each
> request into the correct local parameter state** (a compiled LoRA), and falls back
> to the teacher outside trusted regions.

---

## 0. Executive summary

**Setup:** teacher `F_T` = Qwen2.5-7B-Instruct (cloud), edge base `F_θ0` =
Qwen2.5-1.5B-Instruct (frozen), capabilities = the 6 configured tasks
(cola, sst2, squad, gsm8k, cnn_dailymail, mbpp).

**Known baselines (this repo):** GSM8K exact_match — 7B: **0.873**, 1.5B: **0.688**
(`artifacts/eval/`). Representation-space clustering of the 6 capabilities is strong
(linear CV accuracy ≈ 0.91, `artifacts/analysis/`), which is the premise for routing.

**The pipeline (five stages):**

1. **Specialize** — for each capability `k`, fine-tune a LoRA `Δ_k` on the frozen 1.5B.
   Each `⟨routing boundary, Δ_k⟩` is a **hotspot capability package (热点能力包)** `C_k`.
2. **Route** — embed each request query-only, project by PCA, keep per-capability
   centroids `μ_k, Σ_k`; a request gets Mahalanobis distances `d_k(x)` and soft weights
   `r_k(x) = softmax(−d_k/T)`.
3. **Naive mixing (expected to fail, and that's the point)** — form
   `Δ(x) = Σ_k r_k(x)·Δ_k` by direct weight averaging. Independently trained LoRAs live
   in inconsistent low-rank subspaces, so this should degrade. We run it anyway: the
   measured conflict (cosine/sign conflict vs quality drop) is the *evidence* that
   motivates stage 4.
4. **Shared basis compilation (the fix)** — per layer/module, flatten all K deltas,
   SVD to get a shared weight basis `{B_m}`; express each capability as sparse
   coefficients `c_k` plus a low-rank residual `R_k`. At inference, mix in
   **coefficient space**: `c(x) = Σ_k r_k(x)·c_k`, then compile
   `Δ(x) = Σ_m c_m(x)·B_m + Σ_{TopT} r_k(x)·R_k`. Composition becomes coordinate
   arithmetic instead of weight surgery.
5. **Police** — an Accept test decides per request: serve locally (hard route / compiled
   mix), or **fall back to the 7B**. Monitors decide when to *refresh* a package and
   how much rank each capability deserves.

**End goal:** most requests served by hotspot packages on the 1.5B, a small fraction
falls back; selective accuracy ≈ 7B; latency ≪ 7B.

**Why this is not "just LoRA fine-tuning":** the deliverable is not K adapters but a
**capability cache** — discovered regions, routing boundaries, a shared coordinate
system that makes capabilities composable, selective-risk guarantees, and a fallback +
refresh loop. Fine-tuning is one stage of a delivery system (see PPT outline Slide 6
comparison table).

---

## 1. Formal objective (from the PPT outline)

Not global imitation `F_θ0(x) ≈ F_T(x) ∀x`, but **local conditional approximation**:

```
F_{θ0+Δ(x)}(x) ≈ F_T(x)   for x ∈ ∪_k D_k  (trusted regions)

minimize   E[Cost(ρ(x), x)]                     (routing policy ρ)
subject to R(F_local | Accept) − R(F_T | Accept) ≤ ε      (selective risk)
           P(Accept = 1) ≥ γ                              (coverage)
           latency/memory ≤ edge budget
```

**Primary metrics:** SelectiveRisk, Coverage (= edge hit rate), fallback ratio,
end-to-end accuracy vs 7B, p50/p95 latency, adapter compile/switch time.

---

## 2. Stage SPECIALIZE — per-capability LoRA + accuracy table

**Method:**
1. For each capability `k` ∈ {cola, sst2, squad, gsm8k, cnn_dailymail, mbpp}:
   train LoRA `Δ_k` on the frozen 1.5B base using the capability's train split.
   - v0 supervision: gold labels (simplest, data already local).
   - v1 upgrade (optional): teacher distillation — SFT on `F_T(x_i)` outputs
     (PPT §3.2), which is the IDN-native supervision (no labels needed).
2. All adapters start from the SAME `θ0` and the SAME target modules — this is a hard
   requirement for stage 4 (deltas must be comparable coordinates).
   - Default config: rank r=16, α=32, dropout 0.05, targets = all attention
     projections + MLP (`q,k,v,o,gate,up,down`); bf16; 2–3 epochs.
3. Evaluate each `θ0+Δ_k` on its capability via the existing `idn eval --adapter`.

**Deliverable — the V4 anchor table (fill per capability):**

| capability | 7B | 1.5B base | 1.5B+LoRA_k | gap closed |
|---|---|---|---|---|
| gsm8k | 0.873 | 0.688 | ? | ? |
| ...   | ...   | ...   | ? | ? |

Also record **cross-capability quality**: `θ0+Δ_k` evaluated on capability `j≠k`
(6×6 matrix). This measures interference and later justifies routing.

**Modules:**
- `src/idn/training/train_lora.py` — SFT trainer (peft 0.13.2, trl 0.11.4 available);
  reads capability YAML; saves adapter + `train_meta.json` (base hash, rank, targets,
  seed). Wire the existing `train-lora` CLI stub to it.
- Rank sweep support: `--rank {4,8,16,32,64}` (used by stage 6's budget policy).

**Hardware:** 1.5B LoRA training fits the 3090 easily; 7B is inference-only (teacher
outputs + eval). H100 for sweeps.

---

## 3. Stage ROUTE — query-only routing space

**Method (PPT Slide 2):**
1. Representation `h(x) = ψ(x)`: mean-pooled last-hidden-state of the **1.5B base**
   over prompt tokens (query-only; no teacher call; reuse
   `analysis/embedding_viz.py` sampling machinery, but ψ must be the model embedding,
   not TF-IDF — the router runs at the edge with the model already loaded).
2. Fit PCA on training-split representations of all capabilities:
   `z = P_m^T (h − h̄)`, m ≈ 32–64.
3. Per capability: centroid `μ_k`, covariance `Σ_k` (shrinkage-regularized),
   Mahalanobis `d_k(x)`, soft weights `r_k(x) = softmax(−d_k(x)/T)`.
4. Router health metrics: routing accuracy on held-out eval splits (argmax_k r_k vs
   true capability), coverage/OOD rejection on out-of-domain probes (e.g. shuffled
   text, other-language prompts).

**Modules:**
- `src/idn/routing/space.py` — fit/save/load routing space
  (`P_m, h̄, {μ_k, Σ_k}, T`) → `artifacts/routing/space.npz`
- `src/idn/routing/router.py` — `route(x) → {r_k}, d_k, entropy H(r)`; pure numpy at
  inference.

*Gate: routing accuracy ≥ 0.9 on held-out splits (clustering evidence says this is
attainable); report the confusion matrix.*

---

## 4. Stage MIX-NAIVE — weighted LoRA merge (the documented negative result)

**Method (PPT mode B):** `Δ(x) = Σ_{k∈TopT} r_k(x)·Δ_k` merged in weight space
(per-module: sum of `B_k A_k` products scaled by r_k). T ∈ {1 (=hard route), 2, 3}.

**Measure:**
- Quality per capability under T=2,3 vs hard route (T=1).
- **Conflict diagnostics** (PPT §6.3): pairwise `cos(Δ_i, Δ_j)` and sign-conflict rate
  per module — the quantitative story for *why* naive mixing fails.

**Modules:**
- `src/idn/compose/naive_mix.py` — materialize merged ΔW from multiple adapters +
  weights; apply to base for eval.
- `src/idn/compose/conflict.py` — cosine/sign-conflict metrics between adapters.

*Expected outcome: T≥2 underperforms hard routing. This is a required experiment, not
a detour — it isolates the contribution of stage 5.*

---

## 5. Stage BASIS — shared weight basis, coefficients, residual (the core)

**Decomposition (PPT Slide 4; done per layer `l`, per target module):**
1. Materialize dense deltas `ΔW_{k,l} = B_k A_k` for all K capabilities.
2. Flatten `a_{k,l} = vec(ΔW_{k,l})`; stack to `A_l ∈ R^{K×d}`.
3. SVD on centered stack: `A_l − Ā_l = U Σ V^T`; shared basis
   `B_{l,m} = unvec(V[:,m])`, m = 1..M_l. Store basis in low-rank factored form when
   memory demands (each B_{l,m} is same shape as ΔW; keep M_l small, e.g. 4–8, and/or
   store SVD-factored).
4. Coefficients: `c_k = argmin_c Σ_l ‖ΔW_{k,l} − Σ_m c_{k,l,m} B_{l,m}‖_F² + λ‖c‖_1`
   (lasso per layer; K is tiny so this is cheap).
5. Residual: `R_{k,l} = ΔW_{k,l} − Σ_m c_{k,l,m} B_{l,m}`, compressed low-rank via SVD
   to rank r_res (e.g. 4–8); optionally refined on task loss (PPT §4.5, v1 upgrade).

**Reconstruction sanity (must-pass test):** with M_l = K−1 + residual, the compiled
`Δ_k` must reproduce single-capability LoRA quality within noise. Report quality vs
(M_l, r_res) — the compression/quality frontier of the basis itself.

**Inference-time compilation (PPT mode C):**
```
c_m(x) = Σ_{k∈TopT} r_k(x)·c_{k,m}      # mix coordinates, not weights
ĉ(x)   = TopS(c(x))                      # hard-sparsify
R(x)   = Σ_{k∈TopT} r_k(x)·R_k           # residual only over TopT
Δ(x)   = Σ_m ĉ_m(x)·B_m + R(x)
y      = F_{θ0+Δ(x)}(x)
```

**A hotspot capability package is now:**
```
C_k = ⟨ μ_k, Σ_k, τ_k,            # routing boundary
        c_k, R_k,                  # coordinates in shared basis + residual
        V_k, Π_k,                  # verifier, fallback policy
        lineage ⟩                  # base hash, teacher version, train date
```
The shared basis `{B_m}` ships once per edge node (like a shared base layer);
per-capability delivery = coefficients (KB) + residual (MB). **This is the CDN story:
delivering a new capability to a node that already holds the basis costs megabytes.**

**Modules:**
- `src/idn/compose/basis.py` — build basis from adapter library; save
  `artifacts/basis/{basis.safetensors, coeffs.json, residuals.safetensors}`
- `src/idn/compose/compile.py` — `compile(r, TopT, TopS) → Δ(x)` applied as a
  temporary adapter; must support per-request compile ≤ tens of ms (measure it)
- `tests/test_basis_roundtrip.py` — reconstruction sanity above

*Gate: basis-compiled mixing ≥ naive mixing on boundary samples AND ≈ hard route on
pure samples. If basis ≤ naive everywhere, the decomposition or λ/M_l needs revisits.*

---

## 6. Stage POLICE — accept, fallback, refresh, and rank budgeting

**Accept (PPT §5.2):**
```
Accept(x) = 1{ max_k r_k(x) ≥ τ  ∧  H(r(x)) ≤ τ_H  ∧  V(x,y) = pass }
otherwise → fallback to F_T (7B)
```
- `V` per capability: format/schema check (all tasks), answer-extraction success
  (gsm8k), code executes + smoke asserts (mbpp), length/overlap sanity (cnn_dailymail),
  label-in-vocab (cola/sst2, squad span plausibility). Cheap, rule-based first.
- Calibrate `τ, τ_H` on a validation stream to hit target coverage γ while keeping
  SelectiveRisk ≤ ε (sweep the operating curve, pick per deployment).

**Decision ladder per request:** hard route (max r_k dominant) → basis-compiled mix
(boundary: top-2 close) → fallback (low confidence / high entropy / verifier fail).

**Refresh policy (when to update a package):**
- Drift monitor: rolling mean of `min_k d_k(x)` and coverage; sustained rise → routing
  space refit (cheap) or package retrain.
- Quality monitor: verifier failure rate per capability; sampled shadow audits
  (send p% of accepted requests to 7B asynchronously, compare) → if agreement drops
  below certificate value, mark package stale and retrain from fresh teacher outputs.
- Teacher upgrade: lineage.teacher_version change invalidates all packages (CDN-style
  versioned invalidation).

**Rank budgeting (每个 LoRA 用多少参数):**
- Run the stage-2 rank sweep {4,8,16,32,64} per capability; fit quality-vs-rank curves.
- Allocate rank ∝ marginal utility under a total parameter budget (knapsack over the
  measured curves). Hypothesis: classification (cola/sst2) saturates at r≤8; gsm8k/mbpp
  need r≥32. The resulting allocation table is a paper-ready artifact.

**Modules:**
- `src/idn/serving/gate.py` — accept test + decision ladder (pure function)
- `src/idn/serving/verifiers.py` — per-capability V_k
- `src/idn/serving/monitor.py` — drift/quality counters, refresh triggers (offline
  simulation in V4; no live serving loop required)
- `src/idn/serving/simulate.py` — end-to-end stream simulator (see §7 E5)

---

## 7. Experimental design

**Request stream for end-to-end tests:** interleave held-out eval examples of all 6
capabilities (proportions ~ uniform, plus 5–10% OOD probes). Seeded, reproducible.

- **E1 Specialist table:** per-capability 7B / 1.5B / 1.5B+Δ_k + 6×6 cross matrix.
- **E2 Router:** routing accuracy, confusion, OOD rejection, coverage curve vs τ.
- **E3 Mixing ablation (headline figure 1):** hard route vs naive TopT vs basis-compiled,
  on (a) pure per-capability sets, (b) boundary set (samples with top-2 r_k within δ);
  plus conflict diagnostics correlating with naive-mix drop.
- **E4 Basis economics (headline figure 2):** quality vs (M_l, r_res, TopS); package
  delivery size (coeffs+residual) vs full LoRA vs full model; compile-time p50/p95.
- **E5 End-to-end IDN (headline table):** stream through the full ladder →
  Coverage, fallback ratio, SelectiveRisk vs 7B, end-to-end accuracy
  (edge ∪ fallback) vs 7B-only, p50/p95 latency and throughput vs 7B-only.
  Target: accuracy ≈ 7B ± ε with the majority of requests served at the edge.
- **E6 Rank budget:** quality-vs-rank curves + allocation table.

**Baselines:** edge-only 1.5B; cloud-only 7B; single global LoRA on the union of all 6
train sets (the "普通微调" strawman that must lose to routed specialists);
per-capability hard route (upper anchor for mixing).

---

## 8. Environment

- **3090 (24GB, local):** all 1.5B LoRA training, routing space, basis build, compiled
  inference, eval of 1.5B variants. 7B eval fits in bf16 (14GB) for accuracy rows.
- **H100 (`ssh wanghl@172.22.11.235`):** rank sweeps, teacher-distillation output
  generation, 7B latency baselines under load.
- Env: `control` conda env (py3.8.5, torch 2.0.0+cu117, transformers 4.46.3,
  peft 0.13.2, trl 0.11.4). **bf16/fp32 only (fp16 NaNs).**
- Models relative to machine `model_dir`: `Qwen2.5-7B-Instruct`, `Qwen2.5-1.5B-Instruct`.
- Datasets: existing `configs/capabilities/*.yaml`; loaded from local disk.

---

## 9. Repository structure (additions)

```
src/idn/training/train_lora.py         # stage 2 (wires existing CLI stub)
src/idn/routing/{space.py, router.py}  # stage 3
src/idn/compose/{naive_mix.py, conflict.py, basis.py, compile.py}   # stages 4–5
src/idn/serving/{gate.py, verifiers.py, monitor.py, simulate.py}    # stage 6
src/idn/package/builder.py             # C_k packaging (descriptor + coeffs + residual)
configs/lora/default.yaml              # rank/targets/epochs; per-capability overrides
configs/routing/space.yaml             # ψ layer, PCA dim, T, shrinkage
configs/compose/basis.yaml             # M_l, λ, r_res, TopT, TopS
artifacts/adapters/{capability}/r{rank}/
artifacts/routing/space.npz
artifacts/basis/
artifacts/packages/{capability}/
artifacts/reports/
tests/test_train_smoke.py              # tiny-steps LoRA run
tests/test_router.py                   # routing accuracy on fixture embeddings
tests/test_naive_mix.py                # merged ΔW matches manual computation
tests/test_basis_roundtrip.py          # reconstruction sanity (§5)
tests/test_gate.py                     # accept ladder decisions on synthetic r/H/V
```

---

## 10. CLI surface (additions)

```bash
idn train-lora --config configs/capabilities/gsm8k.yaml --rank 16 \
  --output artifacts/adapters/gsm8k/r16          # stage 2 (existing stub, now real)

idn fit-routing --config-dir configs/capabilities --output artifacts/routing/space.npz
idn route --input "..." --space artifacts/routing/space.npz   # debug: prints r_k, H

idn build-basis --adapters artifacts/adapters/*/r16 --M 6 --r-res 8 \
  --output artifacts/basis                        # stage 5 decomposition

idn eval --config configs/capabilities/gsm8k.yaml --model Qwen2.5-1.5B-Instruct \
  --adapter artifacts/adapters/gsm8k/r16          # existing runner, unchanged

idn eval-mixed --mode {hard,naive,basis} --top-t 2 --stream mixed_eval.jsonl \
  --output artifacts/reports/e3_mixing.json       # E3

idn simulate --stream mixed_eval.jsonl --tau 0.7 --tau-h 0.9 \
  --fallback-model Qwen2.5-7B-Instruct \
  --output artifacts/reports/e5_end2end.json      # E5

idn package --capability gsm8k --basis artifacts/basis \
  --output artifacts/packages/gsm8k               # C_k builder
```

---

## 11. Implementation phases

**Phase 1 — specialists (E1).** `train_lora.py` + smoke test; train r16 LoRAs for
cola, sst2, gsm8k first (fast tasks + one flagship); eval + cross matrix.
*Gate: each Δ_k beats 1.5B base on its own capability; gsm8k gap to 7B measurably
narrows. Do not proceed if LoRA training doesn't reproduce expected gains.*

**Phase 2 — router (E2).** `space.py`/`router.py`, fit on all 6, held-out routing
accuracy + OOD probes. *Gate: ≥0.9 routing accuracy.*

**Phase 3 — naive mixing (E3a).** `naive_mix.py` + `conflict.py`; hard vs T=2,3 on
pure + boundary sets. *Gate: results + conflict diagnostics recorded (negative result
expected and acceptable).*

**Phase 4 — shared basis (E3b, E4).** `basis.py`/`compile.py` + roundtrip test; rerun
E3 with basis compilation; measure compile latency and package sizes.
*Gate: basis ≥ naive on boundary samples; roundtrip sanity passes.*

**Phase 5 — gate + end-to-end (E5).** `gate.py`/`verifiers.py`/`simulate.py`; calibrate
τ, τ_H; run the mixed stream with 7B fallback.
*Gate: end-to-end accuracy ≥ 7B − ε with coverage ≥ γ (set ε per capability YAML,
γ target 0.7–0.8 initially).*

**Phase 6 — full 6 capabilities + budget + packaging (E6).** Remaining LoRAs
(squad, cnn_dailymail, mbpp), rank sweep + allocation, monitor/refresh simulation,
`package/builder.py`, final report with headline figures.

---

## 12. Acceptance criteria

1. **E1 table complete** for all 6 capabilities (+ cross matrix); every specialist
   beats base on its own task.
2. **Router ≥0.9** held-out routing accuracy with calibrated OOD rejection.
3. **The three-way mixing comparison exists** (hard / naive / basis) with conflict
   diagnostics; basis-compiled beats naive on boundary samples.
4. **Basis roundtrip test green;** package delivery size and compile time reported.
5. **End-to-end stream result:** accuracy ≥ 7B − ε, coverage ≥ γ, fallback ratio and
   p50/p95 latency reported; latency of edge-served requests ≪ 7B baseline.
6. **Rank-budget allocation table** from measured quality-vs-rank curves.
7. **≥1 packaged C_k** (descriptor + coefficients + residual + verifier + lineage).
8. Tests green: train smoke, router, naive-mix correctness, basis roundtrip, gate.

---

## 13. Risks and honest failure modes

- **Specialist LoRAs may not close enough of the 7B gap on gsm8k/mbpp** (1.5B capacity
  ceiling). → Coverage/γ absorbs this: route hard cases to fallback; report the
  achievable (coverage, selective-risk) frontier honestly. If gsm8k specialist gains
  are negligible, raise its fallback share and say so.
- **Routing space built from 6 benchmark datasets ≠ real traffic.** Clusters here are
  artificially clean. → Include OOD probes + boundary-sample analysis; frame results
  as "given a capability taxonomy"; real-traffic discovery is future work (PPT §2.4
  extractability checks are the hook).
- **Basis from only K=6 deltas is thin** — SVD over 6 vectors gives ≤5 informative
  directions above the mean. → That may already suffice (report spectrum); if not,
  enrich the library (per-rank variants, sub-capability splits, e.g. per-cluster within
  a dataset) to make the basis meaningful. This is a known scientific risk of the
  decomposition story.
- **Naive mixing might accidentally work** (LoRAs from the same θ0 are sometimes
  benignly mergeable). → Also fine: then basis compilation's claim shifts to delivery
  economics (KB-scale coefficients) + stability at larger K; measure both.
- **Compile-per-request overhead could eat the latency win.** → Cache compiled Δ for
  hard-route requests (only boundary samples need per-request compilation); measure
  compile p95 explicitly (E4).
- **Related work adjacency:** LoRAHub / AdapterFusion / model-soup lines compose
  adapters; TIES/DARE address sign conflicts. Our differentiators: query-only
  Mahalanobis routing + shared-basis *compilation* + selective-risk/fallback system +
  package delivery economics. Cite and compare (LoRAHub-style composition is already
  in the PPT baseline list); do not claim adapter composition per se as novel.

---

## 14. Connection to the IDN paper architecture

- **Hotspot capability package `C_k`** = the paper's capability descriptor: routing
  boundary (μ_k, Σ_k, τ_k), executor coordinates (c_k, R_k), verifier V_k, fallback
  policy Π_k, lineage.
- **Shared basis {B_m}** = the node-resident substrate; per-capability delivery is
  KB–MB scale → CDN-style incremental distribution, content-addressed by lineage,
  invalidated on teacher upgrade.
- **Gate + monitors** = Service Router (capability-aware routing) + State/Cache
  Manager (refresh/invalidation); fallback = origin pull.
- **Selective risk + coverage** = the SLA language of intelligence delivery: users buy
  "teacher-equivalent accuracy on accepted requests, ≥γ acceptance," not "a smaller
  model that's usually okay."

V5 extensions: real-traffic region discovery (unsupervised D_k), online coefficient
adaptation, multi-node placement optimization.

---

## 15. First concrete actions for the coding agent

1. **Implement `train_lora.py`** + smoke test; wire the `train-lora` CLI stub.
2. **Train 3 specialists** (cola, sst2, gsm8k; r16) and produce the E1 rows including
   the cross matrix — confirm specialists work before any composition code.
3. **Build the router** (`space.py`, `router.py`) with ψ = 1.5B pooled hidden states;
   verify ≥0.9 routing accuracy.
4. **Naive mixing + conflict diagnostics** (E3a), then **basis decomposition +
   roundtrip test + basis mixing** (E3b).
5. **Gate + simulator** for the end-to-end stream (E5) with 7B fallback.
6. Only then: remaining 3 capabilities, rank sweeps, packaging, final report.
