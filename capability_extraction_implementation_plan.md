# IDN 能力抽取实现方案

**目标**：把云端 teacher model 的高频、稳定、可验证局部能力抽取成可部署的 IDN capability package，使边缘小模型在命中的局部请求区域内接近 teacher，并在不确定时回退云端。

**依据**：
- `idn-paper.tex`：IDN 六个组件中，能力抽取对应 `Intelligence Capability Abstraction`，产物要能被 Capability Registry、Deployment Controller、Service Router、State/Cache Manager、Trust Manager 使用。
- `局部能力提取方案_PPT大纲.md`：能力抽取由局部区域发现、specialist adapter 学习、共享权重基、在线路由、验证与回退组成。

---

## 1. 能力抽取在 IDN 架构中的位置

IDN 论文把 capability 定义为：

\[
C=\langle name,task,quality,latency,security,resource,lineage\rangle
\]

工程上需要扩展成可执行能力包：

```text
CapabilityPackage C_k = {
  descriptor.json,          # IDN 能力描述符
  router.pkl,               # query-only 表示空间、PCA、cluster 边界
  adapter/,                 # LoRA/adapter 权重 Delta_k
  verifier.json|py,         # schema / task / safety / confidence verifier
  fallback_policy.json,     # 回退云端策略
  metrics.json,             # risk gap / coverage / latency / memory
  lineage.json              # teacher、base model、数据、训练配置、签名哈希
}
```

能力抽取模块上游接收：请求日志、teacher 输出、base 输出、成本/延迟信息。下游输出：Capability Registry 可注册、Service Router 可匹配、Deployment Controller 可放置、边缘节点可加载的能力包。

---

## 2. 第一版技术路线：V1 先跑通闭环

第一版聚焦最稳的 baseline：

```text
request logs
  -> feature extraction
  -> PCA routing space
  -> clustering
  -> extractability validation
  -> per-cluster LoRA training
  -> hard-route inference
  -> verifier
  -> fallback to teacher
  -> capability package export
```

V1 暂时只实现 hard route + per-cluster LoRA。shared basis / residual 放到 V2/V3，因为它依赖多个 adapter 训练完成后才能做组合稳定性研究。

---

## 3. 模块设计

### 3.1 数据采集与日志 schema

输入训练日志：

\[
\mathcal L=\{x_i,F_T(x_i),F_0(x_i),h_i,\tau_i,cost_i\}_{i=1}^n
\]

实现文件：

```text
src/idn/data/schema.py
src/idn/data/build_logs.py
src/idn/data/dataset.py
```

推荐 JSONL 格式：

```json
{
  "request_id": "uuid",
  "prompt": "user request text",
  "teacher_output": "cloud model answer",
  "base_output": "edge base answer",
  "teacher_trace": {
    "schema": "qa|summary|code|extract",
    "tool_calls": [],
    "evidence_type": "none|citation|span|unit_test"
  },
  "cost": {
    "teacher_latency_ms": 0,
    "teacher_tokens_in": 0,
    "teacher_tokens_out": 0
  },
  "meta": {
    "domain": "optional",
    "region": "optional",
    "privacy_label": "public|tenant|private"
  }
}
```

本地可先用公开指令数据或项目提供日志构造 demo；teacher 输出可由服务器上的大模型或 API 离线生成。

---

### 3.2 Query-only 表示提取

目标：对每个请求只基于输入 \(x\) 得到表示：

\[
h_i=\psi(x_i)
\]

实现文件：

```text
src/idn/features/extractor.py
src/idn/features/cache.py
```

第一版策略：
- 使用 edge base model 的 tokenizer + hidden states。
- 默认取最后一层 prompt mean pooling。
- 可选取最后 token hidden state。
- 将向量保存成 `artifacts/features/*.npy`，避免重复前向。

CLI：

```bash
python -m idn.cli extract-features \
  --input data/logs/train.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --output artifacts/features/train_features.npz \
  --pooling mean
```

输出：

```text
features.npz:
  H: [n, d]
  request_ids: [n]
  prompts: optional index
```

---

### 3.3 PCA routing space

目标：把高维请求表示压到可路由空间：

\[
z_i=P_m^\top(h_i-\bar h)
\]

实现文件：

```text
src/idn/routing/pca_space.py
```

第一版实现：
- `sklearn.decomposition.PCA`
- 默认 `m=32` 或按解释方差 90% 截断
- 保存 `mean`, `components`, `explained_variance`

CLI：

```bash
python -m idn.cli fit-routing-space \
  --features artifacts/features/train_features.npz \
  --n-components 32 \
  --output artifacts/router/pca.pkl
```

---

### 3.4 聚类发现候选能力区域

目标：在 z-space 中发现候选区域：

\[
D_k=\{x_i:k_i=k\}
\]

实现文件：

```text
src/idn/routing/cluster.py
src/idn/routing/boundary.py
```

第一版聚类算法：
- 默认 `KMeans`，便于控制 K 和做 hard route。
- 备选 `HDBSCAN`，用于自动发现不规则簇和噪声。

每个 cluster 保存：

\[
\mu_k=\frac1{|D_k|}\sum z_i
\]

\[
\Sigma_k=\frac1{|D_k|}\sum(z_i-\mu_k)(z_i-\mu_k)^\top
\]

路由距离：

\[
d_k(x)=(z(x)-\mu_k)^\top\Sigma_k^{-1}(z(x)-\mu_k)
\]

CLI：

```bash
python -m idn.cli discover-regions \
  --features artifacts/features/train_features.npz \
  --pca artifacts/router/pca.pkl \
  --method kmeans \
  --num-clusters 8 \
  --output artifacts/router/regions.pkl
```

---

### 3.5 可提取性验证

聚类簇需要通过三个门槛才升级为 capability region。

实现文件：

```text
src/idn/extraction/extractability.py
src/idn/eval/metrics.py
```

验证指标：

1. 频率：

\[
p_k=P(x\in D_k)\ge p_{min}
\]

2. teacher 行为稳定：

\[
H(schema|D_k), H(trace|D_k) \le threshold
\]

3. base 到 teacher 的差距可学习：

先训练一个小 probe LoRA 或少量 step 的 adapter，要求：

\[
R_{D_k}(F_{\theta_0+\Delta_k^{probe}})<R_{D_k}(F_0)
\]

第一版风险函数按任务分层：
- 通用问答/摘要：BERTScore / Rouge-L / embedding similarity / LLM judge。
- 结构化抽取：schema valid + field F1。
- 代码：unit test pass rate。
- RAG：answer similarity + citation hit。

输出：

```json
{
  "region_id": 0,
  "size": 1234,
  "frequency": 0.18,
  "schema_entropy": 0.21,
  "base_risk": 0.42,
  "probe_risk": 0.29,
  "extractable": true
}
```

CLI：

```bash
python -m idn.cli validate-regions \
  --logs data/logs/train.jsonl \
  --regions artifacts/router/regions.pkl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher-field teacher_output \
  --output artifacts/regions/validated_regions.json
```

---

### 3.6 每个区域训练 specialist LoRA

目标：对每个通过验证的区域训练局部修正：

\[
\Delta_k=\arg\min_{\Delta\in\mathcal A}\sum_{x\in D_k}\ell(F_{\theta_0+\Delta}(x),F_T(x)) + \lambda\Omega(\Delta)
\]

实现文件：

```text
src/idn/training/lora_train.py
src/idn/training/config.py
src/idn/training/data_collator.py
```

训练库：
- `transformers`
- `peft`
- `trl` 的 `SFTTrainer` 或原生 `Trainer`
- `bitsandbytes` 可选，用于 QLoRA

训练数据：
- input：prompt
- target：teacher_output
- 划分：train / val / boundary / hard cases

服务器 A100 上执行：

```bash
python -m idn.cli train-region-lora \
  --logs data/logs/train.jsonl \
  --region-id 0 \
  --regions artifacts/router/regions.pkl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --output artifacts/capabilities/region_0/adapter \
  --epochs 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 4 \
  --grad-accum 8
```

产物：

```text
artifacts/capabilities/region_0/adapter/
  adapter_config.json
  adapter_model.safetensors
```

---

### 3.7 能力包导出与 descriptor

实现文件：

```text
src/idn/package/descriptor.py
src/idn/package/exporter.py
src/idn/package/registry.py
```

能力包目录：

```text
artifacts/capabilities/capability_region_0_v1/
  descriptor.json
  router.pkl
  adapter/
  verifier.json
  fallback_policy.json
  metrics.json
  lineage.json
```

`descriptor.json` 示例：

```json
{
  "name": "local_capability_region_0",
  "task": "clustered_instruction_following",
  "quality": {
    "risk_gap_to_teacher": 0.05,
    "coverage": 0.18,
    "eval_set": "val_v1"
  },
  "latency": {
    "p50_ms": 80,
    "p95_ms": 160
  },
  "security": {
    "privacy_label": "public",
    "verifier": ["confidence", "schema"]
  },
  "resource": {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter_type": "lora",
    "vram_gb": 6,
    "disk_mb": 80
  },
  "lineage": {
    "teacher_model": "teacher-name",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "training_logs_hash": "sha256:...",
    "created_at": "..."
  }
}
```

---

### 3.8 在线推理与回退

实现文件：

```text
src/idn/serving/router.py
src/idn/serving/adapter_loader.py
src/idn/serving/verifier.py
src/idn/serving/fallback.py
src/idn/serving/server.py
```

在线流程：

```text
x
 -> h(x)
 -> z(x)
 -> Mahalanobis distance to regions
 -> softmax routing score
 -> hard select k*
 -> confidence gate
 -> load base + adapter k*
 -> local generation
 -> verifier
 -> return local answer or fallback teacher
```

接受条件：

\[
Accept(x)=1\{\max_k r_k(x)\ge\tau, H(r(x))\le\tau_H, V(x,y)=pass\}
\]

第一版 hard route：

\[
k^*=\arg\max_k r_k(x),\quad \theta(x)=\theta_0+\Delta_{k^*}
\]

API：

```bash
python -m idn.cli serve \
  --capability-dir artifacts/capabilities \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher-endpoint http://cloud-teacher/v1/chat/completions \
  --host 0.0.0.0 \
  --port 8000
```

请求：

```bash
curl localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...","quality":"normal","privacy":"public"}'
```

响应包含 execution receipt：

```json
{
  "answer": "...",
  "served_by": "local|teacher",
  "capability_id": "local_capability_region_0:v1",
  "routing_score": 0.93,
  "verifier_pass": true,
  "latency_ms": 74
}
```

---

## 4. V2/V3 扩展

### V2：Top-T adapter mixing

实现：

\[
\Delta(x)=\sum_{k\in TopT}r_k(x)\Delta_k
\]

需要新增：

```text
src/idn/composition/adapter_mixing.py
src/idn/eval/conflict.py
```

指标：
- boundary sample gain
- cosine conflict
- sign conflict
- hard route vs Top-2/Top-3

### V3：shared basis + residual

实现：

\[
\Delta_k\approx\sum_m c_{k,m}B_m+R_k
\]

新增：

```text
src/idn/composition/shared_basis.py
src/idn/composition/residual.py
src/idn/composition/compiler.py
```

推理时：

\[
c_m(x)=\sum_{k\in TopT}r_k(x)c_{k,m}
\]

\[
\Delta(x)=\sum_m\tilde c_m(x)B_m+R(x)
\]

指标：
- sparse coefficient count
- compile time
- memory footprint
- latency
- selective risk
- fallback ratio

---

## 5. 推荐代码结构

```text
/home/sig/files/IDN/
  pyproject.toml
  README.md
  info.md
  idn-paper.tex
  局部能力提取方案_PPT大纲.md
  capability_extraction_implementation_plan.md
  configs/
    base.yaml
    train_lora.yaml
    serve.yaml
  data/
    logs/
      train.jsonl
      val.jsonl
  src/
    idn/
      __init__.py
      cli.py
      data/
        schema.py
        build_logs.py
        dataset.py
      features/
        extractor.py
        cache.py
      routing/
        pca_space.py
        cluster.py
        boundary.py
      extraction/
        extractability.py
      training/
        lora_train.py
        config.py
        data_collator.py
      package/
        descriptor.py
        exporter.py
        registry.py
      serving/
        router.py
        adapter_loader.py
        verifier.py
        fallback.py
        server.py
      composition/
        adapter_mixing.py
        shared_basis.py
        residual.py
        compiler.py
      eval/
        metrics.py
        conflict.py
        report.py
  tests/
    test_schema.py
    test_pca_space.py
    test_cluster.py
    test_router.py
    test_package_export.py
```

---

## 6. 本地开发与 A100 服务器测试流程

### 本地

本机负责写代码、单元测试、小样本 CPU smoke test：

```bash
cd /home/sig/files/IDN
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m idn.cli fit-routing-space --help
python -m idn.cli discover-regions --help
```

### 上传服务器

服务器信息来自 `info.md`：

```bash
ssh wanghl@172.22.11.235
```

推荐上传方式：

```bash
rsync -av --exclude .git --exclude .venv --exclude artifacts \
  /home/sig/files/IDN/ wanghl@172.22.11.235:~/IDN/
```

### A100 上测试

```bash
cd ~/IDN
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu]'
nvidia-smi
python -m idn.cli extract-features --input data/logs/train.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct --output artifacts/features/train_features.npz
python -m idn.cli fit-routing-space --features artifacts/features/train_features.npz --output artifacts/router/pca.pkl
python -m idn.cli discover-regions --features artifacts/features/train_features.npz --pca artifacts/router/pca.pkl --num-clusters 8 --output artifacts/router/regions.pkl
python -m idn.cli validate-regions --logs data/logs/train.jsonl --regions artifacts/router/regions.pkl --output artifacts/regions/validated_regions.json
python -m idn.cli train-region-lora --logs data/logs/train.jsonl --region-id 0 --regions artifacts/router/regions.pkl --base-model Qwen/Qwen2.5-1.5B-Instruct --output artifacts/capabilities/region_0/adapter
python -m idn.cli export-capability --region-id 0 --output artifacts/capabilities/capability_region_0_v1
python -m idn.cli eval-capability --capability artifacts/capabilities/capability_region_0_v1 --logs data/logs/val.jsonl
```

---

## 7. 验收标准

V1 达标条件：

1. 能从 JSONL 日志抽取 query-only hidden representation。
2. 能训练 PCA routing space 并保存 router。
3. 能聚类出候选能力区域并计算边界距离。
4. 能用频率、teacher trace entropy、probe improvement 判断可提取性。
5. 能为至少 1 个区域训练 LoRA adapter。
6. 能导出完整 CapabilityPackage。
7. 能在在线推理中完成 hard route、加载 adapter、验证、回退。
8. 评估报告包含：
   - base risk
   - local adapter risk
   - teacher risk
   - risk gap
   - coverage
   - fallback ratio
   - p50/p95 latency
   - VRAM / disk footprint

---

## 8. 实施顺序

### Phase 0：工程骨架

- 创建 Python package、CLI、配置系统、测试框架。
- 定义日志 schema、descriptor schema、capability package schema。

### Phase 1：区域发现

- 实现 feature extraction。
- 实现 PCA routing space。
- 实现 KMeans/HDBSCAN 聚类。
- 实现 Mahalanobis 边界与 routing score。

### Phase 2：可提取性验证

- 实现 cluster 统计。
- 实现 teacher trace entropy。
- 实现 base risk / teacher similarity metrics。
- 实现 probe LoRA 小步训练或先用 full LoRA 训练的前 N step 近似。

### Phase 3：局部 LoRA 训练

- 实现按 region 切数据。
- 实现 LoRA SFT。
- 实现训练日志和 adapter 保存。

### Phase 4：能力包导出

- 生成 descriptor、lineage、metrics、fallback policy。
- 注册到本地 registry。

### Phase 5：在线服务

- 实现 FastAPI server。
- 实现 hard route。
- 实现 adapter loading/cache。
- 实现 verifier 和 fallback。
- 输出 execution receipt。

### Phase 6：A100 验证

- 上传代码。
- 跑小数据 smoke test。
- 跑一个完整 region LoRA。
- 产出第一份 evaluation report。

---

## 9. 第一轮实现建议

优先实现 V1，因为它能最快证明 IDN capability extraction 的核心主张：

\[
F_{\theta_0+\Delta_k}(x) \approx F_T(x),\quad x\in D_k
\]

工程取舍：
- base model 先选 1.5B/3B 级别，A100 上训练快。
- teacher 先用现成大模型输出字段，避免一开始做 teacher serving。
- 聚类先用 KMeans，便于可控实验。
- 推理先 hard route，避免 adapter conflict。
- verifier 先实现 schema/confidence/embedding similarity，任务型 verifier 后续补。
- shared basis 进入第二阶段，基于多个 LoRA adapter 的权重再做。
