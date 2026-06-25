# CoLA 普通 LLM 数据集能力下放评测

## 数据集

- 数据集：CoLA grammatical acceptability
- 任务：英文句子语法可接受性二分类
- 训练集：`data/standard_llm/cola_train.jsonl`，8551 条
- 评测集：`data/standard_llm/cola_eval.jsonl`，527 条
- 合并集：`data/standard_llm/cola_all.jsonl`，9078 条
- 数据格式：普通 instruction/input/output JSONL，可用于 SFT 或离线能力评测
- 原始数据：`data/cola/in_domain_train.tsv` 和 `data/cola/in_domain_dev.tsv`

## 样本格式

```json
{
  "id": "cola-in_domain_dev-00000",
  "task": "grammatical_acceptability",
  "instruction": "Classify whether the English sentence is grammatically acceptable. Answer with acceptable or unacceptable.",
  "input": "the sailors rode the breeze clear of the rocks .",
  "output": "acceptable",
  "label": 1
}
```
```json
{
  "id": "cola-in_domain_dev-00001",
  "task": "grammatical_acceptability",
  "instruction": "Classify whether the English sentence is grammatically acceptable. Answer with acceptable or unacceptable.",
  "input": "the weights made the rope stretch over the pulley .",
  "output": "acceptable",
  "label": 1
}
```
```json
{
  "id": "cola-in_domain_dev-00002",
  "task": "grammatical_acceptability",
  "instruction": "Classify whether the English sentence is grammatically acceptable. Answer with acceptable or unacceptable.",
  "input": "the mechanical doll wriggled itself loose .",
  "output": "acceptable",
  "label": 1
}
```

## 评测设置

- 模型：Qwen2.5-1.5B-Instruct vs Qwen2.5-7B-Instruct
- 评测样本：完整 CoLA dev split，527 条
- Prompt：3-shot per label
- Scoring：A/B 单步 next-token logit score
- A：acceptable，B：unacceptable

## 准确率结果

| 模型 | 样本数 | 正确数 | Accuracy | TP | TN | FP | FN | P50延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 527 | 392 | 74.38% | 350 | 42 | 120 | 15 | 28.6 ms |
| Qwen2.5-7B-Instruct | 527 | 355 | 67.36% | 205 | 150 | 12 | 160 | 27.2 ms |

## 能力下放结论

- 在 CoLA 语法可接受性二分类上，Qwen2.5-1.5B 准确率 74.38%，高于 Qwen2.5-7B 的 67.36%。
- 该任务可作为“语法可接受性/简单语言判断”能力的小模型候选任务。
- 7B 的 false negative 偏高，说明它在当前提示和 A/B logit 评分下更保守，后续可单独调 prompt 或校准阈值。
- 能力分发参考：这类固定标签分类任务适合抽取到小模型或边缘模型；复杂开放生成任务应继续补充摘要、数学、QA、代码等数据集后再判断。

## 复现命令

```bash
/home/sig/anaconda3/bin/python src/idn/build_standard_llm_dataset.py

/home/sig/anaconda3/bin/python src/idn/eval_qwen_choice_accuracy.py \
  --dataset-path data/cola/in_domain_dev.tsv \
  --train-path data/cola/in_domain_train.tsv \
  --num-samples 0 \
  --few-shot-per-label 3 \
  --small-model /home/sig/models/Qwen2.5-1.5B-Instruct \
  --big-model /home/sig/models/Qwen2.5-7B-Instruct \
  --output-path artifacts/eval/cola_qwen25_7b_vs_15b_choice_full527.jsonl \
  --summary-path artifacts/eval/cola_qwen25_7b_vs_15b_choice_full527_summary.json
```
