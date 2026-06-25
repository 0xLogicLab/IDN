# IDN Benchmark v2: 构造数据集上的模型分发能力参考

## 数据集构造

- 构造脚本：`src/idn/build_idn_benchmark.py`
- 全量数据：`data/benchmarks/idn_benchmark_v2.jsonl`，960 条
- Train split：`data/benchmarks/idn_benchmark_v2_train.jsonl`，720 条
- Eval split：`data/benchmarks/idn_benchmark_v2_eval.jsonl`，240 条
- 生成方式：6 类 IDN 分发/能力任务 × 160 条；eval 取 240 条用于真实模型评测
- 任务覆盖：capability descriptor extraction、route decision、fallback decision、resource fit、cache decision、trust policy
- 标签字段：JSON 有效性、关键字段准确率、exact match、延迟

## 总体结果

| 模型 | Eval样本 | JSON有效率 | 关键字段准确率 | Exact Match | 平均延迟 | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 240 | 100.0% | 33.3% | 12.9% | 1862 ms | 1560 ms | 3362 ms |
| Qwen2.5-7B-Instruct | 240 | 70.0% | 20.7% | 5.0% | 2084 ms | 1956 ms | 4775 ms |

## 分任务结果

### cache_decision
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 43 | 100.0% | 2.3% | 0.0% |
| Qwen2.5-7B-Instruct | 43 | 88.4% | 2.3% | 0.0% |

### capability_descriptor_extraction
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 34 | 100.0% | 47.8% | 0.0% |
| Qwen2.5-7B-Instruct | 34 | 61.8% | 17.3% | 0.0% |

### fallback_decision
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 40 | 100.0% | 25.8% | 0.0% |
| Qwen2.5-7B-Instruct | 40 | 90.0% | 44.2% | 0.0% |

### resource_fit
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 46 | 100.0% | 30.4% | 10.9% |
| Qwen2.5-7B-Instruct | 46 | 78.3% | 14.1% | 0.0% |

### route_decision
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 35 | 100.0% | 14.3% | 0.0% |
| Qwen2.5-7B-Instruct | 35 | 88.6% | 28.6% | 17.1% |

### trust_policy
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 42 | 100.0% | 61.9% | 61.9% |
| Qwen2.5-7B-Instruct | 42 | 14.3% | 14.3% | 14.3% |

## 分发参考

- Qwen2.5-1.5B：JSON 稳定性最好，240/240 都能解析，适合做边缘侧结构化决策输出。
- Qwen2.5-7B：字段准确率在 fallback_decision、route_decision 上更高，但 JSON 稳定性只有 70%，需要 schema 修复或约束解码。
- resource_fit / trust_policy：当前小模型表现更稳，可作为默认低成本判断器，再用 teacher 做高风险复核。
- route_decision / fallback_decision：7B 能给出更接近策略的字段，但上线前要加 JSON schema validator。
- cache_decision：两个模型都弱，建议使用显式规则引擎，模型只生成解释或辅助特征。

## 复现命令

```bash
/home/sig/anaconda3/bin/python src/idn/build_idn_benchmark.py \
  --output data/benchmarks/idn_benchmark_v2.jsonl \
  --train-output data/benchmarks/idn_benchmark_v2_train.jsonl \
  --eval-output data/benchmarks/idn_benchmark_v2_eval.jsonl \
  --repeats 20 --seed 42

/home/sig/anaconda3/bin/python src/idn/eval_idn_benchmark.py \
  --dataset data/benchmarks/idn_benchmark_v2_eval.jsonl \
  --small-model /home/sig/models/Qwen2.5-1.5B-Instruct \
  --big-model /home/sig/models/Qwen2.5-7B-Instruct \
  --output-path artifacts/eval/idn_benchmark_v2_qwen25_7b_vs_15b_strict.jsonl \
  --summary-path artifacts/eval/idn_benchmark_v2_qwen25_7b_vs_15b_strict_summary.json
```
