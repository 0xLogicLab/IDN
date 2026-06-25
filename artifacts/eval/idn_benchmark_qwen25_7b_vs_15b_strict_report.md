# IDN Benchmark: Qwen2.5 1.5B vs 7B 分发能力参考

## 数据集

- 数据集：`data/benchmarks/idn_benchmark_v1_eval.jsonl`
- 样本数：48
- 任务覆盖：capability descriptor extraction、route decision、fallback decision、resource fit、cache decision、trust policy
- 评测方式：本地模型贪心解码，要求输出 JSON；统计 JSON 有效率、关键字段准确率、exact match、延迟
- 环境：RTX 3090，本地 `/home/sig/anaconda3/bin/python`，Qwen2.5-Instruct 本地权重

## 总体结果

| 模型 | JSON有效率 | 关键字段准确率 | Exact Match | 平均延迟 | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 100.0% | 35.1% | 12.5% | 1869 ms | 1665 ms | 3347 ms |
| Qwen2.5-7B-Instruct | 60.4% | 19.1% | 4.2% | 1847 ms | 1703 ms | 3906 ms |

## 分任务结果

### cache_decision
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 6 | 100.0% | 8.3% | 0.0% |
| Qwen2.5-7B-Instruct | 6 | 83.3% | 8.3% | 0.0% |

### capability_descriptor_extraction
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 6 | 100.0% | 52.1% | 0.0% |
| Qwen2.5-7B-Instruct | 6 | 33.3% | 8.3% | 0.0% |

### fallback_decision
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 8 | 100.0% | 25.0% | 0.0% |
| Qwen2.5-7B-Instruct | 8 | 87.5% | 50.0% | 0.0% |

### resource_fit
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 8 | 100.0% | 56.2% | 37.5% |
| Qwen2.5-7B-Instruct | 8 | 50.0% | 6.2% | 0.0% |

### route_decision
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 11 | 100.0% | 9.1% | 0.0% |
| Qwen2.5-7B-Instruct | 11 | 90.9% | 27.3% | 9.1% |

### trust_policy
| 模型 | 样本数 | JSON有效率 | 关键字段准确率 | Exact Match |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 9 | 100.0% | 33.3% | 33.3% |
| Qwen2.5-7B-Instruct | 9 | 11.1% | 11.1% | 11.1% |

## 结论和分发建议

- Qwen2.5-1.5B 在当前 IDN JSON 决策 benchmark 上更稳：JSON 有效率 100%，关键字段准确率 35.1%，整体 exact match 12.5%。
- Qwen2.5-7B 当前提示下 JSON 有效率 60.4%，关键字段准确率 19.1%，主要问题是输出解释性文本、非法 JSON、字段类型漂移。
- 当前结果可作为“模型分发能力参考”的第一版离线基准：1.5B 可承担严格结构化输出和简单策略判断；7B 需要更强约束解码、few-shot 或 schema 修复层后再进入自动分发链路。
- 任务级建议：resource_fit 可优先在小模型侧试运行；fallback_decision 可由 7B 辅助判断；route_decision、cache_decision 需要规则/检索约束；trust_policy 建议走显式策略引擎或 teacher 校验。

## 产物

- 评测脚本：`src/idn/eval_idn_benchmark.py`
- 逐样本输出：`artifacts/eval/idn_benchmark_qwen25_7b_vs_15b_strict.jsonl`
- 汇总 JSON：`artifacts/eval/idn_benchmark_qwen25_7b_vs_15b_strict_summary.json`
- 本报告：`artifacts/eval/idn_benchmark_qwen25_7b_vs_15b_strict_report.md`

## 复现实验命令

```bash
/home/sig/anaconda3/bin/python src/idn/eval_idn_benchmark.py \
  --dataset data/benchmarks/idn_benchmark_v1_eval.jsonl \
  --small-model /home/sig/models/Qwen2.5-1.5B-Instruct \
  --big-model /home/sig/models/Qwen2.5-7B-Instruct \
  --output-path artifacts/eval/idn_benchmark_qwen25_7b_vs_15b_strict.jsonl \
  --summary-path artifacts/eval/idn_benchmark_qwen25_7b_vs_15b_strict_summary.json
```
