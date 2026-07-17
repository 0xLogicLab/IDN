# IDN Capability Extraction

This repository implements the experiment pipeline described in
`CAPABILITY_EXTRACTION_PLAN.md`.

Phase 0 provides the package skeleton, YAML config loading and machine overrides,
the capability registry, and a CLI entry point.

```bash
python -m idn.cli --help
python -m idn.cli list-capabilities
python -m idn.cli --machine 3090 train-lora --config configs/capabilities/cola.yaml --rank 16
python -m idn.cli --machine h100 eval --config configs/capabilities/sst2.yaml --model Qwen2.5-7B-Instruct
python -m idn.cli --machine h100 visualize-activations --model Qwen2.5-7B-Instruct --capabilities gsm8k,mbpp,cnn_dailymail
python -m idn.cli --machine 3090 fit-routing --model Qwen2.5-1.5B-Instruct
python -m idn.cli --machine 3090 route --input "Write Python code to sort a list"
scripts/build_route_eval_dataset.py --machine 3090
scripts/eval_oracle_base_lora.sh
python scripts/eval_routed_lora_mix.py --machine 3090
python scripts/build_shared_basis.py --components 4
python scripts/eval_basis_lora_mix.py --machine h100
python scripts/train_gsm8k_improved.py --machine 3090
scripts/train_eval_all_lora.sh
python -m pytest
```

Machine configs are selected with `--machine 3090` or `--machine h100`. Pass model
and dataset directories as paths relative to the selected machine config's
`model_dir` and `data_dir`; the pipeline loads from disk, not from the Hugging
Face Hub.

`visualize-activations` samples request text from the datasets configured under
`configs/capabilities`, runs those samples through a local causal LM, averages
mean absolute hidden-state activations by layer and hidden dimension, and writes
a self-contained HTML dashboard plus JSON and top-neuron CSV files under
`artifacts/analysis`.

`train-lora` performs gold-supervised SFT on prompt-masked targets using the
frozen Qwen2.5-1.5B base. Defaults come from `configs/train/lora_default.yaml`;
rank-16 adapters are written to `artifacts/adapters/{capability}/r16` with
`train_meta.json` lineage and a step-level `training_log.json`.

`scripts/train_eval_all_lora.sh` lists the train and eval commands for all six
capabilities.

`scripts/train_gsm8k_improved.py` is the experimental GSM8K training path. It uses
cleaned rationale targets and base-model KL preservation; candidates are evaluated
at full adapter strength before they can replace the standard adapter.

`fit-routing` mean-pools the Qwen2.5-1.5B base model's final hidden states for every
configured capability, fits PCA and shrinkage-regularized Mahalanobis descriptors,
and writes `artifacts/routing/{space.npz,report.json}`. The report records held-out
accuracy, coverage, a confusion matrix, and OOD rejection. `route` uses the same base
model and returns capability weights, distances, entropy, confidence, and OOD status.

`scripts/build_route_eval_dataset.py` creates a deterministic 1,000-example mixed
end-to-end holdout with prompt inputs, references, gold targets, and a source/overlap
manifest under `artifacts/eval`.

`scripts/eval_oracle_base_lora.sh` evaluates that mixed dataset with the 1.5B base
and with the matching capability LoRA selected from the ground-truth label. It is an
oracle baseline and does not invoke ROUTE. Each run loads the base once; LoRA mode
loads all six adapters and switches with `set_adapter`. The aggregate
`macro_capability_score` is the unweighted mean of the six normalized primary
metrics, giving every capability equal weight.

`scripts/eval_routed_lora_mix.py` evaluates the same stream with semantic ROUTE.
For every request it activates all six LoRAs and scales their contributions by the
six routing probabilities, producing the exact weighted delta without materializing
a new rank-96 adapter. Each prediction records all routing weights and diagnostics.

`scripts/build_shared_basis.py` decomposes the six complete LoRA delta operators into
four global shared basis adapters and one exact residual per capability. The global
Gram eigendecomposition is computed directly from the low-rank factors, without
materializing dense Qwen deltas. `scripts/eval_basis_lora_mix.py` combines routing
weights in the four-dimensional coordinate space and adds only the top-routed
capability residual, scaled by its route probability. It records the route weights,
shared coordinates, selected residual, predictions, and native capability metrics.

The current Phase 1 measurements and gate decision are recorded in
`artifacts/reports/phase1_specialists.json`.
