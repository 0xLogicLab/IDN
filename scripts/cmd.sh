# evaluate a dataset
PYTHONPATH=src .venv_qwen35/bin/python -m idn.cli eval --config configs/capabilities/gsm8k.yaml --model Qwen2.5-1.5B-Instruct --verbose --limit 100

# generate visualization
PYTHONPATH=src .venv_qwen35/bin/python -m idn.cli visualize-embeddings --projection tsne

# print dataset info
PYTHONPATH=src .venv_qwen35/bin/python -m idn.cli list_datasets

# visualize activations
PYTHONPATH=src .venv_qwen35/bin/python -m idn.cli --machine 3090 visualize-activations --model Qwen2.5-1.5B-Instruct --capabilities sst2,cola