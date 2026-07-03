# evaluate a dataset
PYTHONPATH=src /home/sig/anaconda3/envs/control/bin/python -m idn.cli eval --config configs/capabilities/sst2.yaml --model Qwen2.5-7B-Instruct --verbose

# generate visualization
PYTHONPATH=src /home/sig/anaconda3/envs/control/bin/python -m idn.cli visualize-embeddings --projection tsne
