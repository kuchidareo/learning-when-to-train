# Flower Baselines

This directory runs the three baseline methods with Flower simulation:

- `fedavg`: random client selection, no replay, no FedProx
- `fedprox_replay`: random client selection, client-side FedProx proximal loss, persistent replay buffers
- `gc_fedprox_replay`: FedProx + replay, plus server-side update clustering for logging

Examples:

```bash
python baselines/run_baselines.py --method fedavg --drift_mode data_and_concept
python baselines/run_baselines.py --method fedprox_replay --drift_mode data_and_concept
python baselines/run_baselines.py --method gc_fedprox_replay --drift_mode data_and_concept
```

The default dataset is pulled from Hugging Face:

```bash
python baselines/run_baselines.py --method fedavg --hf_dataset zh-plus/tiny-imagenet --hf_split train
```

Logs are written to:

```text
baselines/logs/{method}/{timestamp}/
```

Expected CSVs:

- `round_metrics.csv`
- `client_fit_metrics.csv`
- `task_eval_metrics.csv`
- `cluster_metrics.csv` for `gc_fedprox_replay`
- `config.json`
