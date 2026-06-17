# Hardware FCL

This directory contains the Flower-based hardware path.

Network assumptions:

- Server Jetson Orin Nano: `192.168.0.141`
- Raspberry Pi clients: `192.168.0.112` to `192.168.0.121`
- Flower server port: `8080`

Generate materialized client/task datasets:

```bash
python hardware_fcl/dataset_generator.py \
  --hf_dataset zh-plus/tiny-imagenet \
  --hf_split train \
  --output_dir ./hardware_fcl/data/generated \
  --num_clients 10 \
  --drift_mode data_and_concept \
  --report_mode oracle \
  --overwrite
```

Start Ollama on the Jetson and pull the local model:

```bash
ollama serve
ollama pull qwen3.5:4b
```

Start the server on the Jetson:

```bash
python hardware_fcl/server.py \
  --server_address 0.0.0.0:8080 \
  --advertised_server_address 192.168.0.141:8080 \
  --num_clients 10 \
  --clients_per_round 5 \
  --llm_backend ollama
```

Start each Raspberry Pi client with its mapped `cid`. Example for `192.168.0.112`:

```bash
python hardware_fcl/client.py \
  --cid 0 \
  --server_address 192.168.0.141:8080 \
  --data_root ./hardware_fcl/data/generated \
  --method slm
```

Use `hardware_fcl/configs/hardware_config.yaml` for the default IP-to-client mapping.

Logs are written to:

```text
hardware_fcl/logs/{timestamp}/
```

Expected files:

- `round_metrics.csv`
- `client_metrics.csv`
- `llm_decisions.csv`
- `latency_metrics.csv`
- `config.json`
