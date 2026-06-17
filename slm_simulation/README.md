# SLM-Guided Flower Simulation

This simulation uses Flower. Every candidate client trains and reports local metadata, then the server strategy selects `clients_per_round` clients with either a Hugging Face SLM or the deterministic heuristic fallback. Only selected updates are aggregated with standard FedAvg.

Examples:

```bash
python slm_simulation/run_slm_fcl_simulation.py --report_mode oracle --llm_backend hf
python slm_simulation/run_slm_fcl_simulation.py --report_mode estimated --llm_backend hf
python slm_simulation/run_slm_fcl_simulation.py --report_mode oracle --llm_backend heuristic
```

Full example:

```bash
python slm_simulation/run_slm_fcl_simulation.py \
  --num_clients 20 \
  --clients_per_round 5 \
  --num_global_tasks 5 \
  --rounds_per_task 20 \
  --report_mode oracle \
  --llm_backend hf
```

Logs are written to:

```text
slm_simulation/logs/{timestamp}/
```

Expected CSVs:

- `round_metrics.csv`
- `client_fit_metrics.csv`
- `task_eval_metrics.csv`
- `llm_decisions.csv`
- `selected_client_diversity.csv`
- `config.json`
