from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
except Exception as exc:  # pragma: no cover
    fl = None
    FLOWER_IMPORT_ERROR = exc

from hardware_fcl.client import GeneratedDataAccess, evaluate
from hardware_fcl.llm_decision import heuristic_select_clients, select_clients_with_llm
from hardware_fcl.logger import CSVLogger
from hardware_fcl.model import create_model
from hardware_fcl.prompt import build_client_selection_prompt
from hardware_fcl.utils import get_parameters, load_pt_loader, seen_classes_for_task, set_parameters, set_seed


DEFAULT_CLIENT_IPS = [f"192.168.0.{i}" for i in range(112, 122)]


def round_to_task(server_round: int, rounds_per_task: int, num_global_tasks: int) -> int:
    return min(num_global_tasks, max(1, math.ceil(max(1, server_round) / rounds_per_task)))


class LLMSelectionFedAvg(fl.server.strategy.FedAvg if fl is not None else object):
    def __init__(self, *, args, logger: CSVLogger, **kwargs) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.logger = logger
        self.round_starts: dict[int, float] = {}
        self.round_times: dict[int, float] = {}
        self.aggregation_times: dict[int, float] = {}
        self.llm_times: dict[int, float] = {}
        self.selected_counts: dict[int, int] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_starts[server_round] = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        global_task = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        reports = []
        cid_to_result = {}
        for client_proxy, fit_res in results:
            metrics = dict(fit_res.metrics)
            cid = int(metrics.get("cid", getattr(client_proxy, "cid", -1)))
            cid_to_result[cid] = (client_proxy, fit_res)
            try:
                report = json.loads(str(metrics.get("client_report_json", "{}")))
            except json.JSONDecodeError:
                report = {"cid": cid, "num_samples": int(fit_res.num_examples)}
            report["cid"] = cid
            reports.append(report)
            self.logger.log_row(
                "client_metrics.csv",
                {
                    "round": server_round,
                    "global_task": global_task,
                    "cid": cid,
                    "num_samples": int(fit_res.num_examples),
                    "fit_time_sec": float(metrics.get("fit_time_sec", 0.0)),
                    "train_loss": float(metrics.get("train_loss", 0.0)),
                    "buffer_size": int(metrics.get("buffer_size", 0)),
                    "client_report": report,
                },
            )
        seen_classes = seen_classes_for_task(global_task, self.args.num_classes_per_task, self.args.drift_mode)
        prompt = build_client_selection_prompt(
            round_id=server_round,
            global_task=global_task,
            clients_per_round=self.args.clients_per_round,
            seen_classes=seen_classes,
            client_reports=reports,
        )
        if self.args.llm_backend == "heuristic":
            t0 = time.time()
            selected = heuristic_select_clients(reports, self.args.clients_per_round)
            decision = {
                "selected_clients": selected,
                "llm_inference_time_sec": time.time() - t0,
                "used_fallback": True,
                "raw_response": "",
                "reason": "heuristic backend",
            }
        else:
            decision = select_clients_with_llm(
                prompt=prompt,
                client_reports=reports,
                clients_per_round=self.args.clients_per_round,
                model=self.args.ollama_model,
                host=self.args.ollama_host,
            )
        selected = [cid for cid in decision["selected_clients"] if cid in cid_to_result]
        if not selected:
            selected = [int(r["cid"]) for r in reports[: self.args.clients_per_round]]
        selected_results = [cid_to_result[cid] for cid in selected]
        self.llm_times[server_round] = float(decision.get("llm_inference_time_sec", 0.0))
        self.selected_counts[server_round] = len(selected_results)
        self.logger.log_row(
            "llm_decisions.csv",
            {
                "round": server_round,
                "global_task": global_task,
                "candidate_clients": [int(r["cid"]) for r in reports],
                "selected_clients": selected,
                "llm_inference_time_sec": float(decision.get("llm_inference_time_sec", 0.0)),
                "used_fallback": bool(decision.get("used_fallback", False)),
                "raw_response": decision.get("raw_response", ""),
                "reason": decision.get("reason", decision.get("error", "")),
            },
        )
        wait_time = time.time() - self.round_starts.get(server_round, time.time())
        agg_start = time.time()
        aggregated = super().aggregate_fit(server_round, selected_results, failures)
        agg_time = time.time() - agg_start
        round_time = time.time() - self.round_starts.get(server_round, time.time())
        self.aggregation_times[server_round] = agg_time
        self.round_times[server_round] = round_time
        for _, fit_res in results:
            metrics = dict(fit_res.metrics)
            cid = int(metrics.get("cid", -1))
            self.logger.log_row(
                "latency_metrics.csv",
                {
                    "round": server_round,
                    "global_task": global_task,
                    "cid": cid,
                    "client_training_time_sec": float(metrics.get("fit_time_sec", 0.0)),
                    "communication_time_sec": 0.0,
                    "server_wait_time_sec": float(wait_time),
                    "llm_inference_time_sec": float(decision.get("llm_inference_time_sec", 0.0)),
                    "aggregation_time_sec": float(agg_time),
                    "round_time_sec": float(round_time),
                },
            )
        return aggregated


def make_fit_config_fn(args):
    def fit_config(server_round: int):
        return {
            "global_task": round_to_task(server_round, args.rounds_per_task, args.num_global_tasks),
            "local_epochs": args.local_epochs,
            "proximal_mu": args.proximal_mu,
            "replay_alpha": args.replay_alpha,
            "lr": args.lr,
            "report_mode": args.report_mode,
        }

    return fit_config


def make_evaluate_fn(args, logger: CSVLogger, strategy_holder: dict):
    def evaluate_fn(server_round, parameters, config):
        if server_round <= 0:
            return None
        global_task = round_to_task(server_round, args.rounds_per_task, args.num_global_tasks)
        test_path = Path(args.data_root) / "global_test" / f"task_{global_task:03d}_seen_classes.pt"
        if not test_path.exists():
            return None
        model = create_model(args.num_classes).to(args.device)
        set_parameters(model, parameters)
        metrics = evaluate(model, load_pt_loader(test_path, args.batch_size, False), args.device)
        strategy = strategy_holder["strategy"]
        logger.log_row(
            "round_metrics.csv",
            {
                "round": server_round,
                "global_task": global_task,
                "accuracy_seen_classes": float(metrics["accuracy"]),
                "loss_seen_classes": float(metrics["loss"]),
                "round_time_sec": float(strategy.round_times.get(server_round, 0.0)),
                "num_selected_clients": int(strategy.selected_counts.get(server_round, 0)),
            },
        )
        return float(metrics["loss"]), {"accuracy_seen_classes": float(metrics["accuracy"])}

    return evaluate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Flower server for hardware FCL on Jetson Orin Nano.")
    parser.add_argument("--server_address", default="0.0.0.0:8080")
    parser.add_argument("--advertised_server_address", default="192.168.0.141:8080")
    parser.add_argument("--client_ips", nargs="*", default=DEFAULT_CLIENT_IPS)
    parser.add_argument("--data_root", default="./hardware_fcl/data/generated")
    parser.add_argument("--output_dir", default="./hardware_fcl/logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--clients_per_round", type=int, default=5)
    parser.add_argument("--num_global_tasks", type=int, default=5)
    parser.add_argument("--rounds_per_task", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--num_classes_per_task", type=int, default=20)
    parser.add_argument("--drift_mode", choices=["data_only", "concept_only", "data_and_concept"], default="data_and_concept")
    parser.add_argument("--proximal_mu", type=float, default=0.01)
    parser.add_argument("--replay_alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--report_mode", choices=["oracle", "estimated"], default="oracle")
    parser.add_argument("--llm_backend", choices=["ollama", "heuristic"], default="ollama")
    parser.add_argument("--ollama_model", default="qwen3.5:4b")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    if fl is None:
        raise RuntimeError(f"Flower is required for hardware server: {FLOWER_IMPORT_ERROR}")
    args = parse_args()
    set_seed(args.seed)
    logger = CSVLogger.timestamped(args.output_dir, config=vars(args))
    initial_model = create_model(num_classes=args.num_classes)
    strategy_holder = {}
    evaluate_fn = make_evaluate_fn(args, logger, strategy_holder)
    strategy = LLMSelectionFedAvg(
        args=args,
        logger=logger,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_fit_config_fn=make_fit_config_fn(args),
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(get_parameters(initial_model)),
    )
    strategy_holder["strategy"] = strategy
    print(f"Server listening on {args.server_address}; clients connect to {args.advertised_server_address}")
    print(f"Expected client IPs: {', '.join(args.client_ips)}")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_global_tasks * args.rounds_per_task),
        strategy=strategy,
    )
    print(f"Logs saved under: {logger.output_dir}")


if __name__ == "__main__":
    main()
