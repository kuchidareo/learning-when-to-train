from __future__ import annotations

import argparse
import json
import math
import os
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

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None

from hardware_fcl.client import FCLClient, GeneratedDataAccess, evaluate
from hardware_fcl.dataset_generator import generate_datasets
from hardware_fcl.logger import CSVLogger
from hardware_fcl.model import create_model
from hardware_fcl.utils import ReplayBuffer, get_parameters, load_pt_loader, seen_classes_for_task, set_parameters, set_seed


def round_to_task(server_round: int, rounds_per_task: int, num_global_tasks: int) -> int:
    return min(num_global_tasks, max(1, math.ceil(max(1, server_round) / rounds_per_task)))


def ensure_generated_data(args) -> Path:
    data_root = Path(args.output_dir) / "generated_data"
    expected = data_root / "client_000" / "task_001" / "train.pt"
    if not expected.exists() or args.regenerate_data:
        generate_datasets(
            output_root=data_root,
            hf_dataset=args.hf_dataset,
            hf_split=args.hf_split,
            num_clients=args.num_clients,
            num_global_tasks=args.num_global_tasks,
            num_classes_per_task=args.num_classes_per_task,
            alpha=args.alpha,
            seed=args.seed,
            drift_mode=args.drift_mode,
            report_mode="oracle",
            overwrite=True,
        )
    return data_root


class LoggingFedAvg(fl.server.strategy.FedAvg if fl is not None else object):
    def __init__(self, *, method: str, logger: CSVLogger, data_access: GeneratedDataAccess, args, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.logger = logger
        self.data_access = data_access
        self.args = args
        self.current_global = None
        self.round_starts: dict[int, float] = {}
        self.round_times: dict[int, float] = {}
        self.num_fit_clients: dict[int, int] = {}
        self.best_task_acc: dict[int, float] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_global = parameters_to_ndarrays(parameters)
        self.round_starts[server_round] = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        global_task = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        self.num_fit_clients[server_round] = len(results)
        for client_proxy, fit_res in results:
            metrics = dict(fit_res.metrics)
            self.logger.log_row(
                "client_fit_metrics.csv",
                {
                    "method": self.method,
                    "round": server_round,
                    "global_task": global_task,
                    "cid": int(metrics.get("cid", getattr(client_proxy, "cid", -1))),
                    "num_samples": int(fit_res.num_examples),
                    "fit_time_sec": float(metrics.get("fit_time_sec", 0.0)),
                    "train_loss": float(metrics.get("train_loss", 0.0)),
                    "buffer_size": int(metrics.get("buffer_size", 0)),
                },
            )
        aggregated = super().aggregate_fit(server_round, results, failures)
        self.round_times[server_round] = time.time() - self.round_starts.get(server_round, time.time())
        return aggregated

    def log_task_evals(self, server_round: int, parameters) -> None:
        if server_round <= 0 or server_round % self.args.rounds_per_task != 0:
            return
        global_task = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        model = create_model(num_classes=self.args.num_classes).to(self.args.device)
        set_parameters(model, parameters_to_ndarrays(parameters))
        for eval_task in range(1, global_task + 1):
            path = Path(self.args.data_root_resolved) / "global_test" / f"task_{eval_task:03d}_classes.pt"
            loader = load_pt_loader(path, self.args.batch_size, shuffle=False)
            metrics = evaluate(model, loader, self.args.device)
            acc = float(metrics["accuracy"])
            self.best_task_acc[eval_task] = max(self.best_task_acc.get(eval_task, acc), acc)
            forgetting = self.best_task_acc[eval_task] - acc
            self.logger.log_row(
                "task_eval_metrics.csv",
                {
                    "method": self.method,
                    "global_task": global_task,
                    "eval_task": eval_task,
                    "seen_classes": seen_classes_for_task(eval_task, self.args.num_classes_per_task, self.args.drift_mode),
                    "accuracy": acc,
                    "loss": float(metrics["loss"]),
                    "forgetting": float(forgetting),
                },
            )


class GCFedProxReplay(LoggingFedAvg):
    def __init__(self, num_clusters=3, head_only=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        self.head_only = head_only

    def aggregate_fit(self, server_round, results, failures):
        global_task = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        if self.current_global is not None and len(results) >= self.num_clusters:
            client_vectors = []
            client_ids = []
            for client_proxy, fit_res in results:
                client_params = parameters_to_ndarrays(fit_res.parameters)
                delta = [cp - gp for cp, gp in zip(client_params, self.current_global)]
                vec = self.extract_update_vector(delta)
                client_vectors.append(vec)
                cid = int(fit_res.metrics.get("cid", getattr(client_proxy, "cid", -1)))
                client_ids.append(cid)
            Z = np.stack(client_vectors)
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
            if KMeans is not None:
                cluster_ids = KMeans(n_clusters=self.num_clusters, n_init="auto", random_state=server_round).fit_predict(Z)
            else:
                order = np.argsort(Z[:, 0])
                cluster_ids = np.zeros(len(Z), dtype=int)
                for rank, idx in enumerate(order):
                    cluster_ids[idx] = min(self.num_clusters - 1, int(rank * self.num_clusters / len(Z)))
            sim = Z @ Z.T
            cluster_sizes = [int(np.sum(cluster_ids == k)) for k in range(self.num_clusters)]
            intra = []
            for k in range(self.num_clusters):
                members = np.where(cluster_ids == k)[0]
                if len(members) > 1:
                    sub = sim[np.ix_(members, members)]
                    intra.append(float((sub.sum() - len(members)) / max(1, len(members) * (len(members) - 1))))
            self.logger.log_row(
                "cluster_metrics.csv",
                {
                    "method": self.method,
                    "round": server_round,
                    "global_task": global_task,
                    "client_ids": client_ids,
                    "cluster_ids": cluster_ids.tolist(),
                    "cluster_sizes": cluster_sizes,
                    "mean_cosine": float(np.mean(sim)),
                    "intra_cluster_similarity": float(np.mean(intra)) if intra else 0.0,
                },
            )
        return super().aggregate_fit(server_round, results, failures)

    def extract_update_vector(self, delta):
        selected = delta[-2:] if self.head_only else delta
        return np.concatenate([x.reshape(-1) for x in selected])


def make_evaluate_fn(args, logger: CSVLogger, strategy_holder: dict):
    def evaluate_fn(server_round, parameters, config):
        if server_round <= 0:
            return None
        global_task = round_to_task(server_round, args.rounds_per_task, args.num_global_tasks)
        model = create_model(num_classes=args.num_classes).to(args.device)
        set_parameters(model, parameters)
        loader = load_pt_loader(Path(args.data_root_resolved) / "global_test" / f"task_{global_task:03d}_seen_classes.pt", args.batch_size, False)
        metrics = evaluate(model, loader, args.device)
        strategy = strategy_holder["strategy"]
        logger.log_row(
            "round_metrics.csv",
            {
                "method": args.method,
                "round": server_round,
                "global_task": global_task,
                "accuracy_seen_classes": float(metrics["accuracy"]),
                "loss_seen_classes": float(metrics["loss"]),
                "round_time_sec": float(strategy.round_times.get(server_round, 0.0)),
                "num_selected_clients": int(strategy.num_fit_clients.get(server_round, args.clients_per_round)),
            },
        )
        strategy.log_task_evals(server_round, ndarrays_to_parameters(parameters))
        return float(metrics["loss"]), {"accuracy_seen_classes": float(metrics["accuracy"])}

    return evaluate_fn


def make_fit_config_fn(args):
    def fit_config(server_round: int):
        return {
            "global_task": round_to_task(server_round, args.rounds_per_task, args.num_global_tasks),
            "local_epochs": args.local_epochs,
            "proximal_mu": args.proximal_mu if args.method != "fedavg" else 0.0,
            "replay_alpha": args.replay_alpha if args.method != "fedavg" else 0.0,
            "lr": args.lr,
            "report_mode": "oracle",
        }

    return fit_config


def get_cid(context_or_cid) -> str:
    if isinstance(context_or_cid, str):
        return context_or_cid
    node_config = getattr(context_or_cid, "node_config", {}) or {}
    if "partition-id" in node_config:
        return str(node_config["partition-id"])
    if "cid" in node_config:
        return str(node_config["cid"])
    return str(context_or_cid)


def run(args) -> None:
    if fl is None:
        raise RuntimeError(f"Flower is required for baselines/run_baselines.py: {FLOWER_IMPORT_ERROR}")
    set_seed(args.seed)
    args.device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ensure_generated_data(args)
    args.data_root_resolved = str(data_root)
    log_root = Path(args.output_dir) / "logs" / args.method
    logger = CSVLogger.timestamped(log_root, config=vars(args))
    data_access = GeneratedDataAccess(data_root, batch_size=args.batch_size)
    replay_buffers = {str(cid): ReplayBuffer(max_size=args.replay_size) for cid in range(args.num_clients)}

    def client_fn(context):
        cid = get_cid(context)
        replay = replay_buffers[cid] if args.method != "fedavg" else None
        return FCLClient(
            cid=cid,
            data_access=data_access,
            method=args.method,
            num_classes=args.num_classes,
            replay_buffer=replay,
            device=args.device,
            report_mode="oracle",
        ).to_client()

    initial_model = create_model(num_classes=args.num_classes)
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))
    strategy_holder = {}
    common_kwargs = dict(
        fraction_fit=args.clients_per_round / args.num_clients,
        fraction_evaluate=0.0,
        min_fit_clients=args.clients_per_round,
        min_available_clients=args.num_clients,
        on_fit_config_fn=make_fit_config_fn(args),
        initial_parameters=initial_parameters,
    )
    evaluate_fn = make_evaluate_fn(args, logger, strategy_holder)
    if args.method == "gc_fedprox_replay":
        strategy = GCFedProxReplay(
            method=args.method,
            logger=logger,
            data_access=data_access,
            args=args,
            num_clusters=args.num_clusters,
            head_only=True,
            evaluate_fn=evaluate_fn,
            **common_kwargs,
        )
    else:
        strategy = LoggingFedAvg(
            method=args.method,
            logger=logger,
            data_access=data_access,
            args=args,
            evaluate_fn=evaluate_fn,
            **common_kwargs,
        )
    strategy_holder["strategy"] = strategy
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_global_tasks * args.rounds_per_task),
        strategy=strategy,
        client_resources={"num_cpus": args.client_cpus, "num_gpus": args.client_gpus},
    )
    print(f"Logs saved under: {logger.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Flower simulation baselines for FCL.")
    parser.add_argument("--method", choices=["fedavg", "fedprox_replay", "gc_fedprox_replay"], default="fedavg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=5)
    parser.add_argument("--num_global_tasks", type=int, default=5)
    parser.add_argument("--rounds_per_task", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--hf_dataset", default="zh-plus/tiny-imagenet")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--output_dir", default="./baselines")
    parser.add_argument("--drift_mode", choices=["data_only", "concept_only", "data_and_concept"], default="data_and_concept")
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--num_classes_per_task", type=int, default=20)
    parser.add_argument("--proximal_mu", type=float, default=0.01)
    parser.add_argument("--replay_size", type=int, default=200)
    parser.add_argument("--replay_alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--client_cpus", type=float, default=1.0)
    parser.add_argument("--client_gpus", type=float, default=0.0)
    parser.add_argument("--regenerate_data", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
