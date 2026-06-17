from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
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

from hardware_fcl.client import FCLClient, GeneratedDataAccess, evaluate
from hardware_fcl.dataset_generator import generate_datasets
from hardware_fcl.llm_decision import heuristic_select_clients, parse_llm_json
from hardware_fcl.logger import CSVLogger
from hardware_fcl.model import create_model
from hardware_fcl.prompt import build_client_selection_prompt
from hardware_fcl.utils import (
    ReplayBuffer,
    compute_label_entropy,
    get_parameters,
    load_pt_loader,
    seen_classes_for_task,
    set_parameters,
    set_seed,
)


def round_to_task(server_round: int, rounds_per_task: int, num_global_tasks: int) -> int:
    return min(num_global_tasks, max(1, math.ceil(max(1, server_round) / rounds_per_task)))


class HFSelector:
    def __init__(self, model: str) -> None:
        self.model = model
        self.pipe = None

    def select(self, prompt: str, reports: list[dict], clients_per_round: int) -> dict:
        try:
            if self.pipe is None:
                from transformers import pipeline

                self.pipe = pipeline(task="text-generation", model=self.model, device_map="auto")
            start = time.time()
            out = self.pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
            raw = out[len(prompt) :] if out.startswith(prompt) else out
            parsed = parse_llm_json(raw)
            valid = {int(r["cid"]) for r in reports}
            selected = [int(x) for x in parsed.get("selected_clients", []) if int(x) in valid]
            if len(selected) != clients_per_round:
                selected = heuristic_select_clients(reports, clients_per_round)
                parsed["repair_reason"] = "invalid selected_clients"
            return {
                "selected_clients": selected,
                "llm_inference_time_sec": time.time() - start,
                "used_fallback": False,
                "raw_response": raw,
                "reason": parsed.get("reason", ""),
            }
        except Exception as exc:
            start = time.time()
            selected = heuristic_select_clients(reports, clients_per_round)
            return {
                "selected_clients": selected,
                "llm_inference_time_sec": time.time() - start,
                "used_fallback": True,
                "raw_response": "",
                "reason": f"HF fallback: {exc}",
            }


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
            report_mode=args.report_mode,
            overwrite=True,
        )
    return data_root


def report_drift(report: dict) -> tuple[str, int]:
    drift = report.get("data_drift", {})
    dtype = drift.get("corruption_type") or drift.get("estimated_type") or "unknown"
    sev = drift.get("severity") if "severity" in drift else drift.get("estimated_severity", 0)
    return str(dtype), int(sev)


def log_selected_diversity(logger: CSVLogger, server_round: int, global_task: int, selected_reports: list[dict]) -> None:
    labels = Counter()
    corruptions = []
    severities = []
    norms = []
    for report in selected_reports:
        labels.update({str(k): int(v) for k, v in report.get("label_histogram", {}).items()})
        dtype, sev = report_drift(report)
        corruptions.append(dtype)
        severities.append(sev)
        norms.append(float(report.get("update_stats", {}).get("update_norm", 0.0)))
    logger.log_row(
        "selected_client_diversity.csv",
        {
            "round": server_round,
            "global_task": global_task,
            "selected_clients": [int(r["cid"]) for r in selected_reports],
            "selected_label_coverage": len(labels),
            "selected_label_entropy": compute_label_entropy(dict(labels)),
            "selected_corruption_types": sorted(set(corruptions)),
            "selected_severity_mean": float(np.mean(severities)) if severities else 0.0,
            "selected_update_norm_mean": float(np.mean(norms)) if norms else 0.0,
            "selected_update_norm_std": float(np.std(norms)) if norms else 0.0,
        },
    )


class SLMSelectionFedAvg(fl.server.strategy.FedAvg if fl is not None else object):
    def __init__(self, *, args, logger: CSVLogger, selector: HFSelector | None, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.logger = logger
        self.selector = selector
        self.round_starts: dict[int, float] = {}
        self.round_times: dict[int, float] = {}
        self.selected_counts: dict[int, int] = {}
        self.best_task_acc: dict[int, float] = {}

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
                "client_fit_metrics.csv",
                {
                    "round": server_round,
                    "global_task": global_task,
                    "cid": cid,
                    "num_samples": int(fit_res.num_examples),
                    "fit_time_sec": float(metrics.get("fit_time_sec", 0.0)),
                    "train_loss": float(metrics.get("train_loss", 0.0)),
                    "buffer_size": int(metrics.get("buffer_size", 0)),
                },
            )
        prompt = build_client_selection_prompt(
            round_id=server_round,
            global_task=global_task,
            clients_per_round=self.args.clients_per_round,
            seen_classes=seen_classes_for_task(global_task, self.args.num_classes_per_task, self.args.drift_mode),
            client_reports=reports,
        )
        if self.args.llm_backend == "hf" and self.selector is not None:
            decision = self.selector.select(prompt, reports, self.args.clients_per_round)
        else:
            start = time.time()
            selected = heuristic_select_clients(reports, self.args.clients_per_round)
            decision = {
                "selected_clients": selected,
                "llm_inference_time_sec": time.time() - start,
                "used_fallback": True,
                "raw_response": "",
                "reason": "heuristic backend",
            }
        selected = [cid for cid in decision["selected_clients"] if cid in cid_to_result]
        selected_results = [cid_to_result[cid] for cid in selected]
        selected_reports = [r for r in reports if int(r["cid"]) in set(selected)]
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
                "reason": decision.get("reason", ""),
            },
        )
        log_selected_diversity(self.logger, server_round, global_task, selected_reports)
        aggregated = super().aggregate_fit(server_round, selected_results, failures)
        self.round_times[server_round] = time.time() - self.round_starts.get(server_round, time.time())
        return aggregated

    def log_task_evals(self, server_round: int, parameters) -> None:
        if server_round <= 0 or server_round % self.args.rounds_per_task != 0:
            return
        global_task = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        model = create_model(self.args.num_classes).to(self.args.device)
        set_parameters(model, parameters_to_ndarrays(parameters))
        for eval_task in range(1, global_task + 1):
            path = Path(self.args.data_root_resolved) / "global_test" / f"task_{eval_task:03d}_classes.pt"
            metrics = evaluate(model, load_pt_loader(path, self.args.batch_size, False), self.args.device)
            acc = float(metrics["accuracy"])
            self.best_task_acc[eval_task] = max(self.best_task_acc.get(eval_task, acc), acc)
            self.logger.log_row(
                "task_eval_metrics.csv",
                {
                    "global_task": global_task,
                    "eval_task": eval_task,
                    "seen_classes": seen_classes_for_task(eval_task, self.args.num_classes_per_task, self.args.drift_mode),
                    "accuracy": acc,
                    "loss": float(metrics["loss"]),
                    "forgetting": float(self.best_task_acc[eval_task] - acc),
                },
            )


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


def make_evaluate_fn(args, logger: CSVLogger, holder: dict):
    def evaluate_fn(server_round, parameters, config):
        if server_round <= 0:
            return None
        global_task = round_to_task(server_round, args.rounds_per_task, args.num_global_tasks)
        model = create_model(args.num_classes).to(args.device)
        set_parameters(model, parameters)
        loader = load_pt_loader(Path(args.data_root_resolved) / "global_test" / f"task_{global_task:03d}_seen_classes.pt", args.batch_size, False)
        metrics = evaluate(model, loader, args.device)
        strategy = holder["strategy"]
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
        strategy.log_task_evals(server_round, ndarrays_to_parameters(parameters))
        return float(metrics["loss"]), {"accuracy_seen_classes": float(metrics["accuracy"])}

    return evaluate_fn


def get_cid(context_or_cid) -> str:
    if isinstance(context_or_cid, str):
        return context_or_cid
    node_config = getattr(context_or_cid, "node_config", {}) or {}
    return str(node_config.get("partition-id", node_config.get("cid", context_or_cid)))


def run(args) -> None:
    if fl is None:
        raise RuntimeError(f"Flower is required for SLM simulation: {FLOWER_IMPORT_ERROR}")
    set_seed(args.seed)
    args.device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ensure_generated_data(args)
    args.data_root_resolved = str(data_root)
    logger = CSVLogger.timestamped(Path(args.output_dir) / "logs", config=vars(args))
    data_access = GeneratedDataAccess(data_root, args.batch_size)
    replay_buffers = {str(cid): ReplayBuffer(max_size=args.replay_size) for cid in range(args.num_clients)}

    def client_fn(context):
        cid = get_cid(context)
        return FCLClient(
            cid=cid,
            data_access=data_access,
            method="slm",
            num_classes=args.num_classes,
            replay_buffer=replay_buffers[cid],
            device=args.device,
            report_mode=args.report_mode,
        ).to_client()

    selector = HFSelector(args.hf_model) if args.llm_backend == "hf" else None
    initial = ndarrays_to_parameters(get_parameters(create_model(args.num_classes)))
    holder = {}
    strategy = SLMSelectionFedAvg(
        args=args,
        logger=logger,
        selector=selector,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_fit_config_fn=make_fit_config_fn(args),
        evaluate_fn=make_evaluate_fn(args, logger, holder),
        initial_parameters=initial,
    )
    holder["strategy"] = strategy
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_global_tasks * args.rounds_per_task),
        strategy=strategy,
        client_resources={"num_cpus": args.client_cpus, "num_gpus": args.client_gpus},
    )
    print(f"Logs saved under: {logger.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Flower simulation for SLM-guided FCL.")
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
    parser.add_argument("--output_dir", default="./slm_simulation")
    parser.add_argument("--drift_mode", choices=["data_only", "concept_only", "data_and_concept"], default="data_and_concept")
    parser.add_argument("--report_mode", choices=["oracle", "estimated"], default="oracle")
    parser.add_argument("--llm_backend", choices=["heuristic", "hf"], default="heuristic")
    parser.add_argument("--hf_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--num_classes_per_task", type=int, default=20)
    parser.add_argument("--proximal_mu", type=float, default=0.01)
    parser.add_argument("--replay_size", type=int, default=200)
    parser.add_argument("--replay_alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--client_cpus", type=float, default=1.0)
    parser.add_argument("--client_gpus", type=float, default=0.0)
    parser.add_argument("--regenerate_data", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
