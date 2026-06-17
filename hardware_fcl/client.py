from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

try:
    import flwr as fl
except Exception:  # pragma: no cover
    fl = None

from hardware_fcl.model import create_model
from hardware_fcl.utils import (
    ListDataset,
    ReplayBuffer,
    clone_trainable_params,
    compute_cosine_to_global_update,
    compute_label_entropy,
    compute_prediction_entropy,
    compute_update_norm,
    estimate_drift_type,
    estimate_image_drift_features,
    flatten_delta,
    get_parameters,
    label_histogram_from_loader,
    load_pt_loader,
    proximal_loss,
    read_json,
    set_parameters,
)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> dict:
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    entropy_values: list[float] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))
        entropy_values.append(compute_prediction_entropy(logits))
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "num_samples": total,
        "prediction_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
    }


def train_one_client(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: str,
    epochs: int,
    lr: float = 0.01,
    proximal_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
    replay_buffer: ReplayBuffer | None = None,
    replay_alpha: float = 0.0,
) -> dict:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_steps = 0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            if replay_buffer is not None and len(replay_buffer) > 0 and replay_alpha > 0:
                xr, yr = replay_buffer.sample(batch_size=x.size(0))
                xr, yr = xr.to(device), yr.to(device)
                loss = loss + replay_alpha * ce(model(xr), yr)
            if proximal_mu > 0.0 and global_params is not None:
                loss = loss + (proximal_mu / 2.0) * proximal_loss(model, global_params)
            loss.backward()
            opt.step()
            if replay_buffer is not None:
                replay_buffer.add_batch(x, y)
            total_loss += float(loss.item())
            total_steps += 1
    return {"train_loss": total_loss / max(1, total_steps), "num_steps": total_steps}


class GeneratedDataAccess:
    def __init__(self, data_root: str | Path, batch_size: int) -> None:
        self.data_root = Path(data_root)
        self.batch_size = batch_size

    def trainloader(self, cid: str, global_task: int) -> torch.utils.data.DataLoader:
        return load_pt_loader(
            self.data_root / f"client_{int(cid):03d}" / f"task_{global_task:03d}" / "train.pt",
            batch_size=self.batch_size,
            shuffle=True,
        )

    def valloader(self, cid: str, global_task: int) -> torch.utils.data.DataLoader:
        return load_pt_loader(
            self.data_root / f"client_{int(cid):03d}" / f"task_{global_task:03d}" / "val.pt",
            batch_size=self.batch_size,
            shuffle=False,
        )

    def testloader(self, global_task: int) -> torch.utils.data.DataLoader:
        return load_pt_loader(
            self.data_root / "global_test" / f"task_{global_task:03d}_seen_classes.pt",
            batch_size=self.batch_size,
            shuffle=False,
        )

    def metadata(self, cid: str, global_task: int) -> dict:
        return read_json(self.data_root / f"client_{int(cid):03d}" / f"task_{global_task:03d}" / "metadata.json")


def build_client_report(
    cid: int,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    metadata: dict,
    report_mode: str,
    loss_before: float,
    eval_after: dict,
    old_params: list[np.ndarray],
    new_params: list[np.ndarray],
    reference_delta: np.ndarray | None = None,
) -> dict:
    label_hist = metadata.get("label_histogram") or label_histogram_from_loader(trainloader)
    num_samples = int(sum(int(v) for v in label_hist.values()))
    class_ids = metadata.get("class_ids", [])
    report = {
        "cid": int(cid),
        "num_samples": num_samples,
        "label_histogram": label_hist,
        "label_entropy": compute_label_entropy(label_hist),
        "class_coverage_ratio": len(label_hist) / max(1, len(class_ids)),
        "local_eval": {
            "loss_before": float(loss_before),
            "loss_after": float(eval_after["loss"]),
            "local_val_acc": float(eval_after["accuracy"]),
            "prediction_entropy": float(eval_after.get("prediction_entropy", 0.0)),
        },
        "update_stats": {
            "update_norm": compute_update_norm(old_params, new_params),
            "cosine_to_global_update": compute_cosine_to_global_update(flatten_delta(old_params, new_params), reference_delta),
        },
    }
    if report_mode == "estimated":
        features = estimate_image_drift_features(trainloader)
        estimated_type, estimated_severity, confidence = estimate_drift_type(features)
        report["data_drift"] = {
            "estimated_type": estimated_type,
            "estimated_severity": int(estimated_severity),
            "confidence": float(confidence),
            "features": features,
        }
    else:
        report["data_drift"] = metadata.get("data_drift", {"corruption_type": "unknown", "severity": 0})
    return report


class FCLClient(fl.client.NumPyClient if fl is not None else object):
    def __init__(
        self,
        cid: int | str,
        data_access: GeneratedDataAccess,
        method: str,
        num_classes: int = 200,
        replay_buffer: ReplayBuffer | None = None,
        device: str = "cpu",
        report_mode: str = "oracle",
    ) -> None:
        if fl is None:
            raise RuntimeError("Flower is not installed. Install flwr to use FCLClient.")
        self.cid = str(cid)
        self.model = create_model(num_classes=num_classes).to(device)
        self.data_access = data_access
        self.method = method
        self.replay_buffer = replay_buffer
        self.device = device
        self.report_mode = report_mode

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        start = time.time()
        global_task = int(config.get("global_task", 1))
        self.set_parameters(parameters)
        old_params = [np.asarray(p).copy() for p in parameters]
        trainloader = self.data_access.trainloader(self.cid, global_task)
        valloader = self.data_access.valloader(self.cid, global_task)
        before = evaluate(self.model, valloader, self.device)
        global_params = None
        proximal_mu = 0.0
        replay_alpha = 0.0
        replay_buffer = None
        if self.method in {"fedprox_replay", "gc_fedprox_replay", "slm"}:
            global_params = clone_trainable_params(self.model)
            proximal_mu = float(config.get("proximal_mu", 0.01))
            replay_alpha = float(config.get("replay_alpha", 1.0))
            replay_buffer = self.replay_buffer
        train_metrics = train_one_client(
            model=self.model,
            trainloader=trainloader,
            device=self.device,
            epochs=int(config.get("local_epochs", 1)),
            lr=float(config.get("lr", 0.01)),
            proximal_mu=proximal_mu,
            global_params=global_params,
            replay_buffer=replay_buffer,
            replay_alpha=replay_alpha,
        )
        after = evaluate(self.model, valloader, self.device)
        new_params = self.get_parameters(config)
        metadata = self.data_access.metadata(self.cid, global_task)
        report = build_client_report(
            cid=int(self.cid),
            trainloader=trainloader,
            valloader=valloader,
            metadata=metadata,
            report_mode=str(config.get("report_mode", self.report_mode)),
            loss_before=float(before["loss"]),
            eval_after=after,
            old_params=old_params,
            new_params=new_params,
        )
        fit_time = time.time() - start
        metrics = {
            **train_metrics,
            "fit_time_sec": float(fit_time),
            "cid": int(self.cid),
            "buffer_size": len(self.replay_buffer) if self.replay_buffer is not None else 0,
            "client_report_json": json.dumps(report, sort_keys=True),
        }
        return new_params, len(trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        global_task = int(config.get("global_task", 1))
        loader = self.data_access.testloader(global_task)
        metrics = evaluate(self.model, loader, self.device)
        return float(metrics["loss"]), int(metrics["num_samples"]), {"accuracy": float(metrics["accuracy"])}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Flower FCL client on Raspberry Pi.")
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server_address", default="192.168.0.141:8080")
    parser.add_argument("--data_root", default="./hardware_fcl/data/generated")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--method", choices=["fedavg", "fedprox_replay", "gc_fedprox_replay", "slm"], default="slm")
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--replay_size", type=int, default=200)
    parser.add_argument("--report_mode", choices=["oracle", "estimated"], default="oracle")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    if fl is None:
        raise RuntimeError("Flower is required. Install with `pip install flwr`.")
    args = parse_args()
    replay = ReplayBuffer(args.replay_size) if args.method != "fedavg" else None
    client = FCLClient(
        cid=args.cid,
        data_access=GeneratedDataAccess(args.data_root, args.batch_size),
        method=args.method,
        num_classes=args.num_classes,
        replay_buffer=replay,
        device=args.device,
        report_mode=args.report_mode,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
