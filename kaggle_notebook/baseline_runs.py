"""
Self-contained Kaggle/Notebook Flower baselines for Federated Continual Learning.

Kaggle install cell:
    !pip -q install "flwr[simulation]" scikit-learn datasets

Quick run:
    from baseline_runs import run_kaggle_baselines
    run_kaggle_baselines(method="all")

CLI:
    python baseline_runs.py --method all
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

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


CORRUPTIONS = [
    "clean",
    "gaussian_noise",
    "shot_noise",
    "gaussian_blur",
    "motion_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
    "pixelate",
]
TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD = [0.2302, 0.2265, 0.2262]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CSVLogger:
    def __init__(self, output_dir: str | Path, config: dict | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers: dict[str, list[str]] = {}
        if config is not None:
            with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)

    @classmethod
    def timestamped(cls, root: str | Path, config: dict | None = None) -> "CSVLogger":
        return cls(Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S"), config)

    def log_row(self, filename: str, row: dict) -> None:
        path = self.output_dir / filename
        if filename not in self.headers:
            self.headers[filename] = list(row.keys())
        header = self.headers[filename]
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({k: self._stringify(row.get(k, "")) for k in header})

    @staticmethod
    def _stringify(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        return value


class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.classifier(z.view(z.size(0), -1))


def get_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters) -> None:
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, value in zip(state_dict.keys(), parameters):
        new_state_dict[key] = torch.tensor(value, dtype=state_dict[key].dtype)
    model.load_state_dict(new_state_dict, strict=True)


def clone_trainable_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def proximal_loss(model: torch.nn.Module, global_params: list[torch.Tensor]) -> torch.Tensor:
    prox = torch.zeros((), device=next(model.parameters()).device)
    for p, gp in zip([p for p in model.parameters() if p.requires_grad], global_params):
        prox = prox + torch.sum((p - gp.to(p.device)) ** 2)
    return prox


class ReplayBuffer:
    def __init__(self, max_size: int = 200) -> None:
        self.buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=max_size)

    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for xi, yi in zip(x.detach().cpu(), y.detach().cpu()):
            self.buffer.append((xi, yi))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        x, y = zip(*samples)
        return torch.stack(list(x)), torch.tensor([int(v) for v in y])

    def __len__(self) -> int:
        return len(self.buffer)


class ListDataset(Dataset):
    def __init__(self, samples) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        y = y if torch.is_tensor(y) else torch.tensor(int(y), dtype=torch.long)
        return x, y.long()


def load_pt_loader(path: str | Path, batch_size: int, shuffle: bool):
    return torch.utils.data.DataLoader(ListDataset(torch.load(path, map_location="cpu")), batch_size=batch_size, shuffle=shuffle)


def apply_corruption_pil(img: Image.Image, corruption_type: str, severity: int) -> Image.Image:
    img = img.convert("RGB")
    if corruption_type == "clean" or severity <= 0:
        return img
    severity = int(max(1, min(5, severity)))
    if corruption_type == "gaussian_noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0.0, [0.04, 0.08, 0.12, 0.16, 0.20][severity - 1], arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if corruption_type == "shot_noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        scale = [80, 50, 35, 20, 12][severity - 1]
        arr = np.clip(np.random.poisson(arr * scale) / float(scale), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if corruption_type == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=[0.5, 1.0, 1.5, 2.0, 3.0][severity - 1]))
    if corruption_type == "motion_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=[0.7, 1.0, 1.5, 2.0, 2.8][severity - 1]))
    if corruption_type == "brightness":
        return ImageEnhance.Brightness(img).enhance([0.6, 0.7, 0.85, 1.25, 1.5][severity - 1])
    if corruption_type == "contrast":
        return ImageEnhance.Contrast(img).enhance([0.5, 0.65, 0.8, 1.3, 1.6][severity - 1])
    if corruption_type == "pixelate":
        w, h = img.size
        scale = [0.75, 0.6, 0.5, 0.4, 0.3][severity - 1]
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.NEAREST)
    if corruption_type == "jpeg_compression":
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=[70, 50, 35, 20, 10][severity - 1])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    return img


class CorruptedSubset(Dataset):
    def __init__(self, base_dataset, indices, corruption_type="clean", severity=0, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.corruption_type = corruption_type
        self.severity = int(severity)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, y = self.base_dataset[self.indices[idx]]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        img = apply_corruption_pil(img, self.corruption_type, self.severity)
        return (self.transform(img) if self.transform else img), int(y)


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset) -> None:
        self.dataset = hf_dataset
        labels = list(hf_dataset["label"])
        self.valid_indices = [idx for idx, label in enumerate(labels) if int(label) >= 0]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        row = self.dataset[self.valid_indices[idx]]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        return image.convert("RGB"), int(row["label"])


def build_class_indices(dataset) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        out[int(y)].append(idx)
    return dict(out)


def dirichlet_split(indices, client_ids, alpha: float) -> dict[int, list[int]]:
    indices = list(indices)
    random.shuffle(indices)
    props = np.random.dirichlet(np.ones(len(client_ids)) * alpha)
    counts = np.floor(props * len(indices)).astype(int)
    while counts.sum() < len(indices):
        counts[np.random.randint(0, len(client_ids))] += 1
    out, start = {}, 0
    for cid, count in zip(client_ids, counts):
        out[int(cid)] = indices[start : start + int(count)]
        start += int(count)
    return out


def task_classes(num_classes_per_task: int, num_tasks: int, drift_mode: str) -> dict[int, list[int]]:
    if drift_mode == "data_only":
        return {t: list(range(num_classes_per_task)) for t in range(1, num_tasks + 1)}
    return {t: list(range((t - 1) * num_classes_per_task, t * num_classes_per_task)) for t in range(1, num_tasks + 1)}


def make_schedule(num_clients: int, num_tasks: int, drift_mode: str) -> dict[str, dict[str, dict]]:
    schedule = {}
    for cid in range(num_clients):
        local = {}
        for gtask in range(1, num_tasks + 1):
            if drift_mode == "data_only":
                ltask = 1
            else:
                r = random.random()
                ltask = max(1, gtask - 1) if r < 0.35 else min(num_tasks, gtask + 1) if r < 0.45 else gtask
            if drift_mode == "concept_only":
                ctype, severity = "clean", 0
            else:
                ctype = random.choice(CORRUPTIONS)
                severity = 0 if ctype == "clean" else random.randint(1, 5)
            local[str(gtask)] = {"local_task": ltask, "corruption_type": ctype, "severity": severity}
        schedule[str(cid)] = local
    return schedule


def materialize(ds: Dataset):
    return [ds[i] for i in range(len(ds))]


def load_base_dataset(args):
    from datasets import load_dataset

    hf_ds = load_dataset(args.hf_dataset, split=args.hf_split)
    return HFDatasetWrapper(hf_ds), args.hf_dataset, 200


def generate_data(args, data_root: Path) -> None:
    set_seed(args.seed)
    if data_root.exists() and args.regenerate_data:
        shutil.rmtree(data_root)
    if (data_root / "client_000" / "task_001" / "train.pt").exists():
        return
    data_root.mkdir(parents=True, exist_ok=True)
    base, dataset_name, available_classes = load_base_dataset(args)
    if args.num_global_tasks * args.num_classes_per_task > available_classes and args.drift_mode != "data_only":
        raise ValueError("num_global_tasks * num_classes_per_task exceeds available classes")
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=TINY_MEAN, std=TINY_STD),
        ]
    )
    cls_indices = build_class_indices(base)
    client_ids = list(range(args.num_clients))
    cls_splits = {cls: dirichlet_split(indices, client_ids, args.alpha) for cls, indices in cls_indices.items()}
    tasks = task_classes(args.num_classes_per_task, args.num_global_tasks, args.drift_mode)
    schedule = make_schedule(args.num_clients, args.num_global_tasks, args.drift_mode)
    for cid in client_ids:
        croot = data_root / f"client_{cid:03d}"
        croot.mkdir(parents=True, exist_ok=True)
        with open(croot / "schedule.json", "w", encoding="utf-8") as f:
            json.dump(schedule[str(cid)], f, indent=2)
    for gtask in range(1, args.num_global_tasks + 1):
        for cid in client_ids:
            item = schedule[str(cid)][str(gtask)]
            classes = tasks[int(item["local_task"])]
            indices = []
            for cls in classes:
                indices.extend(cls_splits[int(cls)].get(cid, []))
            random.shuffle(indices)
            n_val = int(0.2 * len(indices))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            out = data_root / f"client_{cid:03d}" / f"task_{gtask:03d}"
            out.mkdir(parents=True, exist_ok=True)
            train = materialize(CorruptedSubset(base, train_idx, item["corruption_type"], item["severity"], transform))
            val = materialize(CorruptedSubset(base, val_idx, item["corruption_type"], item["severity"], transform))
            torch.save(train, out / "train.pt")
            torch.save(val, out / "val.pt")
            hist = Counter(int(y) for _, y in train)
            with open(out / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cid": cid,
                        "global_task": gtask,
                        "local_task": int(item["local_task"]),
                        "class_ids": classes,
                        "num_samples": len(train),
                        "label_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
                        "data_drift": {"corruption_type": item["corruption_type"], "severity": int(item["severity"])},
                        "report_mode": "oracle",
                    },
                    f,
                    indent=2,
                )
    test_root = data_root / "global_test"
    test_root.mkdir(exist_ok=True)
    seen: set[int] = set()
    for gtask in range(1, args.num_global_tasks + 1):
        classes = tasks[gtask]
        task_idx = []
        for cls in classes:
            take = max(1, int(0.2 * len(cls_indices[int(cls)])))
            task_idx.extend(cls_indices[int(cls)][:take])
        torch.save(materialize(CorruptedSubset(base, task_idx, "clean", 0, transform)), test_root / f"task_{gtask:03d}_classes.pt")
        seen.update(classes)
        seen_idx = []
        for cls in sorted(seen):
            take = max(1, int(0.2 * len(cls_indices[int(cls)])))
            seen_idx.extend(cls_indices[int(cls)][:take])
        torch.save(materialize(CorruptedSubset(base, seen_idx, "clean", 0, transform)), test_root / f"task_{gtask:03d}_seen_classes.pt")
    with open(data_root / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, **vars(args)}, f, indent=2, sort_keys=True)


def compute_label_entropy(hist: dict) -> float:
    counts = np.asarray([float(v) for v in hist.values()], dtype=np.float64)
    if counts.sum() <= 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def flatten_delta(old_params, new_params) -> np.ndarray:
    return np.concatenate([(np.asarray(n) - np.asarray(o)).reshape(-1) for o, n in zip(old_params, new_params)])


def update_norm(old_params, new_params) -> float:
    return float(np.linalg.norm(flatten_delta(old_params, new_params)))


def label_hist_from_loader(loader) -> dict[str, int]:
    hist = Counter()
    for _, y in loader:
        for item in y.cpu().tolist():
            hist[int(item)] += 1
    return {str(k): int(v) for k, v in sorted(hist.items())}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str) -> dict:
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(ce(logits, y).item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(x.size(0))
    return {"loss": total_loss / max(1, total), "accuracy": correct / max(1, total), "num_samples": total}


def train_one_client(model, trainloader, device, epochs, lr, proximal_mu=0.0, global_params=None, replay_buffer=None, replay_alpha=0.0) -> dict:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = torch.nn.CrossEntropyLoss()
    total_loss, steps = 0.0, 0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            if replay_buffer is not None and len(replay_buffer) > 0 and replay_alpha > 0:
                xr, yr = replay_buffer.sample(x.size(0))
                loss = loss + replay_alpha * ce(model(xr.to(device)), yr.to(device))
            if proximal_mu > 0 and global_params is not None:
                loss = loss + (proximal_mu / 2.0) * proximal_loss(model, global_params)
            loss.backward()
            opt.step()
            if replay_buffer is not None:
                replay_buffer.add_batch(x, y)
            total_loss += float(loss.item())
            steps += 1
    return {"train_loss": total_loss / max(1, steps), "num_steps": steps}


class GeneratedDataAccess:
    def __init__(self, data_root: str | Path, batch_size: int) -> None:
        self.data_root = Path(data_root)
        self.batch_size = batch_size

    def trainloader(self, cid: str, global_task: int):
        return load_pt_loader(self.data_root / f"client_{int(cid):03d}" / f"task_{global_task:03d}" / "train.pt", self.batch_size, True)

    def valloader(self, cid: str, global_task: int):
        return load_pt_loader(self.data_root / f"client_{int(cid):03d}" / f"task_{global_task:03d}" / "val.pt", self.batch_size, False)

    def testloader(self, global_task: int):
        return load_pt_loader(self.data_root / "global_test" / f"task_{global_task:03d}_seen_classes.pt", self.batch_size, False)

    def metadata(self, cid: str, global_task: int) -> dict:
        with open(self.data_root / f"client_{int(cid):03d}" / f"task_{global_task:03d}" / "metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)


class FCLClient(fl.client.NumPyClient if fl is not None else object):
    def __init__(self, cid, data_access, method, num_classes, replay_buffer, device):
        if fl is None:
            raise RuntimeError(f"Flower is required: {FLOWER_IMPORT_ERROR}")
        self.cid = str(cid)
        self.model = SmallCNN(num_classes).to(device)
        self.data_access = data_access
        self.method = method
        self.replay_buffer = replay_buffer
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        start = time.time()
        gtask = int(config.get("global_task", 1))
        set_parameters(self.model, parameters)
        old_params = [np.asarray(p).copy() for p in parameters]
        trainloader = self.data_access.trainloader(self.cid, gtask)
        valloader = self.data_access.valloader(self.cid, gtask)
        before = evaluate(self.model, valloader, self.device)
        global_params, proximal_mu, replay_alpha, replay = None, 0.0, 0.0, None
        if self.method in {"fedprox_replay", "gc_fedprox_replay"}:
            global_params = clone_trainable_params(self.model)
            proximal_mu = float(config.get("proximal_mu", 0.01))
            replay_alpha = float(config.get("replay_alpha", 1.0))
            replay = self.replay_buffer
        train_metrics = train_one_client(
            self.model,
            trainloader,
            self.device,
            int(config.get("local_epochs", 1)),
            float(config.get("lr", 0.01)),
            proximal_mu,
            global_params,
            replay,
            replay_alpha,
        )
        after = evaluate(self.model, valloader, self.device)
        new_params = self.get_parameters(config)
        meta = self.data_access.metadata(self.cid, gtask)
        hist = meta.get("label_histogram") or label_hist_from_loader(trainloader)
        report = {
            "cid": int(self.cid),
            "num_samples": int(sum(int(v) for v in hist.values())),
            "label_histogram": hist,
            "label_entropy": compute_label_entropy(hist),
            "class_coverage_ratio": len(hist) / max(1, len(meta.get("class_ids", []))),
            "data_drift": meta.get("data_drift", {}),
            "local_eval": {"loss_before": before["loss"], "loss_after": after["loss"], "local_val_acc": after["accuracy"]},
            "update_stats": {"update_norm": update_norm(old_params, new_params), "cosine_to_global_update": 0.0},
        }
        metrics = {
            **train_metrics,
            "fit_time_sec": time.time() - start,
            "cid": int(self.cid),
            "buffer_size": len(self.replay_buffer) if self.replay_buffer is not None else 0,
            "client_report_json": json.dumps(report, sort_keys=True),
        }
        return new_params, len(trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.data_access.testloader(int(config.get("global_task", 1))), self.device)
        return float(metrics["loss"]), int(metrics["num_samples"]), {"accuracy": float(metrics["accuracy"])}


def get_cid(context_or_cid) -> str:
    if isinstance(context_or_cid, str):
        return context_or_cid
    node_config = getattr(context_or_cid, "node_config", {}) or {}
    return str(node_config.get("partition-id", node_config.get("cid", context_or_cid)))


def round_to_task(server_round: int, rounds_per_task: int, num_tasks: int) -> int:
    return min(num_tasks, max(1, math.ceil(max(1, server_round) / rounds_per_task)))


def seen_classes_for_task(global_task: int, num_classes_per_task: int, drift_mode: str) -> list[int]:
    if drift_mode == "data_only":
        return list(range(num_classes_per_task))
    return list(range(global_task * num_classes_per_task))


class LoggingFedAvg(fl.server.strategy.FedAvg if fl is not None else object):
    def __init__(self, *, method, logger, args, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.logger = logger
        self.args = args
        self.current_global = None
        self.round_starts = {}
        self.round_times = {}
        self.num_fit_clients = {}
        self.best_task_acc = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_global = parameters_to_ndarrays(parameters)
        self.round_starts[server_round] = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        gtask = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        self.num_fit_clients[server_round] = len(results)
        for client_proxy, fit_res in results:
            m = dict(fit_res.metrics)
            self.logger.log_row(
                "client_fit_metrics.csv",
                {
                    "method": self.method,
                    "round": server_round,
                    "global_task": gtask,
                    "cid": int(m.get("cid", getattr(client_proxy, "cid", -1))),
                    "num_samples": int(fit_res.num_examples),
                    "fit_time_sec": float(m.get("fit_time_sec", 0.0)),
                    "train_loss": float(m.get("train_loss", 0.0)),
                    "buffer_size": int(m.get("buffer_size", 0)),
                },
            )
        out = super().aggregate_fit(server_round, results, failures)
        self.round_times[server_round] = time.time() - self.round_starts.get(server_round, time.time())
        return out

    def log_task_evals(self, server_round: int, parameters) -> None:
        if server_round <= 0 or server_round % self.args.rounds_per_task != 0:
            return
        gtask = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        model = SmallCNN(self.args.num_classes).to(self.args.device)
        set_parameters(model, parameters_to_ndarrays(parameters))
        for eval_task in range(1, gtask + 1):
            loader = load_pt_loader(Path(self.args.data_root_resolved) / "global_test" / f"task_{eval_task:03d}_classes.pt", self.args.batch_size, False)
            metrics = evaluate(model, loader, self.args.device)
            acc = float(metrics["accuracy"])
            self.best_task_acc[eval_task] = max(self.best_task_acc.get(eval_task, acc), acc)
            self.logger.log_row(
                "task_eval_metrics.csv",
                {
                    "method": self.method,
                    "global_task": gtask,
                    "eval_task": eval_task,
                    "seen_classes": seen_classes_for_task(eval_task, self.args.num_classes_per_task, self.args.drift_mode),
                    "accuracy": acc,
                    "loss": float(metrics["loss"]),
                    "forgetting": float(self.best_task_acc[eval_task] - acc),
                },
            )


class GCFedProxReplay(LoggingFedAvg):
    def __init__(self, *, num_clusters=3, head_only=True, **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.head_only = head_only

    def aggregate_fit(self, server_round, results, failures):
        gtask = round_to_task(server_round, self.args.rounds_per_task, self.args.num_global_tasks)
        if self.current_global is not None and len(results) >= self.num_clusters:
            vectors, cids = [], []
            for client_proxy, fit_res in results:
                params = parameters_to_ndarrays(fit_res.parameters)
                delta = [cp - gp for cp, gp in zip(params, self.current_global)]
                selected = delta[-2:] if self.head_only else delta
                vectors.append(np.concatenate([x.reshape(-1) for x in selected]))
                cids.append(int(fit_res.metrics.get("cid", getattr(client_proxy, "cid", -1))))
            z = np.stack(vectors)
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            if KMeans is not None:
                cluster_ids = KMeans(n_clusters=self.num_clusters, n_init="auto", random_state=server_round).fit_predict(z)
            else:
                cluster_ids = np.arange(len(z)) % self.num_clusters
            sim = z @ z.T
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
                    "global_task": gtask,
                    "client_ids": cids,
                    "cluster_ids": cluster_ids.tolist(),
                    "cluster_sizes": [int(np.sum(cluster_ids == k)) for k in range(self.num_clusters)],
                    "mean_cosine": float(np.mean(sim)),
                    "intra_cluster_similarity": float(np.mean(intra)) if intra else 0.0,
                },
            )
        return super().aggregate_fit(server_round, results, failures)


def fit_config_fn(args, method):
    def fit_config(server_round: int):
        return {
            "global_task": round_to_task(server_round, args.rounds_per_task, args.num_global_tasks),
            "local_epochs": args.local_epochs,
            "proximal_mu": args.proximal_mu if method != "fedavg" else 0.0,
            "replay_alpha": args.replay_alpha if method != "fedavg" else 0.0,
            "lr": args.lr,
        }

    return fit_config


def evaluate_fn(args, logger, strategy_holder, method):
    def _eval(server_round, parameters, config):
        if server_round <= 0:
            return None
        gtask = round_to_task(server_round, args.rounds_per_task, args.num_global_tasks)
        model = SmallCNN(args.num_classes).to(args.device)
        set_parameters(model, parameters)
        loader = load_pt_loader(Path(args.data_root_resolved) / "global_test" / f"task_{gtask:03d}_seen_classes.pt", args.batch_size, False)
        metrics = evaluate(model, loader, args.device)
        strategy = strategy_holder["strategy"]
        logger.log_row(
            "round_metrics.csv",
            {
                "method": method,
                "round": server_round,
                "global_task": gtask,
                "accuracy_seen_classes": float(metrics["accuracy"]),
                "loss_seen_classes": float(metrics["loss"]),
                "round_time_sec": float(strategy.round_times.get(server_round, 0.0)),
                "num_selected_clients": int(strategy.num_fit_clients.get(server_round, args.clients_per_round)),
            },
        )
        strategy.log_task_evals(server_round, ndarrays_to_parameters(parameters))
        return float(metrics["loss"]), {"accuracy_seen_classes": float(metrics["accuracy"])}

    return _eval


def run_one_baseline(args, method: str) -> Path:
    if fl is None:
        raise RuntimeError(f"Flower is required. Install with: pip install 'flwr[simulation]'. Original error: {FLOWER_IMPORT_ERROR}")
    set_seed(args.seed)
    data_root = Path(args.output_dir) / "generated_data"
    generate_data(args, data_root)
    args.data_root_resolved = str(data_root)
    logger = CSVLogger.timestamped(Path(args.output_dir) / "logs" / method, config={**vars(args), "method": method})
    data_access = GeneratedDataAccess(data_root, args.batch_size)
    replay_buffers = {str(cid): ReplayBuffer(args.replay_size) for cid in range(args.num_clients)}

    def client_fn(context):
        cid = get_cid(context)
        replay = replay_buffers[cid] if method != "fedavg" else None
        return FCLClient(cid, data_access, method, args.num_classes, replay, args.device).to_client()

    holder = {}
    common = dict(
        fraction_fit=args.clients_per_round / args.num_clients,
        fraction_evaluate=0.0,
        min_fit_clients=args.clients_per_round,
        min_available_clients=args.num_clients,
        on_fit_config_fn=fit_config_fn(args, method),
        evaluate_fn=evaluate_fn(args, logger, holder, method),
        initial_parameters=ndarrays_to_parameters(get_parameters(SmallCNN(args.num_classes))),
    )
    if method == "gc_fedprox_replay":
        strategy = GCFedProxReplay(method=method, logger=logger, args=args, num_clusters=args.num_clusters, **common)
    else:
        strategy = LoggingFedAvg(method=method, logger=logger, args=args, **common)
    holder["strategy"] = strategy
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_global_tasks * args.rounds_per_task),
        strategy=strategy,
        client_resources={"num_cpus": args.client_cpus, "num_gpus": args.client_gpus},
    )
    print(f"[{method}] logs: {logger.output_dir}")
    return logger.output_dir


def run_kaggle_baselines(**kwargs) -> list[Path]:
    parser = build_arg_parser()
    defaults = vars(parser.parse_args([]))
    defaults.update(kwargs)
    args = argparse.Namespace(**defaults)
    args.device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    methods = ["fedavg", "fedprox_replay", "gc_fedprox_replay"] if args.method == "all" else [args.method]
    out = []
    for idx, method in enumerate(methods):
        if idx > 0:
            args.regenerate_data = False
        out.append(run_one_baseline(args, method))
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Self-contained Kaggle Flower baselines.")
    parser.add_argument("--method", choices=["all", "fedavg", "fedprox_replay", "gc_fedprox_replay"], default="all")
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
    parser.add_argument("--output_dir", default="./fcl_baseline_logs")
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
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_kaggle_baselines(**vars(cli_args))
