from __future__ import annotations

import json
import math
import random
from collections import Counter, deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD = [0.2302, 0.2265, 0.2262]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: Sequence[np.ndarray]) -> None:
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, value in zip(state_dict.keys(), parameters):
        new_state_dict[key] = torch.tensor(value, dtype=state_dict[key].dtype)
    model.load_state_dict(new_state_dict, strict=True)


def clone_trainable_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def proximal_loss(model: torch.nn.Module, global_params: Sequence[torch.Tensor]) -> torch.Tensor:
    prox = torch.zeros((), device=next(model.parameters()).device)
    params = [p for p in model.parameters() if p.requires_grad]
    for p, gp in zip(params, global_params):
        prox = prox + torch.sum((p - gp.to(p.device)) ** 2)
    return prox


def weighted_average(
    client_params_list: Sequence[Sequence[np.ndarray]],
    num_samples_list: Sequence[int],
) -> list[np.ndarray]:
    total = float(sum(num_samples_list))
    if total <= 0:
        raise ValueError("Cannot aggregate zero samples")
    averaged: list[np.ndarray] = []
    for params in zip(*client_params_list):
        avg = sum((n / total) * p for p, n in zip(params, num_samples_list))
        averaged.append(np.asarray(avg))
    return averaged


def compute_label_entropy(label_histogram: dict) -> float:
    counts = np.asarray([float(v) for v in label_histogram.values()], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    return float(-(probs * np.log(probs + 1e-12)).sum())


def compute_prediction_entropy(logits: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    probs = torch.softmax(logits.detach(), dim=1).clamp_min(1e-12)
    entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
    return float(entropy.item())


def flatten_delta(old_params: Sequence[np.ndarray], new_params: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([(np.asarray(n) - np.asarray(o)).reshape(-1) for o, n in zip(old_params, new_params)])


def compute_update_norm(old_params: Sequence[np.ndarray], new_params: Sequence[np.ndarray]) -> float:
    delta = flatten_delta(old_params, new_params)
    return float(np.linalg.norm(delta))


def compute_cosine_to_global_update(client_delta: np.ndarray, reference_delta: np.ndarray | None) -> float:
    if reference_delta is None or client_delta.size == 0 or reference_delta.size == 0:
        return 0.0
    denom = float(np.linalg.norm(client_delta) * np.linalg.norm(reference_delta))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(client_delta, reference_delta) / denom)


def label_histogram_from_loader(loader: torch.utils.data.DataLoader) -> dict[str, int]:
    hist: Counter[int] = Counter()
    for _, y in loader:
        for label in y.detach().cpu().tolist():
            hist[int(label)] += 1
    return {str(k): int(v) for k, v in sorted(hist.items())}


def estimate_image_drift_features(dataset_or_loader, max_batches: int = 8) -> dict[str, float]:
    xs: list[torch.Tensor] = []
    if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
        iterator = iter(dataset_or_loader)
        for batch_idx, (x, _) in enumerate(iterator):
            xs.append(x.detach().cpu())
            if batch_idx + 1 >= max_batches:
                break
    else:
        limit = min(len(dataset_or_loader), max_batches * 32)
        for idx in range(limit):
            x, _ = dataset_or_loader[idx]
            xs.append(x.detach().cpu().unsqueeze(0) if torch.is_tensor(x) and x.ndim == 3 else torch.as_tensor(x).unsqueeze(0))
    if not xs:
        return {
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
            "blur_laplacian_var": 0.0,
            "noise_estimate": 0.0,
            "jpeg_blockiness": 0.0,
            "color_saturation": 0.0,
        }
    x = torch.cat(xs, dim=0).float()
    if x.min() < -0.5:
        mean = torch.tensor(TINY_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(TINY_STD).view(1, 3, 1, 1)
        x = (x * std + mean).clamp(0, 1)
    else:
        x = x.clamp(0, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    lap_kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(gray, lap_kernel, padding=1)
    dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
    dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean()
    block_h = torch.abs(gray[:, :, :, 8::8] - gray[:, :, :, 7::8]).mean() if gray.size(3) > 8 else torch.tensor(0.0)
    block_v = torch.abs(gray[:, :, 8::8, :] - gray[:, :, 7::8, :]).mean() if gray.size(2) > 8 else torch.tensor(0.0)
    maxc = x.max(dim=1).values
    minc = x.min(dim=1).values
    saturation = ((maxc - minc) / (maxc + 1e-6)).mean()
    return {
        "brightness_mean": float(gray.mean().item()),
        "contrast_std": float(gray.std().item()),
        "blur_laplacian_var": float(lap.var().item()),
        "noise_estimate": float(((dx + dy) / 2.0).item()),
        "jpeg_blockiness": float(((block_h + block_v) / 2.0).item()),
        "color_saturation": float(saturation.item()),
    }


def estimate_drift_type(features: dict[str, float]) -> tuple[str, int, float]:
    blur = features.get("blur_laplacian_var", 0.0)
    noise = features.get("noise_estimate", 0.0)
    brightness = features.get("brightness_mean", 0.5)
    contrast = features.get("contrast_std", 0.0)
    blockiness = features.get("jpeg_blockiness", 0.0)
    if blur < 0.01:
        return "blur", 4, 0.65
    if noise > 0.18:
        return "noise", min(5, max(1, int(noise * 20))), 0.62
    if blockiness > 0.08:
        return "jpeg_compression", min(5, max(1, int(blockiness * 40))), 0.58
    if brightness < 0.35 or brightness > 0.65:
        return "brightness", min(5, max(1, int(abs(brightness - 0.5) * 10) + 1)), 0.55
    if contrast < 0.12 or contrast > 0.28:
        return "contrast", min(5, max(1, int(abs(contrast - 0.2) * 20) + 1)), 0.52
    return "clean", 0, 0.50


class ReplayBuffer:
    def __init__(self, max_size: int = 200) -> None:
        self.buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=max_size)

    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().cpu()
        y = y.detach().cpu()
        for xi, yi in zip(x, y):
            self.buffer.append((xi, yi))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, batch_size)
        x, y = zip(*samples)
        return torch.stack(list(x)), torch.tensor([int(v) for v in y])

    def __len__(self) -> int:
        return len(self.buffer)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[tuple[torch.Tensor, int | torch.Tensor]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        if not torch.is_tensor(y):
            y = torch.tensor(int(y), dtype=torch.long)
        return x, y.long()


def load_pt_loader(path: str | Path, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    data = torch.load(path, map_location="cpu")
    return torch.utils.data.DataLoader(ListDataset(data), batch_size=batch_size, shuffle=shuffle)


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def seen_classes_for_task(global_task: int, num_classes_per_task: int, drift_mode: str) -> list[int]:
    if drift_mode == "data_only":
        return list(range(num_classes_per_task))
    return list(range(global_task * num_classes_per_task))
