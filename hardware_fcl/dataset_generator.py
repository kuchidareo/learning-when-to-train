from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from hardware_fcl.utils import set_seed


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


def apply_corruption_pil(img: Image.Image, corruption_type: str, severity: int) -> Image.Image:
    if corruption_type == "clean" or severity <= 0:
        return img.convert("RGB")
    img = img.convert("RGB")
    severity = int(max(1, min(5, severity)))
    if corruption_type == "gaussian_noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        std = [0.04, 0.08, 0.12, 0.16, 0.20][severity - 1]
        arr = np.clip(arr + np.random.normal(0.0, std, arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if corruption_type == "shot_noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        scale = [80, 50, 35, 20, 12][severity - 1]
        arr = np.clip(np.random.poisson(arr * scale) / float(scale), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if corruption_type == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=[0.5, 1.0, 1.5, 2.0, 3.0][severity - 1]))
    if corruption_type == "motion_blur":
        radius = [3, 5, 7, 9, 11][severity - 1]
        kernel = np.zeros((radius, radius), dtype=np.float32)
        kernel[radius // 2, :] = 1.0 / radius
        try:
            from scipy.ndimage import convolve

            arr = np.asarray(img).astype(np.float32)
            arr = np.stack([convolve(arr[:, :, c], kernel, mode="nearest") for c in range(3)], axis=2)
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        except Exception:
            return img.filter(ImageFilter.GaussianBlur(radius=max(1.0, radius / 4.0)))
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
        quality = [70, 50, 35, 20, 10][severity - 1]
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
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
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y)


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset) -> None:
        self.dataset = hf_dataset
        labels = list(hf_dataset["label"])
        self.valid_indices = [idx for idx, label in enumerate(labels) if int(label) >= 0]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        row = self.dataset[self.valid_indices[idx]]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        return img.convert("RGB"), int(row["label"])


def build_class_indices(dataset) -> dict[int, list[int]]:
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        class_to_indices[int(y)].append(idx)
    return dict(class_to_indices)


def dirichlet_split_class_indices(class_indices, client_ids, alpha=0.1) -> dict[int, list[int]]:
    shuffled = list(class_indices)
    random.shuffle(shuffled)
    proportions = np.random.dirichlet(alpha=np.ones(len(client_ids)) * alpha)
    counts = np.floor(proportions * len(shuffled)).astype(int)
    while counts.sum() < len(shuffled):
        counts[np.random.randint(0, len(client_ids))] += 1
    while counts.sum() > len(shuffled):
        j = int(np.argmax(counts))
        counts[j] -= 1
    out = {}
    start = 0
    for cid, count in zip(client_ids, counts):
        out[int(cid)] = shuffled[start : start + int(count)]
        start += int(count)
    return out


def make_all_class_client_splits(class_to_indices, client_ids, alpha) -> dict[int, dict[int, list[int]]]:
    return {
        int(cls): dirichlet_split_class_indices(indices, client_ids=client_ids, alpha=alpha)
        for cls, indices in class_to_indices.items()
    }


def make_task_classes(num_classes_per_task=20, num_tasks=5, drift_mode="data_and_concept") -> dict[int, list[int]]:
    if drift_mode == "data_only":
        return {task_id + 1: list(range(num_classes_per_task)) for task_id in range(num_tasks)}
    return {
        task_id + 1: list(range(task_id * num_classes_per_task, (task_id + 1) * num_classes_per_task))
        for task_id in range(num_tasks)
    }


def make_async_client_schedule(num_clients, num_global_tasks, drift_mode, lag_prob=0.35, lead_prob=0.10):
    schedule = {}
    for cid in range(num_clients):
        client_schedule = {}
        for global_task in range(1, num_global_tasks + 1):
            if drift_mode == "data_only":
                local_task = 1
            else:
                local_task = global_task
                r = random.random()
                if r < lag_prob:
                    local_task = max(1, global_task - 1)
                elif r < lag_prob + lead_prob:
                    local_task = min(num_global_tasks, global_task + 1)
            if drift_mode == "concept_only":
                corruption_type, severity = "clean", 0
            else:
                corruption_type = random.choice(CORRUPTIONS)
                severity = 0 if corruption_type == "clean" else random.randint(1, 5)
            client_schedule[str(global_task)] = {
                "local_task": int(local_task),
                "corruption_type": corruption_type,
                "severity": int(severity),
            }
        schedule[str(cid)] = client_schedule
    return schedule


def _materialize(dataset: Dataset) -> list[tuple[torch.Tensor, int]]:
    return [dataset[i] for i in range(len(dataset))]


def save_client_task_dataset(
    output_root,
    cid,
    global_task,
    local_task,
    class_ids,
    dataset,
    indices,
    corruption_type,
    severity,
    transform,
    report_mode,
    val_ratio=0.2,
):
    client_dir = Path(output_root) / f"client_{cid:03d}" / f"task_{global_task:03d}"
    client_dir.mkdir(parents=True, exist_ok=True)
    indices = list(indices)
    random.shuffle(indices)
    n_val = int(len(indices) * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_data = _materialize(CorruptedSubset(dataset, train_indices, corruption_type, severity, transform))
    val_data = _materialize(CorruptedSubset(dataset, val_indices, corruption_type, severity, transform))
    torch.save(train_data, client_dir / "train.pt")
    torch.save(val_data, client_dir / "val.pt")
    label_hist = Counter(int(y) for _, y in train_data)
    metadata = {
        "cid": int(cid),
        "global_task": int(global_task),
        "local_task": int(local_task),
        "class_ids": [int(x) for x in class_ids],
        "num_samples": len(train_data),
        "label_histogram": {str(k): int(v) for k, v in sorted(label_hist.items())},
        "data_drift": {"corruption_type": corruption_type, "severity": int(severity)},
        "report_mode": report_mode,
    }
    with open(client_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_global_tests(output_root, base_dataset, class_to_indices, task_classes, num_global_tasks, transform, val_ratio=0.2):
    out = Path(output_root) / "global_test"
    out.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    for task in range(1, num_global_tasks + 1):
        task_indices = []
        for cls in task_classes[task]:
            cls_indices = class_to_indices.get(cls, [])
            take = max(1, int(len(cls_indices) * val_ratio))
            task_indices.extend(cls_indices[:take])
        task_data = _materialize(CorruptedSubset(base_dataset, task_indices, "clean", 0, transform))
        torch.save(task_data, out / f"task_{task:03d}_classes.pt")
        seen.update(task_classes[task])
        indices = []
        for cls in sorted(seen):
            cls_indices = class_to_indices.get(cls, [])
            take = max(1, int(len(cls_indices) * val_ratio))
            indices.extend(cls_indices[:take])
        data = _materialize(CorruptedSubset(base_dataset, indices, "clean", 0, transform))
        torch.save(data, out / f"task_{task:03d}_seen_classes.pt")


def load_base_dataset(hf_dataset: str, hf_split: str):
    from datasets import load_dataset

    dataset = load_dataset(hf_dataset, split=hf_split)
    return HFDatasetWrapper(dataset), 200, hf_dataset


def generate_datasets(
    output_root,
    hf_dataset="zh-plus/tiny-imagenet",
    num_clients=20,
    num_global_tasks=5,
    num_classes_per_task=20,
    alpha=0.1,
    seed=42,
    drift_mode="data_and_concept",
    report_mode="oracle",
    hf_split="train",
    overwrite=False,
):
    set_seed(seed)
    output_root = Path(output_root)
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_train, num_classes, dataset_name = load_base_dataset(hf_dataset, hf_split)
    if num_global_tasks * num_classes_per_task > num_classes and drift_mode != "data_only":
        raise ValueError("num_global_tasks * num_classes_per_task exceeds dataset classes")
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
        ]
    )
    task_classes = make_task_classes(num_classes_per_task, num_global_tasks, drift_mode)
    class_to_indices = build_class_indices(base_train)
    client_ids = list(range(num_clients))
    splits = make_all_class_client_splits(class_to_indices, client_ids, alpha)
    schedule = make_async_client_schedule(num_clients, num_global_tasks, drift_mode)
    for cid in client_ids:
        client_root = output_root / f"client_{cid:03d}"
        client_root.mkdir(parents=True, exist_ok=True)
        with open(client_root / "schedule.json", "w", encoding="utf-8") as f:
            json.dump(schedule[str(cid)], f, indent=2)
    for global_task in range(1, num_global_tasks + 1):
        for cid in client_ids:
            item = schedule[str(cid)][str(global_task)]
            local_task = int(item["local_task"])
            class_ids = task_classes[local_task]
            selected_indices = []
            for cls in class_ids:
                selected_indices.extend(splits[int(cls)].get(cid, []))
            save_client_task_dataset(
                output_root=output_root,
                cid=cid,
                global_task=global_task,
                local_task=local_task,
                class_ids=class_ids,
                dataset=base_train,
                indices=selected_indices,
                corruption_type=item["corruption_type"],
                severity=int(item["severity"]),
                transform=transform,
                report_mode=report_mode,
            )
    save_global_tests(output_root, base_train, class_to_indices, task_classes, num_global_tasks, transform)
    with open(output_root / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "hf_dataset": hf_dataset,
                "hf_split": hf_split,
                "num_clients": num_clients,
                "num_global_tasks": num_global_tasks,
                "num_classes_per_task": num_classes_per_task,
                "alpha": alpha,
                "seed": seed,
                "drift_mode": drift_mode,
                "report_mode": report_mode,
            },
            f,
            indent=2,
        )
    print(f"Generated client/task datasets under: {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate client/task datasets for FCL experiments.")
    parser.add_argument("--hf_dataset", default="zh-plus/tiny-imagenet")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--output_dir", default="./hardware_fcl/data/generated")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_global_tasks", type=int, default=5)
    parser.add_argument("--num_classes_per_task", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift_mode", choices=["data_only", "concept_only", "data_and_concept"], default="data_and_concept")
    parser.add_argument("--report_mode", choices=["oracle", "estimated"], default="oracle")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_datasets(
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        output_root=args.output_dir,
        num_clients=args.num_clients,
        num_global_tasks=args.num_global_tasks,
        num_classes_per_task=args.num_classes_per_task,
        alpha=args.alpha,
        seed=args.seed,
        drift_mode=args.drift_mode,
        report_mode=args.report_mode,
        overwrite=args.overwrite,
    )
