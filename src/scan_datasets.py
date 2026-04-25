from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "Unsupervised-Classification"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from data.cifar import CIFAR10, CIFAR20  # type: ignore  # noqa: E402
from data.stl import STL10  # type: ignore  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train_dataset: Dataset
    eval_dataset: Dataset
    num_classes: int
    input_size: int
    paper_name: str


class HuggingFaceImageDataset(Dataset):
    """Torch wrapper around a Hugging Face image dataset split."""

    def __init__(self, hf_dataset, transform: Optional[Callable] = None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        raw_labels = list(hf_dataset["label"])
        unique_labels = sorted(set(raw_labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.targets = [self.label_to_index[label] for label in raw_labels]
        self.classes = [str(label) for label in unique_labels]

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        row = self.hf_dataset[index]
        image = row["image"].convert("RGB")
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": target,
            "meta": {"index": index, "class_name": self.classes[target]},
        }


def get_eval_transform(dataset_name: str):
    if dataset_name in {"cifar-10", "cifar-20"}:
        mean_std = {
            "cifar-10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            "cifar-20": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        }[dataset_name]
        mean, std = mean_std
        return transforms.Compose(
            [
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    if dataset_name == "stl-10":
        return transforms.Compose(
            [
                transforms.CenterCrop(96),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    if dataset_name == "imagenet-10":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    raise ValueError(f"Unsupported dataset {dataset_name}")


def build_dataset_bundle(name: str, imagenet10_eval_ratio: float = 0.2, seed: int = 42) -> DatasetBundle:
    transform = get_eval_transform(name)

    if name == "cifar-10":
        train_dataset = CIFAR10(root=str(DATA_ROOT), train=True, transform=transform, download=True)
        eval_dataset = CIFAR10(root=str(DATA_ROOT), train=False, transform=transform, download=True)
        return DatasetBundle(name, train_dataset, eval_dataset, 10, 32, "CIFAR-10")

    if name == "cifar-20":
        train_dataset = CIFAR20(root=str(DATA_ROOT), train=True, transform=transform, download=True)
        eval_dataset = CIFAR20(root=str(DATA_ROOT), train=False, transform=transform, download=True)
        return DatasetBundle(name, train_dataset, eval_dataset, 20, 32, "CIFAR-100 (20 superclasses)")

    if name == "stl-10":
        train_dataset = STL10(root=str(DATA_ROOT), split="train", transform=transform, download=True)
        eval_dataset = STL10(root=str(DATA_ROOT), split="test", transform=transform, download=True)
        return DatasetBundle(name, train_dataset, eval_dataset, 10, 96, "STL-10")

    if name == "imagenet-10":
        hf_dataset = load_dataset("JamieSJS/imagenet-10", split="test")
        labels = np.asarray(hf_dataset["label"])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=imagenet10_eval_ratio, random_state=seed)
        train_idx, eval_idx = next(splitter.split(np.zeros(len(labels)), labels))
        train_dataset = HuggingFaceImageDataset(hf_dataset.select(train_idx.tolist()), transform=transform)
        eval_dataset = HuggingFaceImageDataset(hf_dataset.select(eval_idx.tolist()), transform=transform)
        return DatasetBundle(name, train_dataset, eval_dataset, 10, 224, "ImageNet-10")

    raise ValueError(f"Unsupported dataset bundle {name}")
