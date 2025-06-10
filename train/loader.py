"""
loader.py

Utility helpers for building DataLoader objects from pre-computed split JSON.

* Supports custom 9-class ImageFolder dataset under `data/train/raw`
* Falls back to CIFAR-10 if `dataset_name`(model name) starts with "cifar"
* Automatically selects input resolution:
    - EfficientNet-B4 → 380×380
    - others          → 224×224
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10


# ──────────────────────────────────────────────────────────────
# Helper: infer proper image size from model / dataset name
# ──────────────────────────────────────────────────────────────

def _infer_img_size(name: str) -> int:
    """Return input resolution based on model/dataset identifier."""
    name = name.lower()
    if "efficientnet_b4" in name or "efficientnet-b4" in name:
        return 380
    return 224


# ──────────────────────────────────────────────────────────────
# Dataset-specific transform builders
# ──────────────────────────────────────────────────────────────

def _get_transform(train: bool = True, img_size: int = 224):
    """Return torchvision transforms for train / test phases."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def _load_dataset(name: str, root: str | Path, *, train: bool = True, img_size: int = 224):
    """Instantiate a torchvision dataset based on name."""
    root = Path(root)
    name_lc = name.lower()

    if name_lc == "custom9":
        if not root.exists():
            raise FileNotFoundError(f"Dataset root path does not exist: {root}")
        return ImageFolder(root=root, transform=_get_transform(train, img_size))

    if name_lc.startswith("cifar"):
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        return CIFAR10(root=root,
                       train=train,
                       download=True,
                       transform=_get_transform(train, img_size))

    raise ValueError(f"Unsupported dataset identifier '{name}'. Supported: ['custom9', 'cifar10']")


# ──────────────────────────────────────────────────────────────
# Public: DataLoader builder from client split indices
# ──────────────────────────────────────────────────────────────

def get_dataloaders_from_split(
    *,
    client_id: int,
    split_indices: List[int],
    data_root: str | Path,
    batch_size: int,
    dataset_name: str = "custom9",
) -> Tuple[DataLoader, DataLoader]:
    """
    Return train & test DataLoader for a given client.

    Parameters
    ----------
    client_id : int
        Numeric client ID (for logging/hashing only).
    split_indices : list[int]
        Index list belonging to this client (train split).
    data_root : str | Path
        Root directory of *train* images (ImageFolder).
    batch_size : int
        Batch size for loaders.
    dataset_name : str, optional
        Identifier for dataset or (preferred) model name; default "custom9".

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        train_loader, test_loader
    """
    img_size = _infer_img_size(dataset_name)

    # 1) Train subset specific to this client
    full_train_ds = _load_dataset(dataset_name, data_root, train=True, img_size=img_size)
    train_subset = Subset(full_train_ds, split_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # 2) Shared test/validation set
    #    Priority: data/test/** → data/val/** → 10 % random of client's train
    data_root = Path(data_root)
    test_root = data_root.parent / "test"   # e.g. data/train/raw -> data/test
    if not test_root.exists():
        test_root = data_root.parent / "val"

    if test_root.exists():
        test_ds = ImageFolder(root=test_root, transform=_get_transform(False, img_size))
    else:
        # Fallback: first 10 % of this client's split (quick sanity check)
        val_len = max(1, int(0.1 * len(split_indices)))
        val_indices = split_indices[:val_len]
        test_ds = Subset(full_train_ds, val_indices)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
