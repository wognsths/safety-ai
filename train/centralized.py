import csv
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from .models import init_net
from .loader import _infer_img_size, _get_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders(root: Path, batch_size: int, dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    img_size = _infer_img_size(dataset_name)
    train_ds = ImageFolder(root=root, transform=_get_transform(True, img_size))
    test_root = root.parent / "test"
    test_ds = ImageFolder(root=test_root, transform=_get_transform(False, img_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def _evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def run_centralized_training(cfg: DictConfig) -> None:
    train_loader, test_loader = _build_loaders(Path(cfg.dataset.root), cfg.train.batch_size, cfg.dataset.name)
    model = init_net(cfg.model.name, cfg.model.output_dim, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"loss": [], "acc": []}
    for epoch in range(cfg.train.epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = _evaluate(model, test_loader, criterion)
        history["loss"].append(val_loss)
        history["acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{cfg.train.epochs} - loss: {val_loss:.4f} acc: {val_acc:.4f}")

    out_dir = Path(cfg.save.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy"])
        for i, (l, a) in enumerate(zip(history["loss"], history["acc"])):
            writer.writerow([i + 1, l, a])
    plt.figure()
    plt.plot(range(1, len(history["loss"]) + 1), history["loss"], label="loss")
    plt.plot(range(1, len(history["acc"]) + 1), history["acc"], label="accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "history.png")
    plt.close()
