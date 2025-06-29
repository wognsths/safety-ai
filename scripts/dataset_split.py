import argparse
import json
import gzip
import pathlib
import sys
from collections import defaultdict

import numpy as np
import yaml
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


def dirichlet_split(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    label_indices = defaultdict(list)
    for idx, lab in enumerate(labels):
        label_indices[lab].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for lab, idxs in label_indices.items():
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (proportions.cumsum() * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cuts)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    for idx_list in client_indices:
        rng.shuffle(idx_list)
    return client_indices


def save_json(obj, path: pathlib.Path, compress: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        with gzip.open(path.with_suffix(".json.gz"), "wt", encoding="utf-8") as f:
            json.dump(obj, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)


def main(split_yaml: pathlib.Path, out_dir: pathlib.Path):
    cfg = yaml.safe_load(split_yaml.read_text(encoding="utf-8"))

    ds_name = cfg["dataset"]["name"]
    root = pathlib.Path(cfg["dataset"]["root"])
    sp_cfg = cfg["dataset"]["split"]
    n_clients = int(sp_cfg["num_clients"])
    seed = int(sp_cfg.get("seed", 42))

    # Load dataset from image folder (custom 9-class dataset)
    if ds_name.lower() == "custom9":
        transform = transforms.ToTensor()
        full_ds: Dataset = ImageFolder(root=root, transform=transform)
        labels = np.array([sample[1] for sample in full_ds.samples])
    else:
        sys.exit(f"Unsupported dataset {ds_name}")

    # Only Dirichlet is supported
    split_type = sp_cfg["type"]
    if split_type != "dirichlet":
        sys.exit("Only 'dirichlet' split is supported in this script.")

    alpha = float(sp_cfg["alpha"])
    indices = dirichlet_split(labels, n_clients, alpha, seed)

    # ── Compute per-client class distribution and entropy ──────────────────
    n_classes = len(np.unique(labels))
    class_counts = np.zeros((n_clients, n_classes), dtype=int)
    entropies = {}
    for i, idx_list in enumerate(indices):
        cls = labels[idx_list]
        counts = np.bincount(cls, minlength=n_classes)
        class_counts[i] = counts
        p = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        ent = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))
        entropies[f"client_{i}"] = float(ent)

    meta = {
        "dataset": ds_name,
        "num_clients": n_clients,
        "seed": seed,
        **{k: v for k, v in sp_cfg.items() if k != "type"},
        "entropy": entropies,
    }
    save_obj = {
        "meta": meta,
        "class_counts": {f"client_{i}": class_counts[i].tolist() for i in range(n_clients)},
        "splits": {f"client_{i}": idx for i, idx in enumerate(indices)},
    }

    # Plot class distribution -------------------------------------------------
    proportions = class_counts / class_counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(n_clients)
    for c in range(n_classes):
        ax.bar(
            np.arange(n_clients),
            proportions[:, c],
            bottom=bottom,
            label=f"class {c}",
        )
        bottom += proportions[:, c]
    ax.set_xlabel("Client")
    ax.set_ylabel("Proportion")
    ax.set_title("Dataset split distribution")
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    plot_path = out_dir / f"{split_yaml.stem}_dist.png"
    fig.savefig(plot_path)
    plt.close(fig)
    plot_path = plot_path.resolve()
    print(f"[✓] Distribution plot → {plot_path.relative_to(pathlib.Path.cwd())}")

    out_path = out_dir / (split_yaml.with_suffix("" ).name + ".json")
    save_json(save_obj, out_path)
    print(f"[✓] Split saved → {out_path.resolve().relative_to(pathlib.Path.cwd())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate dataset splits from YAML config")
    p.add_argument("--split", required=True, type=pathlib.Path, help="YAML file path")
    p.add_argument(
        "--out", default="data/split", type=pathlib.Path, help="Output directory"
    )
    args = p.parse_args()
    main(args.split, args.out)
