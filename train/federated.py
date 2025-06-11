# train/federated.py
"""Federated training entry‑point.

* Supports GPU automatically (uses first CUDA device if available)
* Handles model list (e.g. ResNet50 & EfficientNet‑B4) via outer loop (run_federated_training)
* Designed for FedBN but strategy is selected via `get_strategy(cfg)`

Dependencies
------------
* utils.loader.get_dataloaders_from_split
* utils.evaluation (optional)
* models.init_net
* strategies.get_strategy (must include FedBN strategy that ignores BN params)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import flwr as fl
import torch
from flwr.client import NumPyClient
from flwr.common import Context
from omegaconf import DictConfig

from train.models import init_net
from train.strategies import get_strategy
from train.loader import get_dataloaders_from_split

# ──────────────────────────────────────────────────────────────
# Global device configuration
# ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# Flower Client Class
# ──────────────────────────────────────────────────────────────
class FederatedClient(NumPyClient):
    """Single FL client conforming to Flower NumPyClient API."""

    def __init__(self, model: torch.nn.Module, train_loader, test_loader, cfg: DictConfig):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        # FedBN 지원을 위한 로컬 BN 파라미터 저장
        self._local_bn_params = {}

    # Flower API ------------------------------------------------------------------
    def get_parameters(self, config: Dict | None):  # noqa: D401
        """Return model parameters as a list of NumPy ndarrays."""
        return [t.detach().cpu().numpy() for t in self.model.state_dict().values()]

    def set_parameters(self, parameters: List):
        """Set model parameters, preserving BN parameters for FedBN strategy."""
        state_dict = self.model.state_dict()
        param_names = list(state_dict.keys())
        
        # FedBN인 경우 BN 파라미터는 로컬 값 유지
        is_fedbn = self.cfg.train.strategy.lower() == "fedbn"
        
        for i, (param_name, param_tensor) in enumerate(zip(param_names, parameters)):
            if is_fedbn and self._is_bn_param(param_name):
                # FedBN: BN 파라미터는 로컬 값 유지 (서버 값 무시)
                if param_name in self._local_bn_params:
                    state_dict[param_name] = self._local_bn_params[param_name]
                # 아니면 현재 로컬 값 그대로 유지
            else:
                # 비-BN 파라미터는 서버 값 적용
                state_dict[param_name] = torch.tensor(param_tensor, device=self.device)
        
        self.model.load_state_dict(state_dict, strict=True)
        
        # FedBN인 경우 현재 BN 파라미터를 로컬 저장소에 백업
        if is_fedbn:
            for param_name, param_value in state_dict.items():
                if self._is_bn_param(param_name):
                    self._local_bn_params[param_name] = param_value.clone()

    def _is_bn_param(self, name: str) -> bool:
        """Check if parameter name corresponds to BatchNorm parameter."""
        return (
            ".running_mean" in name
            or ".running_var" in name
            or ".num_batches_tracked" in name
        )

    def fit(self, parameters, config):  # noqa: D401
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        criterion = torch.nn.CrossEntropyLoss()
        global_params = [p.detach().clone() for p in self.model.parameters()]
        mu = float(config.get("mu", 0.0))

        for _ in range(self.cfg.train.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                if mu > 0:
                    prox = 0.0
                    for w, w0 in zip(self.model.parameters(), global_params):
                        prox += (w - w0).pow(2).sum()
                    loss += 0.5 * mu * prox
                loss.backward()
                optimizer.step()

        # FedBN: 훈련 후 BN 파라미터 로컬 저장소 업데이트
        if self.cfg.train.strategy.lower() == "fedbn":
            for param_name, param_value in self.model.state_dict().items():
                if self._is_bn_param(param_name):
                    self._local_bn_params[param_name] = param_value.clone()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):  # noqa: D401
        self.set_parameters(parameters)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return loss_sum / total, total, {"accuracy": correct / total}


# ──────────────────────────────────────────────────────────────
# Simulation launcher
# ──────────────────────────────────────────────────────────────

def run_federated_training(cfg: DictConfig) -> None:
    """Launch a Flower simulation based on YAML/OmegaConf config."""

    # 1. Load split indices ----------------------------------------------------------------
    split_path = Path(cfg.dataset.split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as fp:
        split_data = json.load(fp)
    client_splits = split_data["splits"]

    data_root = Path(cfg.dataset.root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    # 2. Flower client factory --------------------------------------------------------------
    def client_fn(context: Context | str):
        """Create a single federated client.

        Supports both new (``Context``) and old (``cid`` string) signatures for
        backward compatibility.
        """
        try:
            cid = context.cid  # type: ignore[attr-defined]
        except AttributeError:
            cid = context

        client_id = int(cid)
        try:
            train_loader, test_loader = get_dataloaders_from_split(
                client_id=client_id,
                split_indices=client_splits[f"client_{client_id}"],
                data_root=data_root,
                batch_size=cfg.train.batch_size,
                dataset_name=cfg.dataset.name,
            )
            model = init_net(cfg.model.name, cfg.model.output_dim)
            numpy_client = FederatedClient(model, train_loader, test_loader, cfg)
            return numpy_client.to_client()
        except Exception as err:
            print(f"[client_fn] Error building client {cid}: {err}")
            raise

    # 3. Strategy (FedBN, FedAvg, etc.) -----------------------------------------------------
    strategy = get_strategy(cfg)

    # 4. Launch simulation ------------------------------------------------------------------
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.fl.min_available_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.train.rounds),
        strategy=strategy,
    )
    return history
