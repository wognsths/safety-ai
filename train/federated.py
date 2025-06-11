# train/federated.py
"""PyTorch × Flower 연합학습: GPU + tqdm 지원 버전.

- 가상 클라이언트마다 ray.get_gpu_ids()로 GPU 지정
- tqdm 진행률 표시 (배치·평가 루프)
- FedBN, FedProx 지원
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import flwr as fl
import torch
from flwr.client import NumPyClient
from flwr.common import Context
from omegaconf import DictConfig
import ray                           # GPU ID 확인
from tqdm.auto import tqdm           # auto: Jupyter/CLI 모두 대응 :contentReference[oaicite:5]{index=5}
import logging
from train.loader import get_dataloaders_from_split
from train.models import init_net
from train.strategies import get_strategy

# ────────────────────── 로깅 포맷 - tqdm 안전 ─────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("train")
tqdm_logging = lambda msg: tqdm.write(msg)  # tqdm 진행률 바 보존 :contentReference[oaicite:6]{index=6}

# ────────────────────── 전역 장치 (백업) ─────────────────────
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────── Flower 클라이언트 ─────────────────────
class FederatedClient(NumPyClient):
    """Flower NumPyClient with tqdm + GPU awareness."""
    def __init__(self, model, train_loader, test_loader, cfg: DictConfig):
        gpu_ids = ray.get_gpu_ids()                    # ex) [0] :contentReference[oaicite:7]{index=7}
        if gpu_ids:
            torch.cuda.set_device(int(gpu_ids[0]))
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.cfg = cfg
        self._local_bn_params: Dict[str, torch.Tensor] = {}

    # Flower 필수 메서드 ------------------------------------------------------
    def get_parameters(self, _config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params: List):
        state = self.model.state_dict()
        ism   = self.cfg.train.strategy.lower() == "fedbn"
        for name, tensor in zip(state.keys(), params):
            if ism and self._is_bn(name):        # FedBN → BN 로컬 유지 :contentReference[oaicite:8]{index=8}
                continue
            state[name] = torch.tensor(tensor, device=self.device)
        self.model.load_state_dict(state, strict=True)
        if ism:
            for n, p in state.items():
                if self._is_bn(n):
                    self._local_bn_params[n] = p.clone()

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        criterion = torch.nn.CrossEntropyLoss()
        mu = float(config.get("mu", 0.0))        # FedProx μ

        global_params = [p.detach().clone() for p in self.model.parameters()]
        epochs = self.cfg.train.local_epochs
        for ep in range(epochs):
            pbar = tqdm(self.train_loader, desc=f"Client{config.get('round',0)}-E{ep+1}/{epochs}", leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                optim.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                if mu > 0:
                    prox = sum((w - w0).pow(2).sum() for w, w0 in zip(self.model.parameters(), global_params))
                    loss += 0.5 * mu * prox
                loss.backward()
                optim.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Eval", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_sum += criterion(logits, y).item() * x.size(0)
                correct  += (logits.argmax(1) == y).sum().item()
                total    += y.size(0)
        return loss_sum / total, total, {"accuracy": correct / total}

    @staticmethod
    def _is_bn(name: str) -> bool:
        return any(k in name for k in (".running_mean", ".running_var", ".num_batches_tracked"))

# ────────────────────── 시뮬레이션 런처 ─────────────────────
def run_federated_training(cfg: DictConfig):
    split_path = Path(cfg.dataset.split_path)
    with split_path.open("r", encoding="utf-8") as f:
        client_splits = json.load(f)["splits"]
    data_root = Path(cfg.dataset.root)

    def client_fn(ctx: Context | str):
        part_id = int(ctx.node_config["partition-id"]) if isinstance(ctx, Context) else int(ctx)  # :contentReference[oaicite:9]{index=9}
        key = f"client_{part_id}"
        if key not in client_splits:
            raise KeyError(f"{key} 분할 없음")
        # 데이터로더
        train_loader, test_loader = get_dataloaders_from_split(
            client_id=part_id,
            split_indices=client_splits[key],
            data_root=data_root,
            batch_size=cfg.train.batch_size,
            dataset_name=cfg.dataset.name,
        )
        # 모델
        model = init_net(cfg.model.name, cfg.model.output_dim)
        return FederatedClient(model, train_loader, test_loader, cfg).to_client()

    strategy = get_strategy(cfg)

    log.info("Flower simulation starting …")
    fl.simulation.start_simulation(
        client_fn       = client_fn,
        num_clients     = len(client_splits),
        client_resources={"num_gpus": 1, "num_cpus": 1},
        config          = fl.server.ServerConfig(num_rounds=cfg.train.rounds),
        strategy        = strategy,
    )
