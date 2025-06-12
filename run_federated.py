#!/usr/bin/env python3
"""
Federated Learning 실행 스크립트

사용법:
    python run_federated.py --config config/fl/fedavg.yaml
    python run_federated.py --config config/fl/fedbn.yaml  
    python run_federated.py --config config/fl/fedprox.yaml

필수 전제조건:
    1. 데이터셋 스플릿이 미리 생성되어 있어야 함:
       python scripts/dataset_split.py --split config/split/dirichlet_alpha5.yaml
    
    2. 훈련 데이터가 data/train/raw에 준비되어 있어야 함
    3. 테스트 데이터가 data/test에 준비되어 있어야 함
"""

import argparse
from pathlib import Path
import logging
import sys

import matplotlib.pyplot as plt
import csv

from omegaconf import OmegaConf
from train.federated import run_federated_training


def save_history(history, out_dir: Path) -> None:
    """Save FL history to CSV and PNG plot."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = []
    acc = []
    loss = []

    if getattr(history, "metrics_centralized", None):
        acc = [v for _, v in history.metrics_centralized.get("accuracy", [])]
    if getattr(history, "losses_centralized", None):
        loss = [v for _, v in history.losses_centralized]
        rounds = [r for r, _ in history.losses_centralized]

    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "accuracy"])
        for i, r in enumerate(rounds):
            a = acc[i] if i < len(acc) else ""
            l = loss[i] if i < len(loss) else ""
            writer.writerow([r, l, a])

    if rounds:
        plt.figure()
        if loss:
            plt.plot(rounds, loss, label="loss")
        if acc:
            plt.plot(rounds[: len(acc)], acc, label="accuracy")
        plt.xlabel("Round")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "history.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Federated Learning 훈련 실행")
    parser.add_argument(
        "--config", 
        "-c",
        type=Path,
        required=True,
        help="FL 설정 YAML 파일 경로 (예: config/fl/fedavg.yaml)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="로그를 저장할 파일 경로"
    )

    args = parser.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    log = logging.getLogger("run_federated")

    # 설정 파일 로드
    if not args.config.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
    
    cfg = OmegaConf.load(args.config)
    
    # 필수 설정 검증
    required_keys = ["model", "train", "fl", "dataset"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"설정 파일에서 필수 섹션 '{key}'를 찾을 수 없습니다")
    
    # 데이터셋 스플릿 파일 존재 확인
    split_path = Path(cfg.dataset.split_path)
    if not split_path.exists():
        log.warning(f"데이터셋 스플릿 파일이 없습니다: {split_path}")
        return
    log.info(f"   스플릿: {split_path}")
    
    # 데이터 디렉터리 존재 확인  
    data_root = Path(cfg.dataset.root)
    if not data_root.exists():
        log.warning(f"훈련 데이터 디렉터리가 없습니다: {data_root}")
        log.warning("data/train/raw 디렉터리에 훈련 데이터를 준비하세요")
        return
    
    log.info("🚀 Federated Learning 시작")
    log.info(f"   전략: {cfg.train.strategy.upper()}")
    log.info(f"   모델: {cfg.model.name}")
    log.info(f"   라운드: {cfg.train.rounds}")
    log.info(f"   클라이언트: {cfg.fl.min_available_clients}")
    log.info(f"   스플릿: {split_path}")
    
    # FL 훈련 실행
    try:
        history = run_federated_training(cfg)
        log_dir = Path("results") / f"fl_{cfg.train.strategy}"
        save_history(history, log_dir)
        log.info("✅ Federated Learning 완료!")
    except Exception as e:
        log.error(f"❌ 훈련 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 
