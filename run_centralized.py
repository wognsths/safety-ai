#!/usr/bin/env python3
"""Simple centralized training runner."""
import argparse
from pathlib import Path
import logging
import sys

from omegaconf import OmegaConf
from train.centralized import run_centralized_training


def main() -> None:
    p = argparse.ArgumentParser(description="Centralized training")
    p.add_argument("--config", "-c", type=Path, required=True, help="YAML config")
    p.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    p.add_argument("--log-file", type=Path, default=None, help="로그 저장 경로")
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    log = logging.getLogger("run_centralized")

    if not args.config.exists():
        raise FileNotFoundError(args.config)
    cfg = OmegaConf.load(args.config)
    log.info("Starting centralized training")
    run_centralized_training(cfg)


if __name__ == "__main__":
    main()
