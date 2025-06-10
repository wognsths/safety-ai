#!/usr/bin/env python3
"""Simple centralized training runner."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from train.centralized import run_centralized_training


def main() -> None:
    p = argparse.ArgumentParser(description="Centralized training")
    p.add_argument("--config", "-c", type=Path, required=True, help="YAML config")
    args = p.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(args.config)
    cfg = OmegaConf.load(args.config)
    run_centralized_training(cfg)


if __name__ == "__main__":
    main()
