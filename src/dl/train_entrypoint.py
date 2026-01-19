#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main():
    config_path = Path("config.yaml")
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        ddp_enabled = cfg.get("train", {}).get("ddp", {}).get("enabled", False)
        num_gpus = cfg.get("train", {}).get("ddp", {}).get("n_gpus", 2)
    else:
        ddp_enabled = False
        num_gpus = 2

    if ddp_enabled:
        print(f"🚀 Training with DDP using {num_gpus} GPUs...")
        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "--master_port", "29500",
            "-m", "src.dl.train",
        ] + sys.argv[1:]
    else:
        print("🔧 Training with single GPU...")
        cmd = ["python", "-m", "src.dl.train"] + sys.argv[1:]

    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()

