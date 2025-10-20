import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.utils import get_latest_experiment_name
from src.infer.torch_model import Torch_model


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
    )

    img_folder = Path(cfg.train.data_path) / "images"
    img = cv2.imread(str(img_folder.iterdir().__next__()))

    res = {"bs": [], "full_latency": []}
    runs = 512
    bss = [1, 2, 4, 8, 16, 32]

    for bs in bss:
        if bs > 1:
            imgs = np.repeat(img[None, :, :, :], bs, axis=0)
        else:
            imgs = img

        t0 = time.perf_counter()
        for _ in tqdm(range(runs // bs)):
            _ = torch_model(imgs)
        t1 = time.perf_counter()

        fill_infer = (t1 - t0) * 1000 / runs

        logger.info(f"BS {bs} Full inference {fill_infer:.2f}ms")
        print()
        res["bs"].append(bs)
        res["full_latency"].append(fill_infer)

    df = pd.DataFrame(res)
    df.to_csv(Path(cfg.train.path_to_save) / "batched_infer.csv", index=False)


if __name__ == "__main__":
    main()
