import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tabulate import tabulate
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

    res = {"bs": [], "throughput": [], "latency_per_image": []}
    images = 512
    bss = [1, 2, 4, 8, 16, 32]

    for bs in bss:
        if bs > 1:
            imgs = np.repeat(img[None, :, :, :], bs, axis=0)
        else:
            imgs = img

        t0 = time.perf_counter()
        for _ in tqdm(range(images // bs), desc=f"Batch size {bs}"):
            _ = torch_model(imgs)
        t1 = time.perf_counter()

        latency_per_image = (t1 - t0) * 1000 / images
        throughput = images / (t1 - t0)

        res["bs"].append(bs)
        res["latency_per_image"].append(latency_per_image)
        res["throughput"].append(throughput)

    df = pd.DataFrame(res)
    df.to_csv(Path(cfg.train.path_to_save) / "batched_infer.csv", index=False)

    tabulated_data = tabulate(df.round(1), headers="keys", tablefmt="pretty", showindex=False)
    print("\n" + tabulated_data)


if __name__ == "__main__":
    main()
