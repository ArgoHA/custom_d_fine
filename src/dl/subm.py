from pathlib import Path

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.utils import abs_xyxy_to_norm_xywh
from src.infer.torch_model import Torch_model


def visualize(img, pred_boxes, output_path, img_path):
    output_path.mkdir(exist_ok=True, parents=True)
    for pred_box in pred_boxes:
        cv2.rectangle(
            img,
            (int(pred_box[0]), int(pred_box[1])),
            (int(pred_box[2]), int(pred_box[3])),
            (255, 0, 0),
            2,
        )

    cv2.imwrite((str(f"{output_path / Path(img_path).stem}.jpg")), img)


def run(model, folder_path):
    df = pd.DataFrame(
        {
            "id": [],
            "ImageID": [],
            "PredictionString_pred": [],
        }
    )
    img_paths = [img.name for img in folder_path.iterdir() if not str(img).startswith(".")]

    idx = 0
    for img_path in tqdm(img_paths):
        tmp_res = {}
        img = cv2.imread(str(folder_path / img_path))
        res = model(img)

        visualize(img, res["boxes"], folder_path.parent / "visualize", img_path)

        tmp_res["id"] = idx
        tmp_res["ImageID"] = str(Path(img_path).stem)
        boxes = abs_xyxy_to_norm_xywh(res["boxes"], img.shape[0], img.shape[1])

        tmp_pred = []
        for i, (class_id, box) in enumerate(zip(res["labels"], boxes)):
            x_center, y_center, width, height = box
            x1 = float(x_center) - (float(width) / 2)
            y1 = float(y_center) - (float(height) / 2)
            width = float(width)
            height = float(height)
            tmp_pred.append(f"{(class_id)} {res['scores'][i]} {x1} {y1} {width} {height}")
        if len(tmp_pred) == 0:
            tmp_pred = ["0 0 0 0 0 0"]
        tmp_res["PredictionString_pred"] = ",".join(tmp_pred)

        df = pd.concat([df, pd.DataFrame(tmp_res, index=[0])], ignore_index=True)
        idx += 1

    df["id"] = df["id"].astype(int)
    return df


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    conf_thresh = 0.55
    iou_thresh = 0.6

    model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        rect=cfg.export.dynamic_input,
        keep_ratio=cfg.train.keep_ratio,
        # device="cpu",
    )

    csv = run(model, Path(cfg.train.path_to_test_data))

    sample_csv = pd.read_csv(
        Path(cfg.train.data_path).parent / "dlenigma1" / "sample_submission.csv"
    )
    df_merged = pd.merge(
        sample_csv[["id", "ImageID"]],
        csv[["ImageID", "PredictionString_pred"]],
        on="ImageID",
        how="left",
    )
    df_merged.to_csv(Path(cfg.train.data_path).parent / "custom_d_fine_base.csv", index=False)


if __name__ == "__main__":
    main()

# python -m src.dl.subm
