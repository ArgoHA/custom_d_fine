from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.utils import abs_xyxy_to_norm_xywh, vis_one_box
from src.infer.torch_model import Torch_model


def visualize(img, boxes, labels, scores, output_path, img_path):
    output_path.mkdir(parents=True, exist_ok=True)
    for box, label, score in zip(boxes, labels, scores):
        vis_one_box(img, box, label, mode="pred", score=score)

    cv2.imwrite((str(f"{output_path / Path(img_path).stem}.jpg")), img)


def save_yolo_annotations(res, output_path, img_path, img_shape):
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / Path(img_path).with_suffix(".txt"), "a") as f:
        for class_id, box in zip(res["labels"], res["boxes"]):
            norm_box = abs_xyxy_to_norm_xywh(box[None], img_shape[0], img_shape[1])[0]
            f.write(f"{int(class_id)} {norm_box[0]} {norm_box[1]} {norm_box[2]} {norm_box[3]}\n")


def run(torch_model, folder_path, output_path):
    batch = 0
    imag_paths = [img.name for img in folder_path.iterdir() if not str(img).startswith(".")]
    labels = set()
    for img_path in tqdm(imag_paths):
        img = cv2.imread(str(folder_path / img_path))
        res = torch_model(img)

        res = {
            "boxes": res[batch]["boxes"],
            "labels": res[batch]["labels"],
            "scores": res[batch]["scores"],
        }

        visualize(img, res["boxes"], res["labels"], res["scores"], output_path / "images", img_path)

        for class_id in res["labels"]:
            labels.add(class_id)
            save_yolo_annotations(res, output_path / "labels", img_path, img.shape)

    with open(output_path / "labels.txt", "w") as f:
        for class_id in labels:
            f.write(f"{int(class_id)}\n")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
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

    folder_path = Path(cfg.train.path_to_test_data)
    output_path = Path(cfg.train.root) / "output" / "infer"

    run(torch_model, folder_path, output_path)


if __name__ == "__main__":
    main()
