"""
YOLO TensorRT wrapper for inference

Trined and exported with:

from ultralytics import YOLO

model_size = "s"

model = YOLO(f"yolo26{model_size}-seg.pt")

model.train(
    data="/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=24,
)

model.export(format="tensorrt", half=True)

"""

from pathlib import Path
from typing import Dict, List

import torch
from numpy.typing import NDArray
from ultralytics import YOLO


class YOLO_TRT_model:
    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        imgsz: int = 640,
        half: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.half = half
        self.model = YOLO(str(self.model_path))

    def __call__(self, img: NDArray) -> List[Dict[str, torch.Tensor]]:
        result = self.model(
            img,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
            retina_masks=True,
        )[0]

        # Handle empty detections
        if result.boxes is None or len(result.boxes) == 0:
            h, w = img.shape[:2]
            out = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "mask_probs": torch.zeros((0, h, w), dtype=torch.float32),
            }
        else:
            out = {
                "boxes": result.boxes.xyxy.to(torch.float32),
                "scores": result.boxes.conf.to(torch.float32),
                "labels": result.boxes.cls.to(torch.int64),
                "mask_probs": result.masks.data.to(torch.float32),
            }
        return [out]
