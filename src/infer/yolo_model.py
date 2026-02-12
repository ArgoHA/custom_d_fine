from pathlib import Path
from typing import Dict, List

import torch
from numpy.typing import NDArray
from ultralytics import YOLO


class YOLO_model:
    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        imgsz: int = 640,
        half: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.half = half
        self.model = YOLO(str(self.model_path))

    def __call__(
        self, img: NDArray, return_raw_latency: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        result = self.model(
            img,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
            retina_masks=True,
        )[0]

        # Extract raw inference time from YOLO result (in ms)
        inference_time_ms = result.speed.get("inference", 0.0) if result.speed else 0.0

        # Handle empty detections
        if result.boxes is None or len(result.boxes) == 0:
            h, w = img.shape[:2]
            out = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.float32),
            }
        else:
            out = {
                "boxes": result.boxes.xyxy.to(torch.float32),
                "scores": result.boxes.conf.to(torch.float32),
                "labels": result.boxes.cls.to(torch.int64),
            }
            if result.masks is not None:
                out["masks"] = result.masks.data.to(torch.float32)

        if return_raw_latency:
            return [out], inference_time_ms
        return [out]
