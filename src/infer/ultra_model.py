import numpy as np
import torch
from ultralytics import YOLO


class UltraModel:
    def __init__(self, model_path, img_size=640, conf_thresh=0.5, iou_thresh=0.5, device=None):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = YOLO(model_path)

    def __call__(self, img):
        """
        Returns predictions as a Tuple of torch.tensors
        - boxes (torch.tensor): Tensor of shape (N, 4) containing bounding boxes in [x1, y1, x2, y2] format.
        - scores (torch.tensor): Tensor of shape (N,) containing confidence scores for each box.
        - classes (torch.tensor): Tensor of shape (N,) containing class ids for each box.
        """
        results = self.model.predict(
            img, device=self.device, conf=self.conf_thresh, iou=self.iou_thresh, imgsz=self.img_size
        )
        predictions = []

        for result in results:
            boxes_obj = result.boxes  # Contains the detections for this image
            # Check if there are any detections
            if len(boxes_obj) == 0:
                predictions.append(
                    {
                        "labels": np.array([], dtype=np.int64),
                        "boxes": np.empty((0, 4), dtype=np.float32),
                        "scores": np.array([], dtype=np.float32),
                    }
                )
                continue

            # Extract the tensors for boxes, confidence scores, and class indices.
            boxes = boxes_obj.xyxy  # shape: (N, 4)
            scores = boxes_obj.conf  # shape: (N,)
            labels = boxes_obj.cls  # shape: (N,)

            # Convert to NumPy arrays; if running on GPU these are torch.Tensors.
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
            else:
                boxes = boxes.numpy()
                scores = scores.numpy()
                labels = labels.numpy()

            # Cast to the specified dtypes.
            predictions.append(
                {
                    "labels": labels.astype(np.int64),
                    "boxes": boxes.astype(np.float32),
                    "scores": scores.astype(np.float32),
                }
            )

        return predictions
