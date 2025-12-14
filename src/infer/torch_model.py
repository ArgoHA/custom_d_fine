from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.typing import NDArray
from torchvision.ops import nms

from src.d_fine.dfine import build_model


class Torch_model:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_outputs: int,
        input_width: int = 640,
        input_height: int = 640,
        conf_thresh: float = 0.5,
        rect: bool = False,  # cuts paddings, inference is faster, accuracy might be lower
        half: bool = False,
        keep_ratio: bool = False,
        use_nms: bool = False,
        enable_mask_head: bool = False,
        device: str = None,
    ):
        self.input_size = (input_height, input_width)
        self.n_outputs = n_outputs
        self.model_name = model_name
        self.model_path = model_path
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.use_nms = use_nms
        self.enable_mask_head = enable_mask_head
        self.channels = 3
        self.debug_mode = False

        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        elif isinstance(conf_thresh, list):
            self.conf_threshs = conf_thresh

        if not device:
            self.device = "cpu"
            if torch.backends.mps.is_available():
                self.device = "mps"
            if torch.cuda.is_available():
                self.device = "cuda"
        else:
            self.device = device

        self.np_dtype = np.float32

        self._load_model()
        self._test_pred()

    def _load_model(self):
        self.model = build_model(
            self.model_name,
            self.n_outputs,
            self.enable_mask_head,
            self.device,
            img_size=None,
        )
        self.model.load_state_dict(
            torch.load(self.model_path, weights_only=True, map_location=torch.device("cpu")),
            strict=False,
        )
        self.model.eval()
        self.model.to(self.device)

        logger.info(f"Torch model, Device: {self.device}")

    def _test_pred(self) -> None:
        random_image = np.random.randint(0, 255, size=(1100, 1000, self.channels), dtype=np.uint8)
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(random_image)
        preds = self._predict(processed_inputs)
        self._postprocess(preds, processed_sizes, original_sizes)

    @staticmethod
    def process_boxes(boxes, processed_sizes, orig_sizes, keep_ratio, device):
        boxes = boxes.cpu().numpy()
        final_boxes = np.zeros_like(boxes)
        for idx, box in enumerate(boxes):
            final_boxes[idx] = norm_xywh_to_abs_xyxy(
                box, processed_sizes[idx][0], processed_sizes[idx][1]
            )

        for i in range(len(orig_sizes)):
            if keep_ratio:
                final_boxes[i] = scale_boxes_ratio_kept(
                    final_boxes[i], processed_sizes[i], orig_sizes[i]
                )
            else:
                final_boxes[i] = scale_boxes(final_boxes[i], orig_sizes[i], processed_sizes[i])
        return torch.tensor(final_boxes).to(device)

    @staticmethod
    def process_masks(
        pred_masks,  # Tensor [B, Q, Hm, Wm] or [Q, Hm, Wm]
        processed_size,  # (H, W) of network input (after your A.Compose)
        orig_sizes,  # Tensor [B, 2] (H, W)
        keep_ratio: bool,
    ) -> List[torch.Tensor]:
        """
        Returns list of length B with masks resized to original image sizes:
        Each item: Float Tensor [Q, H_orig, W_orig] in [0,1] (no thresholding here).
        - Handles letterbox padding removal if keep_ratio=True.
        - Works for both batched and single-image inputs.
        """
        single = pred_masks.dim() == 3  # [Q,Hm,Wm]
        if single:
            pred_masks = pred_masks.unsqueeze(0)  # -> [1,Q,Hm,Wm]

        B, Q, Hm, Wm = pred_masks.shape
        device = pred_masks.device
        dtype = pred_masks.dtype

        # 1) Upsample masks to processed (input) size
        proc_h, proc_w = int(processed_size[0]), int(processed_size[1])
        masks_proc = torch.nn.functional.interpolate(
            pred_masks, size=(proc_h, proc_w), mode="bilinear", align_corners=False
        )  # [B,Q,Hp,Wp] with Hp=proc_h, Wp=proc_w

        out = []
        for b in range(B):
            H0, W0 = int(orig_sizes[b, 0].item()), int(orig_sizes[b, 1].item())
            m = masks_proc[b]  # [Q, Hp, Wp]
            if keep_ratio:
                # Compute same gain/pad as in scale_boxes_ratio_kept
                gain = min(proc_h / H0, proc_w / W0)
                padw = round((proc_w - W0 * gain) / 2 - 0.1)
                padh = round((proc_h - H0 * gain) / 2 - 0.1)

                # Remove padding before final resize
                y1 = max(padh, 0)
                y2 = proc_h - max(padh, 0)
                x1 = max(padw, 0)
                x2 = proc_w - max(padw, 0)
                m = m[:, y1:y2, x1:x2]  # [Q, cropped_h, cropped_w]

            # 2) Resize to original size
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
            ).squeeze(0)  # [Q, H0, W0]
            out.append(m.clamp_(0, 1).to(device=device, dtype=dtype))

        if single:
            return [out[0]]
        return out

    def _preds_postprocess(
        self,
        outputs,
        processed_sizes,
        original_sizes,
        num_top_queries=300,
        use_focal_loss=True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        has_masks = ("pred_masks" in outputs) and (outputs["pred_masks"] is not None)
        pred_masks = outputs["pred_masks"] if has_masks else None  # [B,Q,Hm,Wm]
        B, Q = logits.shape[:2]

        boxes = self.process_boxes(
            boxes, processed_sizes, original_sizes, self.keep_ratio, self.device
        )  # B x TopQ x 4

        # scores/labels and preliminary topK over all Q*C
        if use_focal_loss:
            scores_all = torch.sigmoid(logits)  # [B,Q,C]
            flat = scores_all.flatten(1)  # [B, Q*C]
            # pre-topk to avoid scanning all queries later
            K = min(num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)  # [B,K]
            topk_labels = topk_idx - (topk_idx // self.n_outputs) * self.n_outputs  # [B,K]
            topk_qidx = topk_idx // self.n_outputs  # [B,K]
        else:
            probs = torch.softmax(logits, dim=-1)[:, :, :-1]  # [B,Q,C-1]
            topk_scores, topk_labels = probs.max(dim=-1)  # [B,Q]
            # keep at most K queries per image by score
            K = min(num_top_queries, Q)
            topk_scores, order = torch.topk(topk_scores, K, dim=-1)  # [B,K]
            topk_labels = topk_labels.gather(1, order)  # [B,K]
            topk_qidx = order

        results = []
        for b in range(B):
            sb = topk_scores[b]
            lb = topk_labels[b]
            qb = topk_qidx[b]
            # Apply per-class confidence thresholds
            conf_threshs_tensor = torch.tensor(self.conf_threshs, device=sb.device)
            keep = sb >= conf_threshs_tensor[lb]

            sb = sb[keep]
            lb = lb[keep]
            qb = qb[keep]
            # gather boxes once
            bb = boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))

            out = {
                "labels": lb.detach().cpu().numpy(),
                "boxes": bb.detach().cpu().numpy(),
                "scores": sb.detach().cpu().numpy(),
            }

            if has_masks and qb.numel() > 0:
                # gather only kept masks, then cast to half to save mem during resizing
                mb = pred_masks[b, qb]  # [K', Hm, Wm] logits or probs
                mb = mb.to(dtype=torch.float16)  # reduce VRAM and RAM during resize
                # resize to original size (list of length 1)
                masks_list = self.process_masks(
                    mb.unsqueeze(0),  # [1,K',Hm,Wm]
                    processed_size=np.array(processed_sizes[b]),  # (Hin, Win)
                    orig_sizes=np.array(original_sizes[b])[None],  # [1,2]
                    keep_ratio=self.keep_ratio,
                )
                out["mask_probs"] = (
                    masks_list[0].to(dtype=torch.float32).detach().cpu().numpy()
                )  # [B, H, W]
                # clean up masks outside of the corresponding bbox
                out["mask_probs"] = cleanup_masks(out["mask_probs"], out["boxes"])

            results.append(out)

        return results

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        """
        Get nearest size that is divisible by 32
        """
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]

        # Make sure new dimensions are divisible by the stride
        new_shape = [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]
        return new_shape

    def _preprocess(self, img: NDArray, stride: int = 32) -> torch.tensor:
        if not self.keep_ratio:  # simple resize
            img = cv2.resize(
                img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
            )
        elif self.rect:  # keep ratio and cut paddings
            target_height, target_width = self._compute_nearest_size(
                img.shape[:2], max(*self.input_size)
            )
            img = letterbox(img, (target_height, target_width), stride=stride, auto=False)[0]
        else:  # keep ratio adding paddings
            img = letterbox(
                img, (self.input_size[0], self.input_size[1]), stride=stride, auto=False
            )[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0

        # save debug image
        if self.debug_mode:
            debug_img = img.reshape([1, *img.shape])
            debug_img = debug_img[0].transpose(1, 2, 0)  # CHW to HWC
            debug_img = (debug_img * 255.0).astype(np.uint8)  # Convert to uint8
            debug_img = debug_img[:, :, ::-1]  # RGB to BGR for saving
            cv2.imwrite("torch_infer.jpg", debug_img)
        return img

    def _prepare_inputs(self, inputs):
        original_sizes = []
        processed_sizes = []

        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:  # single image
            processed_inputs = self._preprocess(inputs)[None]
            original_sizes.append((inputs.shape[0], inputs.shape[1]))
            processed_sizes.append((processed_inputs[0].shape[1], processed_inputs[0].shape[2]))

        elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:  # batch of images
            processed_inputs = np.zeros(
                (inputs.shape[0], self.channels, self.input_size[0], self.input_size[1]),
                dtype=self.np_dtype,
            )
            for idx, image in enumerate(inputs):
                processed_inputs[idx] = self._preprocess(image)
                original_sizes.append((image.shape[0], image.shape[1]))
                processed_sizes.append(
                    (processed_inputs[idx].shape[1], processed_inputs[idx].shape[2])
                )

        tensor = torch.from_numpy(processed_inputs)  # no copying
        if self.device == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        else:
            tensor = tensor.to(self.device)
        return tensor, processed_sizes, original_sizes

    @torch.no_grad()
    def _predict(self, inputs) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.model(inputs)

    def _postprocess(
        self,
        preds: torch.tensor,
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
    ):
        output = self._preds_postprocess(preds, processed_sizes, original_sizes)
        if self.use_nms:
            for idx, res in enumerate(output):
                boxes, scores, classes, masks = non_max_suppression(
                    res["boxes"],
                    res["scores"],
                    res["labels"],
                    masks=res.get("mask_probs", None),
                    iou_threshold=0.5,
                )
                output[idx]["boxes"] = boxes
                output[idx]["scores"] = scores
                output[idx]["labels"] = classes
                if "mask_probs" in res:
                    output[idx]["mask_probs"] = masks

        return output

    @torch.no_grad()
    def __call__(self, inputs: NDArray[np.uint8]) -> List[Dict[str, np.ndarray]]:
        """
        Input image as ndarray (BGR, HWC) or BHWC
        Output:
            List of batch size length. Each element is a dict {"labels", "boxes", "scores"}
            labels: np.ndarray of shape (N,), dtype np.int64
            boxes: np.ndarray of shape (N, 4), dtype np.float32, abs values
            scores: np.ndarray of shape (N,), dtype np.float32
            masks: np.ndarray of shape (N, H, W), dtype np.uint8. N = number of objects
        """
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(inputs)
        preds = self._predict(processed_inputs)
        return self._postprocess(preds, processed_sizes, original_sizes)

    @staticmethod
    def mask2poly(masks: np.ndarray, img_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Convert binary masks to normalized polygon coordinates for YOLO segmentation format.

        Args:
            masks: Binary masks array of shape [N, H, W] where N is number of instances
            img_shape: Tuple of (height, width) of the original image

        Returns:
            List of normalized polygon coordinates, each as array of shape [num_points, 2]
            with values in range [0, 1]. Returns empty array for invalid masks.
        """
        h, w = img_shape[:2]
        polys = []

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                contour = contour.reshape(-1, 2)
                if len(contour) >= 3:  # Need at least 3 points for a valid polygon
                    # Normalize coordinates
                    norm_contour = contour.astype(np.float32)
                    norm_contour[:, 0] /= w
                    norm_contour[:, 1] /= h
                    polys.append(norm_contour)
                else:
                    polys.append(np.array([]))
            else:
                polys.append(np.array([]))

        return polys


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes_ratio_kept(boxes, img1_shape, img0_shape, ratio_pad=None, padding=True):
    """
    img1_shape: (height, width) after resize
    img0_shape: (height, width) before resize
    """
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_boxes(boxes, orig_shape, resized_shape):
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def norm_xywh_to_abs_xyxy(boxes: np.ndarray, height: int, width: int, to_round=True) -> np.ndarray:
    # Convert normalized centers to absolute pixel coordinates
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height

    # Compute the top-left and bottom-right coordinates
    x_min = x_center - (box_width / 2)
    y_min = y_center - (box_height / 2)
    x_max = x_center + (box_width / 2)
    y_max = y_center + (box_height / 2)

    # Convert coordinates to integers
    if to_round:
        x_min = np.maximum(np.floor(x_min), 1)
        y_min = np.maximum(np.floor(y_min), 1)
        x_max = np.minimum(np.ceil(x_max), width - 1)
        y_max = np.minimum(np.ceil(y_max), height - 1)
        return np.stack([x_min, y_min, x_max, y_max], axis=1)
    else:
        x_min = np.maximum(x_min, 0)
        y_min = np.maximum(y_min, 0)
        x_max = np.minimum(x_max, width)
        y_max = np.minimum(y_max, height)
        return np.stack([x_min, y_min, x_max, y_max], axis=1)


def filter_preds(preds, conf_threshs: List[float]):
    conf_threshs = torch.tensor(conf_threshs, device=preds[0]["scores"].device)
    for pred in preds:
        mask = pred["scores"] >= conf_threshs[pred["labels"]]
        pred["scores"] = pred["scores"][mask]
        pred["boxes"] = pred["boxes"][mask]
        pred["labels"] = pred["labels"][mask]
    return preds


def cleanup_masks(masks, boxes):
    # clean up masks outside of the corresponding bbox
    N, H, W = masks.shape
    ys = np.arange(H)[None, :, None]  # (1, H, 1)
    xs = np.arange(W)[None, None, :]  # (1, 1, W)

    x1, y1, x2, y2 = boxes.T
    inside = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )  # (N, H, W), bool
    masks = masks * inside.astype(masks.dtype)
    return masks


def non_max_suppression(boxes, scores, classes, masks=None, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter bounding boxes.

    Parameters:
    - boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes in [x1, y1, x2, y2] format.
    - scores (torch.Tensor): Tensor of shape (N,) containing confidence scores for each box.
    - classes (torch.Tensor): Tensor of shape (N,) containing class indices for each box.
    - masks (torch.Tensor, optional): Tensor of shape (N, H, W) containing masks for each box.
    - iou_threshold (float): Intersection Over Union (IOU) threshold for NMS.

    Returns:
    - filtered_boxes (torch.Tensor): Tensor containing filtered bounding boxes after NMS.
    - filtered_scores (torch.Tensor): Tensor containing confidence scores of the filtered boxes.
    - filtered_classes (torch.Tensor): Tensor containing class indices of the filtered boxes.
    - filtered_masks (torch.Tensor or None): Tensor containing masks of the filtered boxes, or None if masks was None.
    """
    boxes = torch.from_numpy(boxes) if not isinstance(boxes, torch.Tensor) else boxes
    scores = torch.from_numpy(scores) if not isinstance(scores, torch.Tensor) else scores
    classes = torch.from_numpy(classes) if not isinstance(classes, torch.Tensor) else classes
    if masks is not None:
        masks = torch.from_numpy(masks) if not isinstance(masks, torch.Tensor) else masks
    # Prepare lists to collect the filtered boxes, scores, and classes
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    filtered_masks = []

    # Get unique classes present in the detections
    unique_classes = classes.unique()

    # Step 2: Perform NMS for each class separately
    for unique_class in unique_classes:
        # Get indices of boxes belonging to the current class
        cls_mask = classes == unique_class
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Apply NMS for the current class
        nms_indices = nms(cls_boxes, cls_scores, iou_threshold)

        # Collect the filtered boxes, scores, and classes
        filtered_boxes.append(cls_boxes[nms_indices])
        filtered_scores.append(cls_scores[nms_indices])
        filtered_classes.append(classes[cls_mask][nms_indices])

        if masks is not None:
            cls_masks = masks[cls_mask]
            filtered_masks.append(cls_masks[nms_indices])

    # Step 3: Concatenate the results
    if filtered_boxes:
        filtered_boxes = torch.cat(filtered_boxes)
        filtered_scores = torch.cat(filtered_scores)
        filtered_classes = torch.cat(filtered_classes)
        if masks is not None:
            filtered_masks = torch.cat(filtered_masks)
        else:
            filtered_masks = None
    else:
        # If no boxes remain after NMS, return empty tensors
        filtered_boxes = torch.empty((0, 4))
        filtered_scores = torch.empty((0,))
        filtered_classes = torch.empty((0,), dtype=classes.dtype)

        filtered_masks = None
        if masks is not None:
            filtered_masks = torch.empty(
                (0, *masks.shape[1:]), dtype=masks.dtype, device=masks.device
            )

    return filtered_boxes, filtered_scores, filtered_classes, filtered_masks
