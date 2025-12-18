from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray


class ONNX_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        input_width: int = 640,
        input_height: int = 640,
        conf_thresh: float | List[float] = 0.5,
        rect: bool = False,
        half: bool = False,
        keep_ratio: bool = False,
        device: str | None = None,
    ):
        self.input_size = (input_height, input_width)
        self.n_outputs = n_outputs
        self.model_path = model_path
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.channels = 3
        self.debug_mode = False

        # per-class confidence thresholds
        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        else:
            self.conf_threshs = conf_thresh

        # pick execution provider
        self.device = device or "cpu"
        self.np_dtype = np.float32

        self._load_model()
        self._test_pred()  # sanity check that shapes line up

    def _load_model(self) -> None:
        providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        provider_options = (
            [{"cudnn_conv_algo_search": "DEFAULT"}] if self.device == "cuda" else [{}]
        )
        self.model = ort.InferenceSession(
            self.model_path, providers=providers, provider_options=provider_options
        )
        print(f"ONNX model loaded: {self.model_path} on {self.device}")

    def _test_pred(self) -> None:
        """Run one dummy inference so that latent bugs fail fast."""
        dummy = np.random.randint(0, 255, size=(1100, 1000, self.channels), dtype=np.uint8)
        proc, proc_sz, orig_sz = self._prepare_inputs(dummy)
        out = self._predict(proc)
        self._postprocess(out, proc_sz, orig_sz)

    @staticmethod
    def process_boxes(
        boxes: NDArray,
        proc_sizes,
        orig_sizes,
        keep_ratio: bool,
    ) -> NDArray:
        """Convert normalised xywh→absolute xyxy & rescale to original img size."""
        boxes = boxes.numpy()
        B, Q, _ = boxes.shape
        out = np.empty_like(boxes)
        for b in range(B):
            abs_xyxy = norm_xywh_to_abs_xyxy(boxes[b], proc_sizes[b][0], proc_sizes[b][1])
            if keep_ratio:
                abs_xyxy = scale_boxes_ratio_kept(abs_xyxy, proc_sizes[b], orig_sizes[b])
            else:
                abs_xyxy = scale_boxes(abs_xyxy, orig_sizes[b], proc_sizes[b])
            out[b] = abs_xyxy
        return torch.from_numpy(out)

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

    def _compute_nearest_size(self, shape, target_size, stride: int = 32) -> Tuple[int, int]:
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]
        return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]

    def _preprocess(self, img: NDArray[np.uint8], stride: int = 32) -> NDArray[np.float32]:
        if not self.keep_ratio:  # plain resize
            img = cv2.resize(
                img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
            )
        elif self.rect:  # keep ratio & crop
            h_t, w_t = self._compute_nearest_size(img.shape[:2], max(*self.input_size))
            img = letterbox(img, (h_t, w_t), stride=stride, auto=False)[0]
        else:  # keep ratio & pad
            img = letterbox(
                img, (self.input_size[0], self.input_size[1]), stride=stride, auto=False
            )[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB & HWC→CHW
        img = img.astype(self.np_dtype, copy=False) / 255.0
        return img

    def _prepare_inputs(self, inputs):
        """Returns: batched array, list[(h_p,w_p)], list[(h0,w0)]"""
        original_sizes, processed_sizes = [], []

        if inputs.ndim == 3:  # single image
            proc = self._preprocess(inputs)[None]
            original_sizes.append(inputs.shape[:2])
            processed_sizes.append(proc.shape[2:])
        else:  # batched BHWC
            batch, _, _, _ = inputs.shape
            proc = np.zeros((batch, self.channels, *self.input_size), dtype=self.np_dtype)
            for i, im in enumerate(inputs):
                proc[i] = self._preprocess(im)
                original_sizes.append(im.shape[:2])
                processed_sizes.append(proc[i].shape[1:])

        return proc, processed_sizes, original_sizes

    def _predict(self, inputs: NDArray) -> Dict[str, NDArray]:
        ort_inputs = {self.model.get_inputs()[0].name: inputs.astype(self.np_dtype)}
        outs = self.model.run(None, ort_inputs)
        return {
            "pred_logits": outs[0],
            "pred_boxes": outs[1],
            "pred_masks": outs[2] if len(outs) > 2 else None,
        }

    def _postprocess(
        self,
        outputs: torch.Tensor,
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
        num_top_queries=300,
        use_focal_loss=True,
    ) -> List[Dict[str, NDArray]]:
        """
        Return list length=batch of dicts {"labels","boxes","scores"} (NumPy).
        """
        logits, boxes = (
            torch.from_numpy(outputs["pred_logits"]),
            torch.from_numpy(outputs["pred_boxes"]),
        )
        has_masks = ("pred_masks" in outputs) and (outputs["pred_masks"] is not None)
        pred_masks = outputs["pred_masks"] if has_masks else None  # [B,Q,Hm,Wm]
        B, Q = logits.shape[:2]

        boxes = self.process_boxes(boxes, processed_sizes, original_sizes, self.keep_ratio)

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
                mb = torch.from_numpy(mb).to(
                    dtype=torch.float16
                )  # reduce VRAM and RAM during resize
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

    def __call__(self, inputs: NDArray[np.uint8]) -> List[Dict[str, np.ndarray]]:
        """
        Args:
            inputs (HWC BGR np.uint8) or batch BHWC
        Returns:
            list[dict] with keys: "labels", "boxes", "scores", "masks"
        """
        proc, proc_sz, orig_sz = self._prepare_inputs(inputs)
        preds = self._predict(proc)
        return self._postprocess(preds, proc_sz, orig_sz)

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
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw = dh = 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def scale_boxes_ratio_kept(boxes, img1_shape, img0_shape, ratio_pad=None, padding=True):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]
        boxes[..., [1, 3]] -= pad[1]

    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_boxes(boxes, orig_shape, resized_shape):
    sx, sy = orig_shape[1] / resized_shape[1], orig_shape[0] / resized_shape[0]
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
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
