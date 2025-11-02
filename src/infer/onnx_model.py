from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
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
        self.np_dtype = np.float16 if self.half else np.float32

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
        B, Q, _ = boxes.shape
        out = np.empty_like(boxes)
        for b in range(B):
            abs_xyxy = norm_xywh_to_abs_xyxy(boxes[b], proc_sizes[b][0], proc_sizes[b][1])
            if keep_ratio:
                abs_xyxy = scale_boxes_ratio_kept(abs_xyxy, proc_sizes[b], orig_sizes[b])
            else:
                abs_xyxy = scale_boxes(abs_xyxy, orig_sizes[b], proc_sizes[b])
            out[b] = abs_xyxy
        return out

    def _preds_postprocess(
        self,
        outputs: Dict[str, NDArray],
        proc_sizes,
        orig_sizes,
        num_top_queries: int = 300,
        use_focal_loss: bool = True,
    ) -> List[Dict[str, NDArray]]:
        """
        Return list length=batch of dicts {"labels","boxes","scores"} (NumPy).
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        boxes = self.process_boxes(boxes, proc_sizes, orig_sizes, self.keep_ratio)

        batch_results: List[Dict[str, NDArray]] = []
        for b in range(logits.shape[0]):
            log_b, box_b = logits[b], boxes[b]

            if use_focal_loss:
                prob = 1.0 / (1.0 + np.exp(-log_b))  # sigmoid
                flat = prob.reshape(-1)
                top_idx = np.argsort(-flat)[:num_top_queries]
                scores = flat[top_idx]
                labels = top_idx % self.n_outputs
                q_idx = top_idx // self.n_outputs
                sel_boxes = box_b[q_idx]
            else:
                exp = np.exp(log_b - log_b.max(axis=-1, keepdims=True))
                soft = exp / exp.sum(axis=-1, keepdims=True)
                soft = soft[..., :-1]  # drop background
                scores = soft.max(axis=-1)
                labels = soft.argmax(axis=-1)

                if scores.shape[0] > num_top_queries:
                    top_idx = np.argsort(-scores)[:num_top_queries]
                    scores = scores[top_idx]
                    labels = labels[top_idx]
                    sel_boxes = box_b[top_idx]
                else:
                    sel_boxes = box_b

            batch_results.append(
                dict(
                    labels=labels.astype(np.int64),
                    boxes=sel_boxes.astype(np.float32),
                    scores=scores.astype(np.float32),
                )
            )
        return batch_results

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
        return {"pred_logits": outs[0], "pred_boxes": outs[1]}

    def _postprocess(self, preds, proc_sz, orig_sz):
        out = self._preds_postprocess(preds, proc_sz, orig_sz)
        return filter_preds(out, self.conf_threshs)

    def __call__(self, inputs: NDArray[np.uint8]) -> List[Dict[str, np.ndarray]]:
        """
        Args:
            inputs (HWC BGR np.uint8) or batch BHWC
        Returns:
            list[dict] with keys: "labels", "boxes", "scores"
        """
        proc, proc_sz, orig_sz = self._prepare_inputs(inputs)
        preds = self._predict(proc)
        return self._postprocess(preds, proc_sz, orig_sz)


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


def norm_xywh_to_abs_xyxy(boxes: np.ndarray, h: int, w: int, clip=True):
    """
    Normalised (x_c, y_c, w, h) -> absolute (x1, y1, x2, y2)
    Keeps full floating-point precision; no rounding.
    """
    x_c, y_c, bw, bh = boxes.T
    x_min = x_c * w - bw * w / 2
    y_min = y_c * h - bh * h / 2
    x_max = x_c * w + bw * w / 2
    y_max = y_c * h + bh * h / 2

    if clip:
        x_min = np.clip(x_min, 0, w - 1)
        y_min = np.clip(y_min, 0, h - 1)
        x_max = np.clip(x_max, 0, w - 1)
        y_max = np.clip(y_max, 0, h - 1)

    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def filter_preds(preds, conf_threshs: List[float]):
    thresh = np.asarray(conf_threshs, dtype=np.float32)
    for p in preds:
        mask = p["scores"] >= thresh[p["labels"]]
        p["labels"] = p["labels"][mask]
        p["boxes"] = p["boxes"][mask]
        p["scores"] = p["scores"][mask]
    return preds


if __name__ == "__main__":
    import time

    # model = ONNX_model(model_path="model.onnx", n_outputs=2)
    model = ONNX_model(
        model_path="/home/argo/Desktop/Projects/Veryfi/dt/output/models/test_640s_2025-10-31/model.onnx",
        n_outputs=2,
    )
    inp = np.random.randint(0, 255, size=(1920, 1080, 3), dtype=np.uint8)

    latency = []

    for _ in range(100):
        start = time.perf_counter()
        outputs = model(inp)
        latency.append(time.perf_counter() - start)
    print(f"Average latency: {np.mean(latency) * 1000:.2f} ms")
