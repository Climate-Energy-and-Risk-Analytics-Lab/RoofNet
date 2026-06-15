from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GroundingPrediction:
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    phrases: list[str]
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class SamMaskPrediction:
    masks: list[np.ndarray]
    scores: list[float]
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class DependencyStatus:
    ok: bool
    message: str


@dataclass(frozen=True)
class GroundingDinoBundle:
    model_id: str
    processor: Any
    model: Any
    device: str


@dataclass(frozen=True)
class SamPredictorBundle:
    model_id: str
    processor: Any
    model: Any
    device: str


def _import_transformers() -> Any:
    import transformers

    return transformers


def _to_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def check_grounding_sam_dependencies() -> dict[str, DependencyStatus]:
    statuses: dict[str, DependencyStatus] = {}
    try:
        transformers = _import_transformers()
        required_names = (
            "GroundingDinoForObjectDetection",
            "GroundingDinoProcessor",
            "SamModel",
            "SamProcessor",
        )
        missing = [name for name in required_names if not hasattr(transformers, name)]
        if missing:
            statuses["groundingdino"] = DependencyStatus(False, f"transformers missing: {missing}")
            statuses["segment_anything"] = DependencyStatus(False, f"transformers missing: {missing}")
        else:
            statuses["groundingdino"] = DependencyStatus(True, "transformers GroundingDINO classes available")
            statuses["segment_anything"] = DependencyStatus(True, "transformers SAM classes available")
    except Exception as exc:
        message = f"transformers import failed: {exc!r}"
        statuses["groundingdino"] = DependencyStatus(False, message)
        statuses["segment_anything"] = DependencyStatus(False, message)
    return statuses


def load_groundingdino_model(model_id: str, *, device: str = "cpu") -> GroundingDinoBundle:
    transformers = _import_transformers()
    processor = transformers.GroundingDinoProcessor.from_pretrained(model_id)
    model = transformers.GroundingDinoForObjectDetection.from_pretrained(model_id)
    model.to(device=device)
    return GroundingDinoBundle(model_id=model_id, processor=processor, model=model, device=device)


def load_sam_predictor(model_id: str, *, device: str = "cpu") -> SamPredictorBundle:
    transformers = _import_transformers()
    processor = transformers.SamProcessor.from_pretrained(model_id)
    model = transformers.SamModel.from_pretrained(model_id)
    model.to(device=device)
    return SamPredictorBundle(model_id=model_id, processor=processor, model=model, device=device)


def run_groundingdino_inference(
    model: GroundingDinoBundle,
    image: Image.Image,
    *,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> GroundingPrediction:
    image_rgb = image.convert("RGB")
    height, width = image_rgb.height, image_rgb.width
    inputs = model.processor(images=image_rgb, text=prompt, return_tensors="pt").to(model.device)
    outputs = model.model(**inputs)
    results = model.processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.get("input_ids"),
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(height, width)],
    )
    result = results[0] if results else {}
    boxes = _to_numpy(result.get("boxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
    scores = _to_numpy(result.get("scores", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    phrases = [str(label) for label in result.get("text_labels", result.get("labels", []))]
    if boxes.size == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores.size == 0:
        scores = np.zeros((0,), dtype=np.float32)
    return GroundingPrediction(
        boxes_xyxy=boxes,
        confidences=scores,
        phrases=phrases,
        image_shape=(height, width),
    )


def run_sam_box_refinement(
    predictor: SamPredictorBundle,
    image: Image.Image,
    boxes_xyxy: Sequence[Sequence[float]],
) -> SamMaskPrediction:
    image_rgb = image.convert("RGB")
    height, width = image_rgb.height, image_rgb.width
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return SamMaskPrediction(masks=[], scores=[], image_shape=(height, width))

    boxes_list = [[[float(v) for v in box]] for box in boxes_xyxy]
    inputs = predictor.processor(
        image_rgb,
        input_boxes=[boxes_list],
        return_tensors="pt",
    ).to(predictor.device)
    outputs = predictor.model(**inputs)
    processed_masks = predictor.processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )
    mask_candidates = _to_numpy(processed_masks[0], dtype=bool)
    iou_scores = _to_numpy(outputs.iou_scores, dtype=np.float32)
    if iou_scores.ndim == 3 and iou_scores.shape[0] == 1:
        iou_scores = iou_scores[0]
    print(
        "SAM refinement diagnostics: "
        f"boxes={len(boxes_xyxy)}, "
        f"mask_candidates.shape={getattr(mask_candidates, 'shape', None)}, "
        f"iou_scores.shape={getattr(iou_scores, 'shape', None)}"
    )

    masks: list[np.ndarray] = []
    scores: list[float] = []
    num_boxes = min(len(boxes_xyxy), int(mask_candidates.shape[0]))
    for box_idx in range(num_boxes):
        box_masks = np.asarray(mask_candidates[box_idx])
        while box_masks.ndim > 3 and box_masks.shape[0] == 1:
            box_masks = box_masks[0]
        if box_masks.ndim == 2:
            box_masks = box_masks[None, ...]
        elif box_masks.ndim != 3:
            raise ValueError(
                f"Unexpected SAM mask shape for box {box_idx}: {box_masks.shape}"
            )

        box_scores = np.asarray(iou_scores[box_idx], dtype=np.float32)
        box_scores = np.ravel(box_scores)
        if box_scores.size == 0:
            box_scores = np.zeros((1,), dtype=np.float32)

        best_idx = int(np.argmax(box_scores))
        clamped_idx = min(best_idx, box_masks.shape[0] - 1)
        if clamped_idx != best_idx:
            print(
                f"SAM refinement warning: score index {best_idx} out of range for "
                f"box {box_idx} with {box_masks.shape[0]} mask candidates; using {clamped_idx}."
            )
        masks.append(np.asarray(box_masks[clamped_idx], dtype=bool))
        scores.append(float(box_scores[min(best_idx, box_scores.size - 1)]))
    return SamMaskPrediction(masks=masks, scores=scores, image_shape=(height, width))
