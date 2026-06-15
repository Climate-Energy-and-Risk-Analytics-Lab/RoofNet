from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskMetrics:
    attribution_area: int
    sam_mask_area: int
    intersection_area: int
    union_area: int
    attribution_iou: float
    inside_ratio: float
    coverage_ratio: float


def _as_2d_array(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}, got shape {out.shape}")
    return out


def _as_bool_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    return _as_2d_array(mask, name=name).astype(bool, copy=False)


def _pil_resample(name: str):
    resampling = getattr(Image, "Resampling", Image)
    if name == "nearest":
        return resampling.NEAREST
    if name == "bilinear":
        return resampling.BILINEAR
    raise ValueError(f"Unsupported resample mode: {name}")


def resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    bool_mask = _as_bool_mask(mask, name="mask")
    height, width = shape
    image = Image.fromarray(bool_mask.astype(np.uint8) * 255)
    resized = image.resize((width, height), resample=_pil_resample("nearest"))
    return np.asarray(resized) > 0


def resize_float_map_to_shape(attribution_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = _as_2d_array(np.asarray(attribution_map, dtype=np.float32), name="attribution_map")
    height, width = shape
    image = Image.fromarray(arr)
    resized = image.resize((width, height), resample=_pil_resample("bilinear"))
    return np.asarray(resized, dtype=np.float32)


def binarize_attribution_percentile(attribution_map: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    arr = _as_2d_array(np.asarray(attribution_map, dtype=np.float32), name="attribution_map")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"Percentile must be in [0, 100], got {percentile}")
    threshold = float(np.percentile(arr, percentile))
    return arr >= threshold, threshold


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = _as_bool_mask(mask_a, name="mask_a")
    b = _as_bool_mask(mask_b, name="mask_b")
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes differ: {a.shape} vs {b.shape}")
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(intersection / union) if union > 0 else 0.0


def compute_attribution_mass(
    attribution_map: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Fraction of attribution mass inside vs outside a boolean mask.

    Uses raw (continuous) attribution values, not binarized masks.
    Returns dict with keys: mass_inside, mass_outside, total_mass.
    """
    arr = _as_2d_array(np.asarray(attribution_map, dtype=np.float64), name="attribution_map")
    m = _as_bool_mask(mask, name="mask")
    if arr.shape != m.shape:
        raise ValueError(f"Shape mismatch: attribution {arr.shape} vs mask {m.shape}")
    total = float(arr.sum())
    inside = float(arr[m].sum()) if m.any() else 0.0
    outside = total - inside
    return {
        "mass_inside": inside / total if total > 0 else 0.0,
        "mass_outside": outside / total if total > 0 else 0.0,
        "total_mass": total,
    }


def compute_attribution_mask_metrics(binary_attribution_mask: np.ndarray, sam_mask: np.ndarray) -> MaskMetrics:
    attribution = _as_bool_mask(binary_attribution_mask, name="binary_attribution_mask")
    sam = _as_bool_mask(sam_mask, name="sam_mask")
    if attribution.shape != sam.shape:
        raise ValueError(f"Mask shapes differ: {attribution.shape} vs {sam.shape}")
    intersection = int(np.logical_and(attribution, sam).sum())
    union = int(np.logical_or(attribution, sam).sum())
    attribution_area = int(attribution.sum())
    sam_area = int(sam.sum())
    iou = float(intersection / union) if union > 0 else 0.0
    inside_ratio = float(intersection / attribution_area) if attribution_area > 0 else 0.0
    coverage_ratio = float(intersection / sam_area) if sam_area > 0 else 0.0
    return MaskMetrics(
        attribution_area=attribution_area,
        sam_mask_area=sam_area,
        intersection_area=intersection,
        union_area=union,
        attribution_iou=iou,
        inside_ratio=inside_ratio,
        coverage_ratio=coverage_ratio,
    )


def combine_instance_masks(sam_masks: Sequence[np.ndarray]) -> np.ndarray:
    masks = [_as_bool_mask(mask, name="sam_mask") for mask in sam_masks]
    if not masks:
        raise ValueError("Need at least one SAM mask to combine")
    base_shape = masks[0].shape
    for mask in masks[1:]:
        if mask.shape != base_shape:
            raise ValueError(f"SAM mask shapes differ: {base_shape} vs {mask.shape}")
    return np.logical_or.reduce(masks)


def compute_instance_iou_table(
    binary_attribution_mask: np.ndarray,
    sam_masks: Sequence[np.ndarray],
    boxes: Sequence[Sequence[float]] | None = None,
    confidences: Sequence[float] | None = None,
) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    boxes = list(boxes) if boxes is not None else [None] * len(sam_masks)
    confidences = list(confidences) if confidences is not None else [None] * len(sam_masks)
    for idx, (mask, box, confidence) in enumerate(zip(sam_masks, boxes, confidences)):
        metrics = compute_attribution_mask_metrics(binary_attribution_mask, mask)
        box_area = None
        if box is not None:
            x0, y0, x1, y1 = [float(v) for v in box]
            box_area = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
        rows.append({
            "instance_id": idx,
            "gdino_confidence": None if confidence is None else float(confidence),
            "box_area": box_area,
            "sam_mask_area": metrics.sam_mask_area,
            "attribution_area": metrics.attribution_area,
            "intersection_area": metrics.intersection_area,
            "union_area": metrics.union_area,
            "attribution_iou": metrics.attribution_iou,
            "inside_ratio": metrics.inside_ratio,
            "coverage_ratio": metrics.coverage_ratio,
        })
    return rows


def random_mask_iou_baseline(
    sam_masks: Sequence[np.ndarray],
    attribution_density: float,
    shape: tuple[int, int],
    *,
    n_samples: int = 100,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    if not (0.0 <= attribution_density <= 1.0):
        raise ValueError(f"attribution_density must be in [0, 1], got {attribution_density}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    combined = combine_instance_masks(sam_masks)
    if combined.shape != shape:
        combined = resize_mask_to_shape(combined, shape)
    total_pixels = int(shape[0] * shape[1])
    num_true = int(round(attribution_density * total_pixels))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_samples):
        flat = np.zeros(total_pixels, dtype=bool)
        if num_true > 0:
            chosen = rng.choice(total_pixels, size=min(num_true, total_pixels), replace=False)
            flat[chosen] = True
        random_mask = flat.reshape(shape)
        samples.append(compute_mask_iou(random_mask, combined))
    mean_iou = float(np.mean(samples)) if samples else 0.0
    std_iou = float(np.std(samples)) if samples else 0.0
    return {
        "n_samples": n_samples,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "samples": samples,
    }


def sweep_threshold_iou(
    attribution_map: np.ndarray,
    sam_masks: Sequence[np.ndarray],
    percentiles: Iterable[float],
) -> list[dict[str, float]]:
    if not sam_masks:
        raise ValueError("Need at least one SAM mask for threshold sweep")
    combined = combine_instance_masks(sam_masks)
    resized_map = resize_float_map_to_shape(attribution_map, combined.shape)
    rows: list[dict[str, float]] = []
    for percentile in percentiles:
        binary_mask, threshold = binarize_attribution_percentile(resized_map, float(percentile))
        per_instance_rows = compute_instance_iou_table(binary_mask, sam_masks)
        per_instance_mean_iou = float(np.mean([row["attribution_iou"] for row in per_instance_rows])) if per_instance_rows else 0.0
        combined_metrics = compute_attribution_mask_metrics(binary_mask, combined)
        rows.append({
            "percentile": float(percentile),
            "threshold_value": float(threshold),
            "combined_iou": combined_metrics.attribution_iou,
            "mean_instance_iou": per_instance_mean_iou,
            "attribution_area": float(combined_metrics.attribution_area),
        })
    return rows
