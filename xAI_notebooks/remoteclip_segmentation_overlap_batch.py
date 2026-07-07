#!/usr/bin/env python3
"""RemoteCLIP Segmentation Overlap Batch Runner.

Non-interactive batch script that reproduces the main segmentation-overlap
analysis from the companion marimo notebook across the full holdout split.

Features:
- resumable execution (skips already completed images)
- continue-on-error with per-image failure logging
- per-image combined mask + overlay export
- aggregate summary JSON and consistency plots

Usage:
    python xAI_notebooks/remoteclip_segmentation_overlap_batch.py
    python xAI_notebooks/remoteclip_segmentation_overlap_batch.py --limit 2
    python xAI_notebooks/remoteclip_segmentation_overlap_batch.py --force-recompute
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Resolve repo root and add to path so attribution_helpers is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attribution_helpers import grounding_sam
from attribution_helpers import segmentation_iou
from attribution_helpers import dataset_split_helpers
from attribution_helpers.transformer_explainability import transformer_explainability

# Load HF_TOKEN from .env if present (suppresses unauthenticated HF Hub warnings)
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k == "HF_TOKEN" and _v:
                os.environ.setdefault("HF_TOKEN", _v)

try:
    import open_clip
except ImportError as exc:
    sys.exit("open_clip_torch not installed. Run: pip install open_clip_torch")

try:
    import kagglehub  # noqa: F401
except ImportError:
    kagglehub = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MATERIAL_CLASSES = [
    "Thatch", "StoneSlates", "ClayTiles", "AsphaltTiles",
    "ConcreteTiles", "WoodTiles", "MetalSheetMaterials", "PolycarbonateSheetMaterials",
    "GlassSheetMaterials", "AmorphousConcrete", "AmorphousAsphalt",
    "AmorphousMembrane", "AmorphousFabric", "Unknown", "GreenVegetative",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

MODEL_NAME = "ViT-L-14"
PRETRAINED_WEIGHTS = "laion2b_s32b_b82k"
IMAGE_SIZE = (224, 224)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RemoteCLIP Segmentation Overlap Batch Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="holdout", help="Dataset split")
    parser.add_argument("--method", default="transformer_explainability", help="Attribution method")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "xAI_outputs" / "segmentation"), help="Output root")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N images (for parallel job splitting)")
    parser.add_argument("--device", default=None, help="Device override (auto-detect if absent)")
    parser.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-huge")
    parser.add_argument("--grounding-text-prompt", default="building . rooftop . roof")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.15)
    parser.add_argument("--attribution-percentile", type=float, default=80.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--force-recompute", action="store_true", help="Ignore resume state, recompute all")
    parser.add_argument("--skip-failed", action="store_true", help="Skip previously failed images (reserved)")
    parser.add_argument("--model-weights", default=str(REPO_ROOT / "best_clip_model_balanced.pth"))
    parser.add_argument("--image-dir", default=str(REPO_ROOT / "RoofNet-Images"))
    parser.add_argument("--metadata-csv", default=str(REPO_ROOT / "roofnet_metadata.csv"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, level: str) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_{timestamp}.log"

    logger = logging.getLogger("batch_runner")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(getattr(logging, level))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Image ID generation
# ---------------------------------------------------------------------------

def make_image_id(image_path: Path, image_root: Path) -> str:
    """Deterministic image ID from normalized relative path."""
    try:
        rel = image_path.resolve().relative_to(image_root.resolve())
    except ValueError:
        rel = Path(image_path.name)
    raw = str(rel).replace("\\", "/")
    # Replace separators and unsafe chars
    image_id = raw.replace("/", "__")
    for ch in [":", "*", "?", '"', "<", ">", "|", " "]:
        image_id = image_id.replace(ch, "_")
    return image_id


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------

@dataclass
class AssetPaths:
    root: Path
    model_weights: Path
    image_dir: Path
    metadata_csv: Optional[Path] = None


def resolve_assets(args: argparse.Namespace) -> AssetPaths:
    weights = Path(args.model_weights)
    image_dir = Path(args.image_dir)
    metadata_csv = Path(args.metadata_csv)

    if not weights.exists():
        raise FileNotFoundError(f"Missing model weights: {weights}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    return AssetPaths(
        root=REPO_ROOT,
        model_weights=weights,
        image_dir=image_dir,
        metadata_csv=metadata_csv if metadata_csv.exists() else None,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_remoteclip(weights_path: Path, device: str):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_WEIGHTS)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer


def load_segmentation_models(args: argparse.Namespace, device: str, logger: logging.Logger):
    """Load GroundingDINO and SAM. Returns (gdino_bundle, sam_bundle)."""
    logger.info("Loading GroundingDINO from %s ...", args.gdino_model_id)
    gdino_bundle = grounding_sam.load_groundingdino_model(args.gdino_model_id, device=device)
    logger.info("GroundingDINO loaded.")

    logger.info("Loading SAM from %s ...", args.sam_model_id)
    sam_bundle = grounding_sam.load_sam_predictor(args.sam_model_id, device=device)
    logger.info("SAM loaded.")

    return gdino_bundle, sam_bundle


# ---------------------------------------------------------------------------
# Prompts / city extraction
# ---------------------------------------------------------------------------

def extract_city_name(filename: str) -> str:
    from pathlib import Path as _P
    base = _P(filename).stem
    if "-" in base:
        return base.split("-")[0].replace("_", " ").title()
    if "height" in base:
        return base.split("_height")[0].replace("_", " ").title()
    if "imsat" in base:
        return base.split("_imsat")[0].replace("_", " ").title()
    return base


def build_prompts(city_name: str) -> List[str]:
    return [f"{m} in {city_name}" for m in MATERIAL_CLASSES]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def make_preprocess():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
    ])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, tokenizer, image_tensor: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(prompts).to(image_tensor.device)
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(tokenized)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_features @ text_features.T
    probs = logits.softmax(dim=-1)
    return {"logits": logits.squeeze(0), "probs": probs.squeeze(0)}


# ---------------------------------------------------------------------------
# Target score forward (for attribution)
# ---------------------------------------------------------------------------

def target_score_forward(model, tokenizer, image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> torch.Tensor:
    tokenized = tokenizer(prompts).to(image_tensor.device)
    image_features = model.encode_image(image_tensor)
    with torch.no_grad():
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_features @ text_features.T
    return logits[:, target_idx]


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

@dataclass
class ResumeState:
    completed_ids: set[str] = field(default_factory=set)
    existing_results: List[Dict[str, Any]] = field(default_factory=list)


def load_resume_state(results_csv: Path, masks_dir: Path, overlays_dir: Path, logger: logging.Logger) -> ResumeState:
    """Load existing results and verify artifact presence."""
    state = ResumeState()
    if not results_csv.exists():
        return state

    import csv as _csv
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id", "")
            if not image_id:
                continue
            mask_path = masks_dir / f"{image_id}__combined_mask.png"
            overlay_path = overlays_dir / f"{image_id}__segmentation_overlay.png"
            if mask_path.exists() and overlay_path.exists():
                state.completed_ids.add(image_id)
                state.existing_results.append(row)
            else:
                logger.debug("Incomplete artifacts for %s, will recompute", image_id)

    logger.info("Resume: %d fully completed images found.", len(state.completed_ids))
    return state


# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------

@dataclass
class OutputDirs:
    root: Path
    masks: Path
    overlays: Path
    tables: Path
    summary: Path
    plots: Path
    logs: Path


def ensure_output_dirs(root: Path) -> OutputDirs:
    dirs = OutputDirs(
        root=root,
        masks=root / "masks",
        overlays=root / "overlays",
        tables=root / "tables",
        summary=root / "summary",
        plots=root / "plots",
        logs=root / "logs",
    )
    for d in [dirs.masks, dirs.overlays, dirs.tables, dirs.summary, dirs.plots, dirs.logs]:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def save_combined_mask(combined_mask: np.ndarray, path: Path) -> None:
    """Save boolean mask as foreground=255 / background=0 PNG."""
    img = Image.fromarray(combined_mask.astype(np.uint8) * 255, mode="L")
    img.save(str(path))


def save_overlay_image(
    pil_img: Image.Image,
    combined_mask: np.ndarray,
    path: Path,
    *,
    overlay_alpha: float = 0.45,
    contour_color: str = "cyan",
) -> None:
    """Save overlay: original + green mask + contour."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pil_img)
    # Green overlay
    green = np.zeros((*combined_mask.shape, 4), dtype=np.float32)
    green[combined_mask] = [0.0, 1.0, 0.0, overlay_alpha]
    ax.imshow(green)
    # Contour
    from matplotlib.colors import to_rgba
    contour_rgba = to_rgba(contour_color, alpha=0.9)
    mask_uint8 = combined_mask.astype(np.uint8) * 255
    # Simple contour via gradient magnitude
    try:
        import cv2
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c = c.squeeze(1)
            if len(c) > 1:
                ax.plot(c[:, 0], c[:, 1], color=contour_rgba, linewidth=1.5)
    except ImportError:
        # Fallback: draw edge of mask using gradient
        gy, gx = np.gradient(combined_mask.astype(float))
        edge = (gx ** 2 + gy ** 2) > 0
        contour_overlay = np.zeros((*edge.shape, 4), dtype=np.float32)
        contour_overlay[edge] = list(contour_rgba)
        ax.imshow(contour_overlay)

    ax.axis("off")
    fig.tight_layout(pad=0.1)
    plt.savefig(str(path), dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

RESULTS_COLUMNS = [
    "image_id", "image_path", "image_filename", "split", "method", "city_name",
    "predicted_class_index", "predicted_class_label", "predicted_probability",
    "gdino_model_id", "sam_model_id", "grounding_text_prompt",
    "gdino_box_threshold", "gdino_text_threshold",
    "gdino_num_boxes", "sam_num_masks", "used_full_image_fallback", "combined_mask_area",
    "attribution_percentile", "threshold_value", "attribution_density",
    "attribution_map_height", "attribution_map_width", "mask_height", "mask_width",
    "attribution_mass_inside", "attribution_mass_outside",
    "combined_iou", "inside_ratio", "coverage_ratio",
    "attribution_area", "intersection_area", "union_area", "sam_mask_area",
    "runtime_seconds", "status",
]


def append_results_csv(row: Dict[str, Any], csv_path: Path) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_failure_jsonl(record: Dict[str, Any], jsonl_path: Path) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    *,
    image_path: Path,
    image_id: str,
    args: argparse.Namespace,
    model,
    tokenizer,
    preprocess,
    gdino_bundle,
    sam_bundle,
    device: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Process one image. Returns result dict. Raises on fatal error."""
    t0 = time.time()
    split = args.split
    method = args.method

    # --- Load image ---
    pil_img = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # --- City + prompts ---
    city_name = extract_city_name(image_path.name)
    prompts = build_prompts(city_name)

    # --- Prediction ---
    pred = predict(model, tokenizer, image_tensor, prompts)
    probs = pred["probs"]
    top_idx = int(probs.argmax().item())
    top_prob = float(probs[top_idx].item())
    predicted_label = MATERIAL_CLASSES[top_idx]
    logger.info("  Predicted: %s (%.4f)", predicted_label, top_prob)

    # --- Attribution ---
    logger.info("  Running %s attribution ...", method)
    attribution_heatmap = transformer_explainability(
        model=model,
        tokenizer=tokenizer,
        image_tensor=image_tensor,
        prompts=prompts,
        target_idx=top_idx,
        image_size=IMAGE_SIZE,
        verbose=False,
    )
    attr_h, attr_w = attribution_heatmap.shape

    # --- GroundingDINO ---
    logger.info("  Running GroundingDINO ...")
    grounding_prediction = grounding_sam.run_groundingdino_inference(
        gdino_bundle, pil_img,
        prompt=args.grounding_text_prompt,
        box_threshold=args.gdino_box_threshold,
        text_threshold=args.gdino_text_threshold,
    )
    num_boxes = len(grounding_prediction.boxes_xyxy)
    logger.info("  GDINO boxes: %d", num_boxes)

    used_fallback = False
    if num_boxes == 0:
        w, h = pil_img.size
        boxes_for_sam = [[0.0, 0.0, float(w), float(h)]]
        used_fallback = True
        logger.info("  Using full-image fallback box for SAM.")
    else:
        boxes_for_sam = grounding_prediction.boxes_xyxy

    # --- SAM ---
    logger.info("  Running SAM refinement ...")
    sam_prediction = grounding_sam.run_sam_box_refinement(sam_bundle, pil_img, boxes_for_sam)
    num_masks = len(sam_prediction.masks)
    logger.info("  SAM masks: %d", num_masks)

    if num_masks == 0:
        raise RuntimeError("SAM returned zero masks even after fallback")

    # --- Combined mask ---
    combined_mask = segmentation_iou.combine_instance_masks(sam_prediction.masks)
    mask_h, mask_w = combined_mask.shape

    # --- Metrics ---
    resized_map = segmentation_iou.resize_float_map_to_shape(attribution_heatmap, combined_mask.shape)
    binary_mask, threshold_value = segmentation_iou.binarize_attribution_percentile(
        resized_map, args.attribution_percentile
    )
    combined_metrics = segmentation_iou.compute_attribution_mask_metrics(binary_mask, combined_mask)
    mass_metrics = segmentation_iou.compute_attribution_mass(resized_map, combined_mask)
    attribution_density = float(binary_mask.mean())

    logger.info(
        "  Mass in=%.4f out=%.4f IoU=%.4f",
        mass_metrics["mass_inside"], mass_metrics["mass_outside"], combined_metrics.attribution_iou,
    )

    # --- Build result row ---
    runtime = time.time() - t0
    row: Dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(image_path),
        "image_filename": image_path.name,
        "split": split,
        "method": method,
        "city_name": city_name,
        "predicted_class_index": top_idx,
        "predicted_class_label": predicted_label,
        "predicted_probability": top_prob,
        "gdino_model_id": args.gdino_model_id,
        "sam_model_id": args.sam_model_id,
        "grounding_text_prompt": args.grounding_text_prompt,
        "gdino_box_threshold": args.gdino_box_threshold,
        "gdino_text_threshold": args.gdino_text_threshold,
        "gdino_num_boxes": num_boxes,
        "sam_num_masks": num_masks,
        "used_full_image_fallback": used_fallback,
        "combined_mask_area": int(combined_mask.sum()),
        "attribution_percentile": args.attribution_percentile,
        "threshold_value": threshold_value,
        "attribution_density": attribution_density,
        "attribution_map_height": attr_h,
        "attribution_map_width": attr_w,
        "mask_height": mask_h,
        "mask_width": mask_w,
        "attribution_mass_inside": mass_metrics["mass_inside"],
        "attribution_mass_outside": mass_metrics["mass_outside"],
        "combined_iou": combined_metrics.attribution_iou,
        "inside_ratio": combined_metrics.inside_ratio,
        "coverage_ratio": combined_metrics.coverage_ratio,
        "attribution_area": combined_metrics.attribution_area,
        "intersection_area": combined_metrics.intersection_area,
        "union_area": combined_metrics.union_area,
        "sam_mask_area": combined_metrics.sam_mask_area,
        "runtime_seconds": round(runtime, 3),
        "status": "success",
    }
    return row, combined_mask, pil_img


# ---------------------------------------------------------------------------
# Aggregation + summary
# ---------------------------------------------------------------------------

def _percentile(values: List[float], p: float) -> float:
    return float(np.percentile(values, p))


def compute_summary(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    total_candidates: int,
    num_failed: int,
    output_dirs: OutputDirs,
) -> Dict[str, Any]:
    """Build aggregate summary dict from successful results."""
    metric_keys = [
        "attribution_mass_inside", "attribution_mass_outside",
        "combined_iou", "inside_ratio", "coverage_ratio",
    ]

    def _metric_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p25": _percentile(values, 25),
            "median": _percentile(values, 50),
            "p75": _percentile(values, 75),
            "max": float(np.max(values)),
        }

    metric_summaries = {}
    for key in metric_keys:
        vals = [float(r[key]) for r in results if key in r]
        metric_summaries[key] = _metric_stats(vals)

    # Consistency rates
    mass_inside_vals = [float(r["attribution_mass_inside"]) for r in results]
    iou_vals = [float(r["combined_iou"]) for r in results]
    n = len(results) if results else 1

    consistency = {}
    if results:
        consistency["mass_inside_ge_50"] = sum(1 for v in mass_inside_vals if v >= 0.50) / n
        consistency["mass_inside_ge_70"] = sum(1 for v in mass_inside_vals if v >= 0.70) / n
        consistency["combined_iou_ge_10"] = sum(1 for v in iou_vals if v >= 0.10) / n
        consistency["combined_iou_ge_20"] = sum(1 for v in iou_vals if v >= 0.20) / n
        consistency["combined_iou_ge_30"] = sum(1 for v in iou_vals if v >= 0.30) / n
        consistency["high_mass_and_iou"] = sum(
            1 for m, i in zip(mass_inside_vals, iou_vals) if m >= 0.70 and i >= 0.20
        ) / n

    # Health stats
    fallback_count = sum(1 for r in results if r.get("used_full_image_fallback"))
    gdino_boxes = [float(r.get("gdino_num_boxes", 0)) for r in results]
    sam_masks = [float(r.get("sam_num_masks", 0)) for r in results]

    summary = {
        "run_config": {
            "split": args.split,
            "method": args.method,
            "output_dir": str(output_dirs.root),
            "gdino_model_id": args.gdino_model_id,
            "sam_model_id": args.sam_model_id,
            "grounding_text_prompt": args.grounding_text_prompt,
            "gdino_box_threshold": args.gdino_box_threshold,
            "gdino_text_threshold": args.gdino_text_threshold,
            "attribution_percentile": args.attribution_percentile,
            "device": args.device or "auto",
            "total_candidate_images": total_candidates,
            "completed_images": len(results),
            "failed_attempt_count": num_failed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "metric_summaries": metric_summaries,
        "consistency_rates": consistency,
        "health_stats": {
            "success_count": len(results),
            "unique_failure_image_count": num_failed,
            "success_rate_over_attempted": len(results) / max(len(results) + num_failed, 1),
            "fallback_box_rate": fallback_count / max(len(results), 1),
            "mean_gdino_boxes": float(np.mean(gdino_boxes)) if gdino_boxes else 0.0,
            "mean_sam_masks": float(np.mean(sam_masks)) if sam_masks else 0.0,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_plots(results: List[Dict[str, Any]], plots_dir: Path, logger: logging.Logger) -> None:
    """Generate all aggregate plots from successful results."""
    if not results:
        logger.warning("No successful results to plot.")
        return

    mass_inside = [float(r["attribution_mass_inside"]) for r in results]
    mass_outside = [float(r["attribution_mass_outside"]) for r in results]
    iou_vals = [float(r["combined_iou"]) for r in results]
    inside_ratio = [float(r["inside_ratio"]) for r in results]
    coverage_ratio = [float(r["coverage_ratio"]) for r in results]

    # 1. Histogram: attribution mass inside
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mass_inside, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", label="0.50 threshold")
    ax.axvline(0.7, color="orange", linestyle="--", label="0.70 threshold")
    ax.set_xlabel("Attribution Mass Inside Segmented Region")
    ax.set_ylabel("Count")
    ax.set_title("Attribution Mass Inside (higher = better subject focus)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(plots_dir / "attribution_mass_inside_hist.png"), dpi=150)
    plt.close(fig)

    # 2. Histogram: attribution mass outside
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mass_outside, bins=30, color="#FF5722", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Attribution Mass Outside Segmented Region")
    ax.set_ylabel("Count")
    ax.set_title("Attribution Mass Outside (lower = better)")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "attribution_mass_outside_hist.png"), dpi=150)
    plt.close(fig)

    # 3. Histogram: combined IoU
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(iou_vals, bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.axvline(0.10, color="red", linestyle="--", label="0.10 threshold")
    ax.axvline(0.20, color="orange", linestyle="--", label="0.20 threshold")
    ax.set_xlabel("Combined IoU")
    ax.set_ylabel("Count")
    ax.set_title("Combined Attribution-Segmentation IoU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(plots_dir / "combined_iou_hist.png"), dpi=150)
    plt.close(fig)

    # 4. Scatter: mass inside vs combined IoU
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mass_inside, iou_vals, alpha=0.5, s=20, c="#673AB7")
    ax.axvline(0.7, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Attribution Mass Inside")
    ax.set_ylabel("Combined IoU")
    ax.set_title("Mass Inside vs Combined IoU")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "mass_inside_vs_iou_scatter.png"), dpi=150)
    plt.close(fig)

    # 5. Metric boxplots
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [mass_inside, iou_vals, inside_ratio, coverage_ratio]
    box_labels = ["Mass Inside", "Combined IoU", "Inside Ratio", "Coverage Ratio"]
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Value")
    ax.set_title("Segmentation-Attribution Metric Distributions")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "metric_boxplots.png"), dpi=150)
    plt.close(fig)

    # 6. Consistency threshold bar chart
    thresholds = {
        "Mass In ≥ 0.50": sum(1 for v in mass_inside if v >= 0.50) / len(mass_inside),
        "Mass In ≥ 0.70": sum(1 for v in mass_inside if v >= 0.70) / len(mass_inside),
        "IoU ≥ 0.10": sum(1 for v in iou_vals if v >= 0.10) / len(iou_vals),
        "IoU ≥ 0.20": sum(1 for v in iou_vals if v >= 0.20) / len(iou_vals),
        "IoU ≥ 0.30": sum(1 for v in iou_vals if v >= 0.30) / len(iou_vals),
        "Both ≥ (0.70, 0.20)": sum(
            1 for m, i in zip(mass_inside, iou_vals) if m >= 0.70 and i >= 0.20
        ) / len(mass_inside),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        list(thresholds.keys()),
        [v * 100 for v in thresholds.values()],
        color=["#2196F3", "#1976D2", "#4CAF50", "#388E3C", "#2E7D32", "#FF9800"],
        edgecolor="white",
    )
    for bar, val in zip(bars, thresholds.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Consistency: Fraction of Images Meeting Thresholds")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "consistency_threshold_bars.png"), dpi=150)
    plt.close(fig)

    # Nice-to-have: ECDF for mass_inside and combined_iou
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, label, color in [
        (ax1, mass_inside, "Attribution Mass Inside", "#2196F3"),
        (ax2, iou_vals, "Combined IoU", "#4CAF50"),
    ]:
        sorted_vals = np.sort(vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, ecdf, color=color, linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF: {label}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(plots_dir / "ecdf_mass_inside_and_iou.png"), dpi=150)
    plt.close(fig)

    logger.info("Plots saved to %s", plots_dir)


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def run_batch(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    dirs = ensure_output_dirs(output_dir)
    logger = setup_logging(dirs.root, args.log_level)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    results_csv = dirs.tables / "segmentation_overlap_results.csv"
    failures_jsonl = dirs.tables / "segmentation_overlap_failures.jsonl"

    logger.info("=" * 60)
    logger.info("RemoteCLIP Segmentation Overlap Batch Runner")
    logger.info("=" * 60)
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Device: %s", device)
    logger.info("Split: %s", args.split)
    logger.info("Method: %s", args.method)
    logger.info("Output dir: %s", dirs.root)

    # --- Resolve assets ---
    assets = resolve_assets(args)
    logger.info("Model weights: %s", assets.model_weights)
    logger.info("Image dir: %s", assets.image_dir)

    # --- Load models ---
    logger.info("Loading RemoteCLIP ...")
    model, tokenizer = load_remoteclip(assets.model_weights, device)
    logger.info("RemoteCLIP loaded from %s", assets.model_weights)

    preprocess = make_preprocess()

    gdino_bundle, sam_bundle = load_segmentation_models(args, device, logger)

    # --- Collect images ---
    if assets.metadata_csv is not None and pd is not None:
        metadata_df = pd.read_csv(assets.metadata_csv, low_memory=False)
        logger.info("Loaded metadata from %s", assets.metadata_csv)
        images, _ = dataset_split_helpers.collect_split_image_paths(
            image_dir=assets.image_dir,
            metadata_df=metadata_df,
            split=args.split,
            image_exts=IMAGE_EXTS,
        )
    else:
        if assets.metadata_csv and pd is None:
            logger.warning("pandas not installed; falling back to unfiltered image discovery.")
        images = sorted(
            p for p in assets.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )

    if args.offset:
        images = images[args.offset:]
    if args.limit:
        images = images[: args.limit]

    logger.info("Total candidate images: %d", len(images))

    if not images:
        logger.error("No images found for split=%s. Exiting.", args.split)
        return

    # --- Resume ---
    if args.force_recompute:
        logger.info("--force-recompute: ignoring existing results.")
        resume = ResumeState()
    else:
        resume = load_resume_state(results_csv, dirs.masks, dirs.overlays, logger)

    pending = [
        (img, make_image_id(img, assets.image_dir))
        for img in images
        if make_image_id(img, assets.image_dir) not in resume.completed_ids
    ]
    logger.info("Pending images: %d", len(pending))

    # --- Progress bar ---
    iterator: Any = pending
    if tqdm is not None:
        iterator = tqdm(pending, desc="Processing", unit="img")

    num_failures = 0

    # --- Process loop ---
    for image_path, image_id in iterator:
        logger.info("[%d/%d] %s", len(resume.completed_ids) + 1, len(images), image_id)
        try:
            row, combined_mask, pil_img = process_image(
                image_path=image_path,
                image_id=image_id,
                args=args,
                model=model,
                tokenizer=tokenizer,
                preprocess=preprocess,
                gdino_bundle=gdino_bundle,
                sam_bundle=sam_bundle,
                device=device,
                logger=logger,
            )

            # Write artifacts
            mask_path = dirs.masks / f"{image_id}__combined_mask.png"
            overlay_path = dirs.overlays / f"{image_id}__segmentation_overlay.png"
            save_combined_mask(combined_mask, mask_path)
            save_overlay_image(pil_img, combined_mask, overlay_path)

            # Append result row
            append_results_csv(row, results_csv)
            resume.completed_ids.add(image_id)
            logger.info("  ✓ Done in %.1fs", row["runtime_seconds"])

        except Exception as exc:
            num_failures += 1
            tb = traceback.format_exc()
            logger.error("  ✗ FAILED: %s\n%s", exc, tb)
            failure_record = {
                "image_id": image_id,
                "image_path": str(image_path),
                "split": args.split,
                "method": args.method,
                "stage": "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Try to identify stage from error
            msg = str(exc).lower()
            if "load" in msg and "image" in msg:
                failure_record["stage"] = "load_image"
            elif "predict" in msg:
                failure_record["stage"] = "predict"
            elif "attribution" in msg or "explainability" in msg:
                failure_record["stage"] = "attribution"
            elif "grounding" in msg or "gdino" in msg:
                failure_record["stage"] = "groundingdino"
            elif "sam" in msg:
                failure_record["stage"] = "sam"
            elif "metric" in msg or "iou" in msg:
                failure_record["stage"] = "metrics"
            elif "save" in msg or "write" in msg or "artifact" in msg:
                failure_record["stage"] = "write_artifacts"
            append_failure_jsonl(failure_record, failures_jsonl)

    # --- Aggregation ---
    logger.info("=" * 60)
    logger.info("Batch processing complete. Generating summary ...")

    # Reload all successful results from disk
    all_results: List[Dict[str, Any]] = []
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results.append(row)

    # Type-cast numeric fields
    numeric_fields = [
        "predicted_class_index", "gdino_num_boxes", "sam_num_masks",
        "combined_mask_area", "attribution_map_height", "attribution_map_width",
        "mask_height", "mask_width", "runtime_seconds",
    ]
    float_fields = [
        "predicted_probability", "gdino_box_threshold", "gdino_text_threshold",
        "threshold_value", "attribution_density",
        "attribution_mass_inside", "attribution_mass_outside",
        "combined_iou", "inside_ratio", "coverage_ratio",
        "attribution_area", "intersection_area", "union_area", "sam_mask_area",
    ]
    for row in all_results:
        for k in numeric_fields:
            if k in row:
                try:
                    row[k] = int(float(row[k]))
                except (ValueError, TypeError):
                    pass
        for k in float_fields:
            if k in row:
                try:
                    row[k] = float(row[k])
                except (ValueError, TypeError):
                    pass
        if "used_full_image_fallback" in row:
            row["used_full_image_fallback"] = row["used_full_image_fallback"] in (True, "True", "true", "1")

    summary = compute_summary(all_results, args, len(images), num_failures, dirs)
    summary_path = dirs.summary / "segmentation_overlap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary JSON: %s", summary_path)

    generate_plots(all_results, dirs.plots, logger)

    logger.info("=" * 60)
    logger.info("DONE. Successful: %d | Failed: %d | Total: %d", len(all_results), num_failures, len(images))
    logger.info("Results CSV: %s", results_csv)
    logger.info("Failures JSONL: %s", failures_jsonl)
    logger.info("Summary JSON: %s", summary_path)
    logger.info("Plots: %s", dirs.plots)
    logger.info("Masks: %s", dirs.masks)
    logger.info("Overlays: %s", dirs.overlays)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_batch(parse_args())
