# xAI Notebooks

This folder contains notebook-oriented code for explainability experiments on the fine-tuned RemoteCLIP roof-material classifier.

Current focus:
- interactive attribution analysis in **marimo**
- segmentation-overlap analysis (GroundingDINO + SAM vs. attribution)
- non-interactive batch runner for holdout-set segmentation overlap
- helper modules for attribution methods, segmentation, and aggregation

## Folder structure

```text
xAI_notebooks/
├── README.md
├── RESULTS.md                                      # experimental results and interpretations
├── remoteclip_xai_attribution_marimo.py            # interactive attribution notebook
├── remoteclip_segmentation_overlap_marimo.py       # interactive segmentation-overlap notebook
├── remoteclip_segmentation_overlap_batch.py        # non-interactive batch runner
└── attribution_helpers/
    ├── __init__.py
    ├── feature_attribution_aggregation.py
    ├── transformer_explainability.py
    ├── manual_gradcam.py
    ├── captum_integrated_gradients.py
    ├── rise.py
    ├── grounding_sam.py
    ├── segmentation_iou.py
    ├── dataset_split_helpers.py
    └── batch_recovery.py
```

## What each file does

### Notebooks

#### `remoteclip_xai_attribution_marimo.py`

Interactive marimo notebook for per-image and batch attribution analysis.

What it handles:
- environment/import checks
- model + asset loading
- dataset-split-aware image sampling via metadata CSV
- RemoteCLIP prediction sanity checks
- attribution method registration (Transformer Explainability, GradCAM, Integrated Gradients, RISE)
- per-method visualization (heatmap, overlay, three-panel view)
- configurable batch attribution export (method subset, split filter)
- transformer explainability aggregation export

If you are exploring this repo and want to start somewhere, start here.

#### `remoteclip_segmentation_overlap_marimo.py`

Interactive marimo notebook for segmentation-overlap analysis: does model attribution attend to the actual roof/building subject?

What it handles:
- GroundingDINO proposal generation with configurable text prompt and thresholds
- SAM mask refinement from proposed bounding boxes
- full-image fallback box when GroundingDINO returns zero detections
- per-attribution-method heatmap vs. combined SAM mask comparison
- IoU, inside-ratio, coverage-ratio, and attribution-mass metrics
- random-baseline IoU comparison
- threshold-sensitivity sweep (percentile 50–99)
- overlay visualization (original image + SAM masks + attribution heatmap)

Use this notebook when you want to interactively explore how well attribution aligns with segmented roof/building regions on individual images.

### Batch runner

#### `remoteclip_segmentation_overlap_batch.py`

Non-interactive CLI script that runs the segmentation-overlap analysis across the full holdout split with resumable execution and robust failure handling.

What it does:
- processes every holdout image through: RemoteCLIP prediction → Transformer Explainability attribution → GroundingDINO proposals → SAM refinement → overlap metrics
- exports per-image combined mask PNG and segmentation-overlay image
- writes results CSV and failures JSONL
- generates aggregate summary JSON with consistency rates and health stats
- produces histogram, scatter, boxplot, and threshold-bar plots
- supports resumable execution: reruns skip already-completed images
- continue-on-error: failures logged per-image, batch continues

```bash
# Smoke test (2 images)
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py --limit 2

# Full holdout run
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py

# Ignore previous state, recompute everything
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py --force-recompute
```

CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--split` | `holdout` | Dataset split |
| `--method` | `transformer_explainability` | Attribution method |
| `--output-dir` | `xAI_outputs/segmentation` | Output root |
| `--limit` | (all) | Max images to process |
| `--device` | auto | Device override |
| `--gdino-model-id` | `IDEA-Research/grounding-dino-base` | GroundingDINO HF model |
| `--sam-model-id` | `facebook/sam-vit-huge` | SAM HF model |
| `--grounding-text-prompt` | `building . house . rooftop . roof . structure .` | Detection prompt |
| `--gdino-box-threshold` | `0.20` | GroundingDINO box confidence |
| `--gdino-text-threshold` | `0.15` | GroundingDINO text confidence |
| `--attribution-percentile` | `80` | Percentile for binarizing attribution |
| `--log-level` | `INFO` | Logging verbosity |
| `--force-recompute` | off | Ignore resume state |

Batch output layout:

```text
xAI_outputs/segmentation/
├── masks/                  # per-image combined mask PNGs (255/0)
├── overlays/               # per-image overlay (original + green mask + contour)
├── tables/
│   ├── segmentation_overlap_results.csv
│   └── segmentation_overlap_failures.jsonl
├── summary/
│   └── segmentation_overlap_summary.json
├── plots/
│   ├── attribution_mass_inside_hist.png
│   ├── attribution_mass_outside_hist.png
│   ├── combined_iou_hist.png
│   ├── mass_inside_vs_iou_scatter.png
│   ├── metric_boxplots.png
│   ├── consistency_threshold_bars.png
│   └── ecdf_mass_inside_and_iou.png
└── logs/
    └── batch_YYYYMMDD_HHMMSS.log
```

**Resumability model:** an image counts as complete only if a successful row exists in `segmentation_overlap_results.csv` AND both `masks/<id>__combined_mask.png` and `overlays/<id>__segmentation_overlay.png` exist on disk. Missing or partial artifacts trigger recomputation on rerun.

**Per-image CSV columns:** `image_id`, `image_path`, `image_filename`, `split`, `method`, `city_name`, `predicted_class_index`, `predicted_class_label`, `predicted_probability`, `gdino_model_id`, `sam_model_id`, `grounding_text_prompt`, `gdino_box_threshold`, `gdino_text_threshold`, `gdino_num_boxes`, `sam_num_masks`, `used_full_image_fallback`, `combined_mask_area`, `attribution_percentile`, `threshold_value`, `attribution_density`, `attribution_map_height`, `attribution_map_width`, `mask_height`, `mask_width`, `attribution_mass_inside`, `attribution_mass_outside`, `combined_iou`, `inside_ratio`, `coverage_ratio`, `attribution_area`, `intersection_area`, `union_area`, `sam_mask_area`, `runtime_seconds`, `status`.

### Attribution helpers

#### `attribution_helpers/feature_attribution_aggregation.py`
Helper module for spatial aggregation of attribution heatmaps.

Current responsibilities:
- heatmap normalization
- center-crop mass statistics
- radial attribution profiles
- centroid / peak offset metrics
- 50% attribution radius metrics
- aggregate summary generation

#### `attribution_helpers/transformer_explainability.py`
Transformer attention-gradient relevance rollout for RemoteCLIP ViT-L/14. Patches visual transformer blocks to capture attention weights and gradients, builds gradient-weighted per-layer relevance matrices, rolls CLS relevance back onto the 16×16 patch grid, and upsamples to input resolution.

#### `attribution_helpers/manual_gradcam.py`
Manual GradCAM utilities for:
- ViT token-level GradCAM (penultimate transformer block)
- patch embedding GradCAM (`model.visual.conv1`)

#### `attribution_helpers/captum_integrated_gradients.py`
Captum Integrated Gradients helper functions. Supports `abs` and `positive` reduction modes.

#### `attribution_helpers/rise.py`
RISE black-box masking attribution helper functions. Runs many masked forward passes in chunks; does not require internal gradients.

#### `attribution_helpers/grounding_sam.py`
GroundingDINO + SAM pipeline helpers:
- `load_groundingdino_model()` / `load_sam_predictor()` — load HuggingFace models
- `run_groundingdino_inference()` — generate bounding-box proposals with configurable prompt/thresholds
- `run_sam_box_refinement()` — refine boxes into instance masks via SAM

#### `attribution_helpers/segmentation_iou.py`
Attribution-mask overlap metrics:
- `resize_float_map_to_shape()` / `resize_mask_to_shape()` — resize attribution or mask to target shape
- `binarize_attribution_percentile()` — threshold attribution by percentile
- `compute_attribution_mask_metrics()` — IoU, inside-ratio, coverage-ratio
- `compute_attribution_mass()` — fraction of continuous attribution mass inside/outside a mask
- `combine_instance_masks()` — OR-combine multiple SAM instance masks
- `sweep_threshold_iou()` — threshold sensitivity analysis
- `random_mask_iou_baseline()` — random-baseline IoU comparison

#### `attribution_helpers/dataset_split_helpers.py`
Utilities for reproducible dataset-split-based image selection:
- `collect_split_image_paths()` — filters metadata CSV by split (`train`/`val`/`holdout`/`all`), joins against actual files on disk
- `write_split_helper_csvs()` — exports split-filtered CSV artifacts for auditability

## Environment setup

Project instructions require using repo root virtual environment: `.venv`.

From repo root:

### 1. Create virtual environment if needed

```bash
python3.14 -m venv .venv
```

### 2. Activate virtual environment

```bash
source .venv/bin/activate
```

### 3. Install base project requirements

```bash
pip install -r requirements.txt
```

### 4. Install notebook/dev requirements on top

```bash
pip install -r requirements-dev.txt
```

Why both:
- `requirements.txt` installs broader project dependencies
- `requirements-dev.txt` adds notebook/xAI-specific tools like `marimo`, `captum`, `kagglehub`, and pinned `kagglesdk`

### 5. Set HuggingFace token (optional, suppresses warnings)

Create `.env` at repo root (excluded from git):

```
HF_TOKEN=hf_your_token_here
```

## Running the notebooks

From repo root, after activating `.venv`:

### Attribution notebook

```bash
# Interactive edit mode
.venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py

# App/run mode
.venv/bin/marimo run xAI_notebooks/remoteclip_xai_attribution_marimo.py
```

### Segmentation-overlap notebook

```bash
# Interactive edit mode
.venv/bin/marimo edit xAI_notebooks/remoteclip_segmentation_overlap_marimo.py

# App/run mode
.venv/bin/marimo run xAI_notebooks/remoteclip_segmentation_overlap_marimo.py
```

### Batch segmentation-overlap runner

```bash
# Smoke test (2 images)
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py --limit 2

# Full holdout run
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py

# Resume after interruption (automatic — just rerun)
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py

# Force recompute all
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py --force-recompute
```

## Typical workflow

From repo root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Interactive exploration
.venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py

# Segmentation overlap exploration
.venv/bin/marimo edit xAI_notebooks/remoteclip_segmentation_overlap_marimo.py

# Full holdout batch run
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py
```

## Expected assets

Notebooks and batch runner expect these assets at repo root:

| Asset | Source | Required by |
|---|---|---|
| `best_clip_model_balanced.pth` | [Kaggle](https://www.kaggle.com/datasets/doubleblindreview/xbd-roof-images) | All notebooks + batch |
| `RoofNet-Images/` | Cropped roof image directory | All notebooks + batch |
| `roofnet_metadata.csv` | Metadata with `filename` and `split` columns | All notebooks + batch |

Asset paths are configurable inside notebook config cells and via CLI arguments for the batch runner.

## Outputs

### Attribution notebook outputs

Written under `xAI_outputs/`:
- per-method attribution PNGs
- per-method-family spatial stats CSV + aggregation summary CSV
- per-method-family radial profile, center-mass, and centroid-offset plot PNGs
- split helper CSV artifacts

### Segmentation overlap outputs

Written under `xAI_outputs/segmentation/` (see batch runner section above for full layout).

## Results

Experimental results and interpretations are documented in [`RESULTS.md`](RESULTS.md).

Key findings from the attribution notebook (617 holdout images, Transformer Explainability):
- model attention is diffuse, not strongly center-biased
- median centroid offset: 0.100 (centered)
- median peak offset: 0.588 (peripheral)
- median radius for 50% mass: 0.693 (large area needed)

Segmentation-overlap results are pending full batch run.

## Notes

- notebooks use **marimo**, not Jupyter, as primary interactive environment
- helper modules are meant to keep notebook cells thinner and easier to test
- GradCAM methods are fully manual (no Captum `LayerGradCam` dependency)
- batch runner supports method selection and dataset-split filtering
- aggregation schema includes a `method` column, enabling cross-method spatial metric comparison
- notebooks contain lightweight self-install logic for some optional packages, but preferred path is still installing from `requirements.txt` and `requirements-dev.txt` first

## Troubleshooting

### Import errors
Make sure `.venv` is activated and both requirements files were installed:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Wrong Python or package set
Check interpreter:

```bash
which python
python --version
```

Expected pattern:
- python should resolve inside `.venv`

### Marimo command not found
Run marimo through repo venv directly:

```bash
.venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py
```

### KaggleHub issues
This repo currently pins:
- `kagglehub==1.0.1`
- `kagglesdk==0.1.23`

Reason: newer `kagglesdk` version previously caused import breakage for notebook workflow.

### Batch runner warnings
- **HF Hub unauthenticated requests:** set `HF_TOKEN` in `.env` at repo root
- **`torch.load` weights_only warning:** batch runner uses `weights_only=True`
- **Matplotlib deprecation:** batch runner uses `tick_labels` (not `labels`)
