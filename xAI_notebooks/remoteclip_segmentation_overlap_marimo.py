import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RemoteCLIP Roof Material XAI — Segmentation Overlap Analysis

    Plug-and-play marimo notebook for GroundingDINO + SAM segmentation overlap analysis
    on fine-tuned RemoteCLIP ViT-L/14 roof-material classifier attribution maps.

    Reuses attribution methods from the companion notebook (`remoteclip_xai_attribution_marimo.py`):
    - Transformer Explainability
    - Manual GradCAM attribution
    - Captum Integrated Gradients
    - RISE black-box masking attribution

    How notebook is organized:
    1. environment and imports
    2. labels, prompts, and preprocessing
    3. asset resolution and model loading
    4. image sampling and prediction sanity check
    5. attribution method registration (Transformer Explainability, GradCAM, IG, RISE)
    6. GroundingDINO rooftop proposals
    7. SAM mask refinement
    8. attribution-mask overlap metrics (IoU, coverage, random baseline)

    Each result section aims to answer:
    - **which image regions does each attribution method consider important?**
    - **how well does that align with roof/building segmentations?**

    Output panels show:
    - **Input:** original roof image
    - **Attribution:** normalized heatmap by itself
    - **Overlay:** heatmap overlaid on input image
    - **Segmentation:** GroundingDINO proposals + SAM masks + attribution overlap metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Project venv and optional dependencies

    **What this section does:** makes sure notebook runs from project virtual environment and can self-install notebook-only dependencies if imports are missing.

    This marimo notebook is intended to run from project venv at `../.venv`.

    From repo root, launch with:

    ```bash
    .venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py
    ```

    **Expected output:** next cell prints either:
    - missing packages being installed into `sys.executable`, or
    - confirmation that all required notebook packages already import cleanly.

    If install step runs, rerunning cell later should usually print clean "already installed" status.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and notebook config

    **What this section does:** imports core libraries, then defines one centralized typed config block used by all later cells.

    **Expected output:** device selection and inferred repository root. Config cell below then exposes all main notebook knobs in one place for interactive editing.
    """)
    return


@app.cell
def _():
    import os
    import random
    import shutil
    import tempfile
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Callable, Dict, List, Optional, Tuple

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    from attribution_helpers import batch_recovery
    from attribution_helpers import grounding_sam
    from attribution_helpers import feature_attribution_aggregation as faa
    from attribution_helpers import segmentation_iou
    from attribution_helpers import manual_gradcam
    from attribution_helpers import captum_integrated_gradients
    from attribution_helpers import dataset_split_helpers
    from attribution_helpers import rise
    from attribution_helpers.transformer_explainability import transformer_explainability

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("Install open_clip_torch: pip install open_clip_torch") from exc

    try:
        import kagglehub
    except ImportError:
        kagglehub = None

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        IntegratedGradients = None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    REPO_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "xAI_notebooks" else Path.cwd().resolve()

    print(f"Device: {DEVICE}")
    print(f"Repo root: {REPO_ROOT}")
    return (
        Callable,
        DEVICE,
        Dict,
        Image,
        IntegratedGradients,
        List,
        Optional,
        Path,
        REPO_ROOT,
        Tuple,
        captum_integrated_gradients,
        dataclass,
        dataset_split_helpers,
        grounding_sam,
        kagglehub,
        manual_gradcam,
        nn,
        np,
        open_clip,
        pd,
        plt,
        random,
        rise,
        segmentation_iou,
        torch,
        transformer_explainability,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Centralized notebook config

    **What this section does:** collects main notebook settings into one typed config object with grouped sections for assets, preprocessing, visualization, attribution, and batch defaults.

    **How to use it:** edit values in this cell, then rerun dependent cells. Notebook is designed so most routine experimentation should start here.
    """)
    return


@app.cell
def _(Optional, Path, REPO_ROOT, dataclass):
    @dataclass(frozen=True)
    class AssetConfig:
        # =========================
        # Asset source selection
        # =========================
        asset_mode: str  # "local" or "kagglehub"
        kaggle_dataset: str  # KaggleHub dataset slug used when asset_mode="kagglehub"
        local_model_weights: Path  # Local checkpoint path for fine-tuned RemoteCLIP weights
        local_image_dir: Path  # Local directory containing cropped roof images
        local_metadata_csv: Path  # Optional local metadata CSV path
        image_exts: tuple[str, ...]  # File suffixes treated as valid image assets

    @dataclass(frozen=True)
    class PreprocessConfig:
        # =========================
        # Image preprocessing
        # =========================
        image_size: tuple[int, int]  # Input resolution expected by RemoteCLIP
        clip_mean: tuple[float, float, float]  # CLIP channel means for normalization
        clip_std: tuple[float, float, float]  # CLIP channel stds for normalization

    @dataclass(frozen=True)
    class VisualizationConfig:
        # =========================
        # Attribution figure display
        # =========================
        overlay_alpha: float  # Transparency for overlay panel
        cmap: str  # Matplotlib colormap for attribution heatmaps
        figure_size: tuple[int, int]  # Figure size for three-panel attribution plots
        preview_figure_size: tuple[int, int]  # Figure size for sampled input image preview

    @dataclass(frozen=True)
    class IntegratedGradientsConfig:
        # =========================
        # Integrated Gradients
        # =========================
        n_steps: int  # Number of integration steps for Captum IG

    @dataclass(frozen=True)
    class RiseConfig:
        # =========================
        # RISE black-box attribution
        # =========================
        num_masks: int  # Number of random masks to sample
        mask_grid_size: int  # Low-resolution grid size before upsampling masks
        p_save: float  # Probability each mask cell stays visible
        batch_size: int  # Number of masked images scored per forward chunk
        return_diagnostics: bool  # Print runtime/sampling diagnostics during notebook runs

    @dataclass(frozen=True)
    class BatchConfig:
        # =========================
        # Future batch runner defaults
        # =========================
        num_images: Optional[int]  # Placeholder count for future batch processing
        split: str  # Dataset split selector for batch runs
        methods: tuple[str, ...]  # Exact attribution methods to run in batch mode
        target: str  # Attribution target policy for future batch runs
        output_dir: Path  # Directory where future batch outputs should be written
        attribution_npz_dir: Path  # Clean-slate folder for persisted batch attribution heatmaps
        helper_csv_dir: Path  # Optional directory for exported split helper CSV artifacts

    @dataclass(frozen=True)
    class NotebookConfig:
        # =========================
        # Root notebook config
        # =========================
        seed: int  # Global seed for Python/numpy/torch and stochastic methods
        model_name: str  # OpenCLIP model architecture name
        pretrained_weights: str  # Base pretrained weights used before fine-tuned checkpoint load
        assets: AssetConfig
        preprocess: PreprocessConfig
        visualization: VisualizationConfig
        integrated_gradients: IntegratedGradientsConfig
        rise: RiseConfig
        batch: BatchConfig

    CONFIG = NotebookConfig(
        seed=42,
        model_name="ViT-L-14",
        pretrained_weights="laion2b_s32b_b82k",
        assets=AssetConfig(
            asset_mode="local",
            kaggle_dataset="doubleblindreview/xbd-roof-images",
            local_model_weights=REPO_ROOT / "best_clip_model_balanced.pth",
            local_image_dir=REPO_ROOT / "RoofNet-Images",
            local_metadata_csv=REPO_ROOT / "roofnet_metadata.csv",
            image_exts=(".jpg", ".jpeg", ".png", ".webp"),
        ),
        preprocess=PreprocessConfig(
            image_size=(224, 224),
            clip_mean=(0.48145466, 0.4578275, 0.40821073),
            clip_std=(0.26862954, 0.26130258, 0.27577711),
        ),
        visualization=VisualizationConfig(
            overlay_alpha=0.45,
            cmap="inferno",
            figure_size=(12, 4),
            preview_figure_size=(4, 4),
        ),
        integrated_gradients=IntegratedGradientsConfig(
            n_steps=50,
        ),
        rise=RiseConfig(
            num_masks=512,
            mask_grid_size=12,
            p_save=0.5,
            batch_size=32,
            return_diagnostics=True,
        ),
        batch=BatchConfig(
            num_images=None,
            split="holdout",
            methods=("transformer_explainability",),
            target="predicted_top1",
            output_dir=REPO_ROOT / "xAI_outputs",
            attribution_npz_dir=REPO_ROOT / "xAI_outputs" / "attribution_npz",
            helper_csv_dir=REPO_ROOT / "xAI_notebooks",
        ),
    )

    CONFIG
    return (CONFIG,)


@app.cell
def _(CONFIG, np, random, torch):
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    print(f"Seeded python/numpy/torch with CONFIG.seed={CONFIG.seed}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Labels, prompts, and preprocessing

    **What this section does:** defines roof-material label set, derives city-conditioned text prompts from filenames, and builds CLIP-compatible image preprocessing.

    **Why this matters:** classifier prediction is not plain image-only classification; image is compared against text prompts like `{material} in {city_name}`. Small changes here affect every prediction and attribution result downstream.

    **Expected output:** no rich display yet; this block mainly prepares reusable functions and constants for later cells.
    """)
    return


@app.cell
def _(CONFIG, List, Path, torch, transforms):
    MATERIAL_CLASSES = [
        "Thatch", "StoneSlates", "ClayTiles", "AsphaltTiles",
        "ConcreteTiles", "WoodTiles", "MetalSheetMaterials", "PolycarbonateSheetMaterials",
        "GlassSheetMaterials", "AmorphousConcrete", "AmorphousAsphalt",
        "AmorphousMembrane", "AmorphousFabric", "Unknown", "GreenVegetative"
    ]

    def extract_city_name_from_filename(filename: str) -> str:
        base = Path(filename).stem
        if '-' in base:
            city_part = base.split('-')[0]
            return city_part.replace('_', ' ').title()
        if 'height' in base:
            city_part = base.split('_height')[0]
            return city_part.replace('_', ' ').title()
        if 'imsat' in base:
            city_part = base.split('_imsat')[0]
            return city_part.replace('_', ' ').title()
        return base

    def build_prompts(city_name: str) -> List[str]:
        return [f"{material} in {city_name}" for material in MATERIAL_CLASSES]

    preprocess = transforms.Compose([
        transforms.Resize(CONFIG.preprocess.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CONFIG.preprocess.clip_mean),
                             std=list(CONFIG.preprocess.clip_std)),
    ])

    def denormalize_clip_tensor(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(CONFIG.preprocess.clip_mean, device=x.device).view(3, 1, 1)
        std = torch.tensor(CONFIG.preprocess.clip_std, device=x.device).view(3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    return (
        MATERIAL_CLASSES,
        build_prompts,
        extract_city_name_from_filename,
        preprocess,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Asset resolver: local or KaggleHub

    **What this section does:** resolves where notebook should find model weights, image directory, and optional metadata.

    Use this section when switching between:
    - **local mode:** assets already present inside repo/workstation layout
    - **KaggleHub mode:** assets downloaded on demand from dataset mirror

    **Expected output:** resolved filesystem paths plus small diagnostics showing which asset source was selected. If this section fails, later model-loading and inference cells will also fail.
    """)
    return


@app.cell
def _(CONFIG, List, Optional, Path, REPO_ROOT, dataclass, kagglehub):
    @dataclass
    class AssetPaths:
        root: Path
        model_weights: Path
        image_dir: Path
        metadata_csv: Optional[Path] = None

    def _find_first(root: Path, patterns: List[str]) -> Optional[Path]:
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
        return None

    def resolve_assets(asset_mode: str = CONFIG.assets.asset_mode) -> AssetPaths:
        if asset_mode == "local":
            assets = AssetPaths(
                root=REPO_ROOT,
                model_weights=CONFIG.assets.local_model_weights,
                image_dir=CONFIG.assets.local_image_dir,
                metadata_csv=CONFIG.assets.local_metadata_csv if CONFIG.assets.local_metadata_csv.exists() else None,
            )
        elif asset_mode == "kagglehub":
            if kagglehub is None:
                raise ImportError("Install kagglehub: pip install kagglehub")
            kaggle_root = Path(kagglehub.dataset_download(CONFIG.assets.kaggle_dataset)).resolve()
            print("Path to dataset files:", kaggle_root)
            model_weights = _find_first(kaggle_root, ["best_clip_model_balanced.pth", "*.pth"])
            image_dir = _find_first(kaggle_root, ["xBD_cropped_roofs", "*cropped*roofs*"])
            metadata_csv = _find_first(kaggle_root, ["roofnet_metadata.csv", "*.csv"])
            if model_weights is None:
                raise FileNotFoundError(f"Could not find .pth weights under {kaggle_root}")
            if image_dir is None or not image_dir.is_dir():
                # fallback: use parent directory of first image
                first_image = next((p for p in kaggle_root.rglob("*") if p.suffix.lower() in CONFIG.assets.image_exts), None)
                if first_image is None:
                    raise FileNotFoundError(f"Could not find image files under {kaggle_root}")
                image_dir = first_image.parent
            assets = AssetPaths(kaggle_root, model_weights, image_dir, metadata_csv)
        else:
            raise ValueError("asset_mode must be 'local' or 'kagglehub'")

        if not assets.model_weights.exists():
            raise FileNotFoundError(f"Missing model weights: {assets.model_weights}")
        if not assets.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {assets.image_dir}")
        return assets

    assets = resolve_assets(CONFIG.assets.asset_mode)
    assets
    return (assets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Load fine-tuned RemoteCLIP

    **What this section does:** instantiates base RemoteCLIP ViT-L/14 architecture, loads fine-tuned roof-material checkpoint, moves model to active device, and fetches matching tokenizer.

    **Expected output:** path to weight file that was loaded. This is important provenance signal: attribution is only meaningful if notebook is pointing at intended checkpoint.
    """)
    return


@app.cell
def _(CONFIG, DEVICE, Path, assets, open_clip, torch):
    def load_remoteclip_model(weights_path: Path, device: str = DEVICE):
        model, _, _ = open_clip.create_model_and_transforms(CONFIG.model_name, pretrained=CONFIG.pretrained_weights)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(CONFIG.model_name)
        return model, tokenizer

    model, tokenizer = load_remoteclip_model(assets.model_weights)
    print("Loaded model from:", assets.model_weights)
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Image selection and inference wrapper

    **What this section does:** finds candidate roof images, samples one example, preprocesses it, builds city-aware prompts, and runs top-k prediction sanity check.

    **Expected output:**
    - number of images discovered under active image directory
    - sampled filename
    - parsed city context
    - top-5 class probabilities for sampled image
    - displayed input image titled with top-1 predicted material

    Treat this as main pre-attribution checkpoint. If predicted label or prompt context looks wrong here, attribution maps later may still render but answer wrong question.
    """)
    return


@app.cell
def _(
    CONFIG,
    DEVICE,
    Dict,
    Image,
    List,
    MATERIAL_CLASSES,
    Optional,
    Path,
    Tuple,
    assets,
    build_prompts,
    dataset_split_helpers,
    extract_city_name_from_filename,
    model,
    pd,
    plt,
    preprocess,
    random,
    tokenizer,
    torch,
):
    def list_images(image_dir: Path, limit: Optional[int] = None) -> List[Path]:
        paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in CONFIG.assets.image_exts)
        return paths[:limit] if limit else paths

    def load_image_tensor(image_path: Path, device: str = DEVICE) -> Tuple[Image.Image, torch.Tensor]:
        pil = Image.open(image_path).convert("RGB")
        tensor = preprocess(pil).unsqueeze(0).to(device)
        return pil, tensor

    @torch.no_grad()
    def predict(image_tensor: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        tokenized = tokenizer(prompts).to(image_tensor.device)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokenized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        return {"logits": logits.squeeze(0), "probs": probs.squeeze(0)}

    def target_score_forward(image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> torch.Tensor:
        # Captum-compatible scalar target score. Keep gradients enabled for image path.
        tokenized = tokenizer(prompts).to(image_tensor.device)
        image_features = model.encode_image(image_tensor)
        with torch.no_grad():
            text_features = model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        return logits[:, target_idx]

    all_images = list_images(assets.image_dir)
    print(f"Found {len(all_images)} images under {assets.image_dir}")

    split_diagnostics = None
    helper_csv_outputs = {}
    if assets.metadata_csv is not None and assets.metadata_csv.exists():
        metadata_df = pd.read_csv(assets.metadata_csv, low_memory=False)
        print(f"Loaded metadata from: {assets.metadata_csv}")
        helper_csv_outputs = dataset_split_helpers.write_split_helper_csvs(
            metadata_df,
            output_dir=CONFIG.batch.helper_csv_dir,
        )
        images, split_diagnostics = dataset_split_helpers.collect_split_image_paths(
            image_dir=assets.image_dir,
            metadata_df=metadata_df,
            split=CONFIG.batch.split,
            image_exts=CONFIG.assets.image_exts,
        )
    else:
        print("No metadata CSV found. Falling back to unfiltered image discovery.")
        images = all_images

    if not images:
        raise ValueError(f"No images available after applying split filter {CONFIG.batch.split!r}.")

    print(f"Using {len(images)} images for split={CONFIG.batch.split!r}")
    sample_path = random.choice(images)
    city_name = extract_city_name_from_filename(sample_path.name)
    prompts = build_prompts(city_name)
    pil_img, image_tensor = load_image_tensor(sample_path)

    pred = predict(image_tensor, prompts)
    topk = torch.topk(pred["probs"], k=5)
    print("Sample:", sample_path.name)
    print("City prompt context:", city_name)
    for score, class_idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"{MATERIAL_CLASSES[class_idx]:>28s}: {score:.3f}")

    plt.figure(figsize=CONFIG.visualization.preview_figure_size)
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title(MATERIAL_CLASSES[topk.indices[0].item()])
    return (
        image_tensor,
        pil_img,
        prompts,
        sample_path,
        target_score_forward,
        topk,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Attribution method registry

    Each method accepts `(image_tensor, target_idx, prompts)` and returns a 2D heatmap array. Add/replace methods without changing visualization code.
    """)
    return


@app.cell
def _(Callable, Dict, List, np, torch):
    AttributionFn = Callable[[torch.Tensor, int, List[str]], np.ndarray]
    ATTRIBUTION_METHODS: Dict[str, AttributionFn] = {}

    def register_attribution(name: str):
        def decorator(fn: AttributionFn):
            ATTRIBUTION_METHODS[name] = fn
            return fn
        return decorator

    return ATTRIBUTION_METHODS, register_attribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Transformer Explainability

    **What this section does:** registers attention-based attribution method for RemoteCLIP ViT-L/14.

    Intuition:
    - inspect visual transformer attention at each layer
    - weight attention by gradient signal from chosen target score
    - roll relevance from CLS token back onto spatial patch grid

    **Expected output later:** a 2D relevance map showing which patches most supported current top-1 image-text similarity score.

    Notes:
    - patches each visual transformer attention block to request `need_weights=True`
    - captures attention weights plus their gradients with respect to target image-text similarity
    - builds gradient-weighted per-layer relevance matrices
    - rolls CLS relevance back onto 16×16 patch grid, then upsamples to 224×224
    """)
    return


@app.cell
def _(
    CONFIG,
    List,
    model,
    np,
    register_attribution,
    tokenizer,
    torch,
    transformer_explainability,
):
    @register_attribution("transformer_explainability")
    def transformer_explainability_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        return transformer_explainability(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            prompts=prompts,
            target_idx=target_idx,
            image_size=CONFIG.preprocess.image_size,
            verbose=verbose,
        )

    TRANSFORMER_EXPLAINABILITY_REGISTERED = True
    return (TRANSFORMER_EXPLAINABILITY_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Manual GradCAM methods

    **What this section does:** registers two manual GradCAM-style methods that probe different stages of visual encoder.

    GradCAM variants:
    - `vit_token_gradcam`: penultimate visual transformer block, CLS dropped, patch tokens reshaped to grid
    - `manual_patch_gradcam`: raw patch projection layer `model.visual.conv1`

    Interpretation guide:
    - **ViT-token GradCAM:** later, more semantic focus after transformer processing
    - **Patch-embed GradCAM:** earlier, lower-level spatial evidence near image-to-patch projection stage
    """)
    return


@app.cell
def _(
    List,
    manual_gradcam,
    model,
    nn,
    np,
    register_attribution,
    target_score_forward,
    torch,
):
    class TargetScoreModule(nn.Module):
        def __init__(self, prompts: List[str], target_idx: int):
            super().__init__()
            self.prompts = prompts
            self.target_idx = target_idx

        def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
            return target_score_forward(image_tensor, self.target_idx, self.prompts)

    def inspect_visual_layers(model) -> None:
        for name, module in model.visual.named_modules():
            if any(key in name.lower() for key in ["block", "resblock", "attn", "ln_post"]):
                print(name, "->", module.__class__.__name__)

    @register_attribution("vit_token_gradcam")
    def vit_token_gradcam_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        score_module = TargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return manual_gradcam.vit_token_gradcam_heatmap(
                model=model,
                score_forward=score_module,
                image_tensor=image_tensor,
                verbose=verbose,
            )
        finally:
            model.zero_grad(set_to_none=True)

    @register_attribution("manual_patch_gradcam")
    def manual_patch_gradcam_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        score_module = TargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return manual_gradcam.manual_patch_gradcam_heatmap(
                model=model,
                score_forward=score_module,
                image_tensor=image_tensor,
                verbose=verbose,
            )
        finally:
            model.zero_grad(set_to_none=True)

    MANUAL_GRADCAM_METHODS_REGISTERED = True
    return (MANUAL_GRADCAM_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Captum Integrated Gradients

    **What this section does:** registers input-space attribution methods based on path-integrated gradients from zero baseline to actual image tensor.

    Integrated Gradients variants:
    - `captum_integrated_gradients_abs`: absolute channel-sum input attribution
    - `captum_integrated_gradients_positive`: positive-only channel-sum input attribution

    Both use zero baseline, `n_steps=50`, and print Captum convergence delta.

    **Expected output later:** heatmaps often look smoother than raw gradient methods. If absolute and positive variants look very similar, that usually means negative attribution mass was small for this sample.
    """)
    return


@app.cell
def _(
    CONFIG,
    IntegratedGradients,
    List,
    captum_integrated_gradients,
    model,
    nn,
    np,
    register_attribution,
    target_score_forward,
    torch,
):
    class IntegratedGradientsTargetScoreModule(nn.Module):
        def __init__(self, prompts: List[str], target_idx: int):
            super().__init__()
            self.prompts = prompts
            self.target_idx = target_idx

        def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
            return target_score_forward(image_tensor, self.target_idx, self.prompts)

    def _integrated_gradients_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        reduction: str,
        verbose: bool = True,
    ) -> np.ndarray:
        if IntegratedGradients is None:
            raise ImportError("Install captum: pip install captum")
        score_module = IntegratedGradientsTargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return captum_integrated_gradients.integrated_gradients_heatmap(
                score_forward=score_module,
                image_tensor=image_tensor,
                integrated_gradients_cls=IntegratedGradients,
                reduction=reduction,
                n_steps=CONFIG.integrated_gradients.n_steps,
                verbose=verbose,
            )
        finally:
            model.zero_grad(set_to_none=True)

    @register_attribution("captum_integrated_gradients_abs")
    def captum_integrated_gradients_abs_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        return _integrated_gradients_attr(image_tensor, target_idx, prompts, reduction="abs", verbose=verbose)

    @register_attribution("captum_integrated_gradients_positive")
    def captum_integrated_gradients_positive_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        return _integrated_gradients_attr(image_tensor, target_idx, prompts, reduction="positive", verbose=verbose)

    CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED = True
    return (CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. RISE raw-image black-box attribution

    **What this section does:** registers black-box attribution method that repeatedly masks image and measures how target score changes.

    `rise_raw_image` uses randomized input sampling with mean-baseline masks in CLIP-normalized tensor space. It runs many masked RemoteCLIP forwards in chunks, so expect it to be slower than gradient/attention methods.

    **Expected output later:** spatial saliency map plus concise diagnostics about mask sampling and runtime. Good contrast against gradient-based methods because it does not rely on internal gradients through model layers.
    """)
    return


@app.cell
def _(CONFIG, List, model, np, register_attribution, rise, tokenizer, torch):
    def rise_unscaled_target_score(image_batch: torch.Tensor, target_idx: int, prompts: List[str]) -> torch.Tensor:
        tokenized = tokenizer(prompts).to(image_batch.device)
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(tokenized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features @ text_features.T[:, target_idx]

    @register_attribution("rise_raw_image")
    def rise_raw_image_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        generator_device = image_tensor.device if image_tensor.device.type != "mps" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(CONFIG.seed)
        model.eval()
        return rise.rise_heatmap(
            score_forward=lambda masked_batch: rise_unscaled_target_score(masked_batch, target_idx, prompts),
            image_tensor=image_tensor,
            num_masks=CONFIG.rise.num_masks,
            mask_grid_size=CONFIG.rise.mask_grid_size,
            p_save=CONFIG.rise.p_save,
            batch_size=CONFIG.rise.batch_size,
            mask_device=image_tensor.device,
            return_diagnostics=CONFIG.rise.return_diagnostics,
            generator=generator,
            verbose=verbose,
        )[0]

    RISE_METHODS_REGISTERED = True
    return (RISE_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Selected-image segmentation overlap

    **What this section does:** adds optional Hugging Face GroundingDINO proposal generation, SAM box refinement, and attribution-versus-mask overlap analysis for current sampled image.

    **Workflow:**
    1. configure GroundingDINO and SAM model IDs plus thresholds
    2. inspect dependency and model-loading diagnostics
    3. generate GroundingDINO rooftop/building proposals
    4. refine proposals into SAM masks
    5. compare one selected attribution heatmap against those masks

    **Expected output:** warnings if optional segmentation dependencies are missing, then proposal overlays, SAM-mask diagnostics, IoU tables, threshold-sensitivity plots, and random-baseline summary.
    """)
    return


@app.cell
def _(
    ATTRIBUTION_METHODS: "Dict[str, AttributionFn]",
    CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
    MANUAL_GRADCAM_METHODS_REGISTERED,
    RISE_METHODS_REGISTERED,
    TRANSFORMER_EXPLAINABILITY_REGISTERED,
    mo,
):
    _ = (
        MANUAL_GRADCAM_METHODS_REGISTERED,
        CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
        RISE_METHODS_REGISTERED,
        TRANSFORMER_EXPLAINABILITY_REGISTERED,
    )

    segmentation_controls = {
        "gdino_model_id": mo.ui.text(label="GroundingDINO model ID", value="IDEA-Research/grounding-dino-base"),
        "sam_model_id": mo.ui.text(label="SAM model ID", value="facebook/sam-vit-huge"),
        "grounding_text_prompt": mo.ui.text(label="Grounding text prompt", value="building . house . rooftop . roof . structure ."),
        "gdino_box_threshold": mo.ui.slider(0.05, 0.9, value=0.20, step=0.05, label="GroundingDINO box threshold"),
        "gdino_text_threshold": mo.ui.slider(0.05, 0.9, value=0.15, step=0.05, label="GroundingDINO text threshold"),
        "attribution_iou_threshold_pct": mo.ui.slider(50, 99, value=80, step=1, label="Attribution percentile"),
        "random_baseline_samples": mo.ui.number(start=1, stop=1000, value=100, label="Random baseline samples"),
        "segmentation_method_name": mo.ui.dropdown(options=sorted(ATTRIBUTION_METHODS), value="transformer_explainability", label="Attribution method"),
    }
    mo.vstack([
        mo.md("## 14. Selected-image segmentation overlap"),
        mo.md("Configure GroundingDINO, SAM, and attribution-mask overlap settings."),
        *segmentation_controls.values(),
    ])
    return (segmentation_controls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 14.1 Segmentation controls and dependency diagnostics

    **What this section does:** exposes Hugging Face model IDs, prompts, thresholds, and attribution-method selection for overlap analysis.

    **How to read it:**
    - blank GroundingDINO/SAM model IDs keep model loading disabled
    - dependency messages below tell you whether `transformers` exposes the required classes
    - warning panels are non-fatal; attribution notebook can still run without segmentation extras or successful model downloads
    """)
    return


@app.cell
def _(grounding_sam):
    dependency_status = grounding_sam.check_grounding_sam_dependencies()
    for name, status in dependency_status.items():
        print(f"{name}: {status.message}")
    return (dependency_status,)


@app.cell
def _(DEVICE, grounding_sam, mo, segmentation_controls):
    gdino_model = None
    gdino_warning = None
    try:
        gdino_model_id = segmentation_controls["gdino_model_id"].value.strip()
        if not gdino_model_id:
            gdino_warning = "Set a GroundingDINO Hugging Face model ID to enable proposal generation."
        else:
            gdino_model = grounding_sam.load_groundingdino_model(
                gdino_model_id,
                device=DEVICE,
            )
            print(f"Loaded GroundingDINO model from {gdino_model_id}")
    except Exception as exc:
        gdino_warning = str(exc)
        print(f"GroundingDINO load failed: {exc!r}")
    gdino_status = None
    if gdino_warning:
        gdino_status = mo.md(f"**GroundingDINO warning:** {gdino_warning}")
    return gdino_model, gdino_warning


@app.cell
def _(DEVICE, grounding_sam, mo, segmentation_controls):
    sam_predictor = None
    sam_warning = None
    try:
        sam_model_id = segmentation_controls["sam_model_id"].value.strip()
        if not sam_model_id:
            sam_warning = "Set a SAM Hugging Face model ID to enable mask refinement."
        else:
            sam_predictor = grounding_sam.load_sam_predictor(
                sam_model_id,
                device=DEVICE,
            )
            print(f"Loaded SAM model from {sam_model_id}")
    except Exception as exc:
        sam_warning = str(exc)
        print(f"SAM load failed: {exc!r}")
    sam_status = None
    if sam_warning:
        sam_status = mo.md(f"**SAM warning:** {sam_warning}")
    return sam_predictor, sam_warning


@app.cell
def _(dependency_status, mo):
    warning_blocks = []
    if not dependency_status["groundingdino"].ok:
        warning_blocks.append(mo.md(f"**GroundingDINO warning:** {dependency_status['groundingdino'].message}"))
    if not dependency_status["segment_anything"].ok:
        warning_blocks.append(mo.md(f"**SAM warning:** {dependency_status['segment_anything'].message}"))
    if warning_blocks:
        mo.vstack(warning_blocks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 14.2 GroundingDINO proposals and SAM mask refinement

    **What this section does:** runs GroundingDINO on current sampled image using configurable rooftop/building prompt, then refines each detected box with SAM.

    **Expected output:**
    - printed image name, prompt, and GroundingDINO box count
    - box overlay figure for current image
    - SAM mask count and per-mask pixel areas

    If no boxes appear, adjust text prompt or thresholds before interpreting overlap metrics downstream.
    """)
    return


@app.cell
def _(pil_img, sample_path, segmentation_controls):
    configured_gdino_model_id = segmentation_controls["gdino_model_id"].value
    configured_grounding_text_prompt = segmentation_controls["grounding_text_prompt"].value
    configured_gdino_box_threshold = segmentation_controls["gdino_box_threshold"].value
    configured_gdino_text_threshold = segmentation_controls["gdino_text_threshold"].value
    print(f"Selected image: {sample_path.name}")
    print(f"Input image size for GroundingDINO/SAM: {pil_img.size[0]}x{pil_img.size[1]}")
    print(f"GroundingDINO config model_id={configured_gdino_model_id!r}")
    print(f"GroundingDINO config prompt={configured_grounding_text_prompt!r}")
    print(f"GroundingDINO config box_threshold={configured_gdino_box_threshold}")
    print(f"GroundingDINO config text_threshold={configured_gdino_text_threshold}")
    return


@app.cell
def _(
    gdino_model,
    gdino_warning,
    grounding_sam,
    pil_img,
    sample_path,
    segmentation_controls,
):
    grounding_prediction = None
    if gdino_model is not None and not gdino_warning:
        prompt = segmentation_controls["grounding_text_prompt"].value
        box_threshold = segmentation_controls["gdino_box_threshold"].value
        text_threshold = segmentation_controls["gdino_text_threshold"].value
        grounding_prediction = grounding_sam.run_groundingdino_inference(
            gdino_model,
            pil_img,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        print(f"Selected image: {sample_path.name}")
        print(f"GroundingDINO model ID: {segmentation_controls['gdino_model_id'].value}")
        print(f"GroundingDINO image size: {pil_img.size[0]}x{pil_img.size[1]}")
        print(f"GroundingDINO prompt: {prompt!r}")
        print(f"GroundingDINO thresholds: box={box_threshold}, text={text_threshold}")
        print(f"GroundingDINO boxes: {len(grounding_prediction.boxes_xyxy)}")
        print(f"GroundingDINO phrases: {grounding_prediction.phrases}")
        print(f"GroundingDINO confidences: {[float(score) for score in grounding_prediction.confidences]}")
        print(f"GroundingDINO boxes_xyxy: {grounding_prediction.boxes_xyxy.tolist()}")
        if len(grounding_prediction.boxes_xyxy) == 0:
            print("GroundingDINO diagnostic: zero detections. SAM will use full-image fallback box, which often yields near-full-image masks on small inputs.")
    else:
        print("Skipping GroundingDINO inference because model loading is disabled or failed.")
    return (grounding_prediction,)


@app.cell
def _(grounding_prediction, mo, pil_img, plt):
    gdino_proposals_output = None
    if grounding_prediction is None:
        gdino_proposals_output = mo.md("**GroundingDINO proposals unavailable until model loading succeeds.**")
    elif len(grounding_prediction.boxes_xyxy) == 0:
        gd_width, gd_height = pil_img.size
        print(f"GroundingDINO boxes: 0")
        print(f"Using full-image fallback box for SAM.")
        print(f"Fallback box: [0, 0, {gd_width}, {gd_height}]")
        gdino_proposals_output = mo.md("**No GroundingDINO boxes detected for current image. Will fall back to full-image SAM.**")
    else:
        gd_fig, gd_ax = plt.subplots(figsize=(6, 6))
        gd_ax.imshow(pil_img)
        for gd_proposal_idx, (gd_box, gd_conf) in enumerate(zip(grounding_prediction.boxes_xyxy, grounding_prediction.confidences)):
            gd_x0, gd_y0, gd_x1, gd_y1 = gd_box
            gd_rect = plt.Rectangle((gd_x0, gd_y0), gd_x1 - gd_x0, gd_y1 - gd_y0, fill=False, edgecolor="cyan", linewidth=2)
            gd_ax.add_patch(gd_rect)
            gd_ax.text(gd_x0, gd_y0, f"{gd_proposal_idx}: {gd_conf:.3f}", color="white", bbox={"facecolor": "black", "alpha": 0.6})
        gd_ax.axis("off")
        gd_ax.set_title("GroundingDINO rooftop proposals")
        plt.close(gd_fig)
        gdino_proposals_output = gd_fig
    gdino_proposals_output


@app.cell
def _(
    grounding_prediction,
    grounding_sam,
    mo,
    np,
    pil_img,
    plt,
    sam_predictor,
    sam_warning,
    segmentation_controls,
):
    sam_prediction = None
    sam_overlay_output = None
    sam_overlay_warning = sam_warning
    gate_reason = None
    fallback_active = False
    boxes_for_sam = None
    if grounding_prediction is None:
        gate_reason = "GroundingDINO prediction unavailable."
    elif len(grounding_prediction.boxes_xyxy) == 0:
        sam_width, sam_height = pil_img.size
        fallback_active = True
        boxes_for_sam = [[0.0, 0.0, float(sam_width), float(sam_height)]]
        print(f"GroundingDINO boxes: 0 \u2014 using full-image fallback box for SAM.")
        print(f"Fallback box: [0, 0, {sam_width}, {sam_height}]")
    elif sam_predictor is None:
        gate_reason = "SAM predictor not loaded."
    elif sam_warning:
        gate_reason = f"SAM loader warning present: {sam_warning}"

    if gate_reason is None:
        try:
            effective_boxes = boxes_for_sam if fallback_active else grounding_prediction.boxes_xyxy
            sam_prediction = grounding_sam.run_sam_box_refinement(sam_predictor, pil_img, effective_boxes)
            print(f"SAM model ID: {segmentation_controls['sam_model_id'].value}")
            print(f"SAM masks: {len(sam_prediction.masks)}")
            print(f"SAM mask areas: {[int(mask.sum()) for mask in sam_prediction.masks]}")
        except Exception as exc:
            sam_overlay_warning = f"SAM refinement failed: {exc}"
            print(sam_overlay_warning)
    else:
        print(f"Skipping SAM mask refinement: {gate_reason}")
    if sam_prediction is None:
        if sam_overlay_warning:
            sam_overlay_output = mo.md(f"**SAM overlay unavailable:** {sam_overlay_warning}")
        else:
            sam_overlay_output = mo.md("**SAM overlay unavailable.**")
    else:
        sam_fig, sam_axes = plt.subplots(1, 2, figsize=(12, 6))
        sam_input_ax, sam_overlay_ax = sam_axes
        sam_input_ax.imshow(pil_img)
        sam_input_ax.axis("off")
        sam_input_ax.set_title("Original image")

        sam_overlay_ax.imshow(pil_img)
        sam_cmap = plt.get_cmap("spring")
        if fallback_active:
            sam_x0, sam_y0, sam_x1, sam_y1 = boxes_for_sam[0]
            sam_rect = plt.Rectangle((sam_x0, sam_y0), sam_x1 - sam_x0, sam_y1 - sam_y0, fill=False, edgecolor="cyan", linewidth=2, linestyle="--")
            sam_overlay_ax.add_patch(sam_rect)
            sam_overlay_ax.text(sam_x0, sam_y0, "fallback", color="white", bbox={"facecolor": "black", "alpha": 0.6})
        else:
            for sam_proposal_idx, (sam_box, sam_conf) in enumerate(zip(grounding_prediction.boxes_xyxy, grounding_prediction.confidences)):
                sam_x0, sam_y0, sam_x1, sam_y1 = sam_box
                sam_rect = plt.Rectangle((sam_x0, sam_y0), sam_x1 - sam_x0, sam_y1 - sam_y0, fill=False, edgecolor="cyan", linewidth=2)
                sam_overlay_ax.add_patch(sam_rect)
                sam_overlay_ax.text(sam_x0, sam_y0, f"{sam_proposal_idx}: {sam_conf:.3f}", color="white", bbox={"facecolor": "black", "alpha": 0.6})
        for mask_idx, mask in enumerate(sam_prediction.masks):
            masked_overlay = np.ma.masked_where(~mask, mask)
            sam_overlay_ax.imshow(masked_overlay, cmap=sam_cmap, alpha=0.28 + 0.08 * (mask_idx % 3))
        sam_overlay_ax.axis("off")
        sam_overlay_ax.set_title("Full-image fallback box + SAM mask" if fallback_active else "GroundingDINO proposals + SAM masks")
        sam_fig.tight_layout()
        plt.close(sam_fig)
        sam_overlay_output = sam_fig
    sam_overlay_output
    return (sam_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 14.3 Attribution-mask overlap metrics

    **What this section does:** computes one attribution heatmap for current top-1 prediction, resizes it into SAM-mask space, binarizes by percentile, and scores overlap.

    **Metrics reported:**
    - **attribution_iou:** intersection over union between attribution mask and SAM mask
    - **inside_ratio:** fraction of attribution pixels that fall inside mask
    - **coverage_ratio:** fraction of SAM mask covered by attribution pixels
    - **random baseline:** IoU expected from random masks at same attribution density
    - **threshold sensitivity:** how IoU changes as attribution percentile moves from 50 to 99
    """)
    return


@app.cell
def _(
    ATTRIBUTION_METHODS: "Dict[str, AttributionFn]",
    image_tensor,
    np,
    prompts,
    segmentation_controls,
    topk,
):
    method_name = segmentation_controls["segmentation_method_name"].value
    target_idx = int(topk.indices[0].item())
    attribution_heatmap = ATTRIBUTION_METHODS[method_name](image_tensor, target_idx, prompts, verbose=False)
    print(f"Selected attribution method: {method_name}")
    print(f"Raw attribution shape: {np.asarray(attribution_heatmap).shape}")
    return (attribution_heatmap,)


@app.cell
def _(
    attribution_heatmap,
    grounding_prediction,
    sam_prediction,
    segmentation_controls,
    segmentation_iou,
):
    attribution_analysis = None
    if sam_prediction is not None and sam_prediction.masks:
        target_shape = sam_prediction.masks[0].shape
        resized_map = segmentation_iou.resize_float_map_to_shape(attribution_heatmap, target_shape)
        binary_mask, threshold_value = segmentation_iou.binarize_attribution_percentile(
            resized_map,
            segmentation_controls["attribution_iou_threshold_pct"].value,
        )
        combined_mask = segmentation_iou.combine_instance_masks(sam_prediction.masks)
        instance_rows = segmentation_iou.compute_instance_iou_table(
            binary_mask,
            sam_prediction.masks,
            boxes=getattr(grounding_prediction, "boxes_xyxy", None),
            confidences=getattr(grounding_prediction, "confidences", None),
        )
        combined_metrics = segmentation_iou.compute_attribution_mask_metrics(binary_mask, combined_mask)
        attribution_density = float(binary_mask.mean())
        attribution_analysis = {
            "resized_map": resized_map,
            "binary_mask": binary_mask,
            "threshold_value": threshold_value,
            "combined_mask": combined_mask,
            "instance_rows": instance_rows,
            "combined_row": {
                "instance_id": "combined",
                "gdino_confidence": None,
                "box_area": None,
                "sam_mask_area": combined_metrics.sam_mask_area,
                "attribution_area": combined_metrics.attribution_area,
                "intersection_area": combined_metrics.intersection_area,
                "union_area": combined_metrics.union_area,
                "attribution_iou": combined_metrics.attribution_iou,
                "inside_ratio": combined_metrics.inside_ratio,
                "coverage_ratio": combined_metrics.coverage_ratio,
            },
            "attribution_density": attribution_density,
        }
        print(f"Resized attribution shape: {resized_map.shape}")
        print(f"Threshold value: {threshold_value:.6f}")
        print(f"Attribution density: {attribution_density:.6f}")
    else:
        print("Skipping overlap analysis because no SAM masks are available.")
    return (attribution_analysis,)


@app.cell
def _(attribution_analysis, mo, pd):
    iou_table_output = None
    if attribution_analysis is None:
        iou_table_output = mo.md("**IoU analysis unavailable until SAM masks are generated.**")
    else:
        iou_table_output = pd.DataFrame(
            attribution_analysis["instance_rows"] + [attribution_analysis["combined_row"]]
        )
    iou_table_output


@app.cell
def _(attribution_analysis, segmentation_controls, segmentation_iou):
    threshold_rows = None
    baseline_summary = None
    if attribution_analysis is not None:
        threshold_rows = segmentation_iou.sweep_threshold_iou(
            attribution_analysis["resized_map"],
            [attribution_analysis["combined_mask"]],
            list(range(50, 100)),
        )
        baseline = segmentation_iou.random_mask_iou_baseline(
            [attribution_analysis["combined_mask"]],
            attribution_analysis["attribution_density"],
            attribution_analysis["combined_mask"].shape,
            n_samples=int(segmentation_controls["random_baseline_samples"].value),
            seed=42,
        )
        actual_iou = attribution_analysis["combined_row"]["attribution_iou"]
        random_iou_std = float(baseline["std_iou"])
        random_iou_z = float((actual_iou - baseline["mean_iou"]) / random_iou_std) if random_iou_std > 0 else 0.0
        baseline_summary = {
            "actual_iou": actual_iou,
            "random_iou_mean": float(baseline["mean_iou"]),
            "random_iou_std": random_iou_std,
            "random_iou_z": random_iou_z,
        }
        print(f"Baseline summary: {baseline_summary}")
    else:
        print("Skipping threshold sweep and random baseline because overlap analysis is unavailable.")
    return baseline_summary, threshold_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 14.4 Overlays and sensitivity plots

    **How to read these outputs:**
    - overlay panel compares binary attribution support against combined SAM footprint
    - baseline table asks whether observed IoU beats random placement at same attribution density
    - threshold curve checks whether overlap is stable or brittle to binarization cutoff

    Strong evidence usually means actual IoU stays above random baseline across a reasonable threshold band, not just at one percentile.
    """)
    return


@app.cell
def _(CONFIG, attribution_analysis, baseline_summary, mo, np, pil_img, plt):
    overlay_fig_output = None
    if attribution_analysis is None:
        overlay_fig_output = mo.md("**Binary attribution and SAM overlay unavailable.**")
    else:
        ov_fig, ov_axes = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: original
        ov_axes[0, 0].imshow(pil_img)
        ov_axes[0, 0].set_title("Original image")

        # Top-right: binary attribution mask
        ov_axes[0, 1].imshow(attribution_analysis["binary_mask"], cmap="gray")
        ov_axes[0, 1].set_title("Binary attribution mask")

        # Bottom-left: overlaid image (attribution heatmap on original)
        ov_axes[1, 0].imshow(pil_img)
        ov_axes[1, 0].imshow(
            attribution_analysis["resized_map"],
            cmap=CONFIG.visualization.cmap,
            alpha=CONFIG.visualization.overlay_alpha,
        )
        ov_axes[1, 0].set_title("Attribution overlay")

        # Bottom-right: SAM mask overlay on original
        ov_axes[1, 1].imshow(pil_img)
        ov_combined_mask = np.ma.masked_where(
            ~attribution_analysis["combined_mask"], attribution_analysis["combined_mask"]
        )
        ov_axes[1, 1].imshow(ov_combined_mask, cmap="spring", alpha=0.45)
        overlay_title = "SAM overlap"
        if baseline_summary is not None:
            overlay_title = f"SAM overlap | IoU={baseline_summary['actual_iou']:.3f}"
        ov_axes[1, 1].set_title(overlay_title)

        for ov_ax in ov_axes.flat:
            ov_ax.axis("off")
        ov_fig.tight_layout()
        plt.close(ov_fig)
        overlay_fig_output = ov_fig
    overlay_fig_output


@app.cell
def _(baseline_summary, mo, pd):
    baseline_table_output = None
    if baseline_summary is None:
        baseline_table_output = mo.md("**Random baseline unavailable until overlap analysis runs.**")
    else:
        baseline_table_output = pd.DataFrame([baseline_summary])
    baseline_table_output


@app.cell
def _(mo, pd, plt, threshold_rows):
    threshold_df = None
    threshold_fig_output = None
    if not threshold_rows:
        threshold_fig_output = mo.md("**Threshold sensitivity unavailable until overlap analysis runs.**")
    else:
        thresh_df = pd.DataFrame(threshold_rows)
        thresh_fig, thresh_ax = plt.subplots(figsize=(7, 4))
        thresh_ax.plot(thresh_df["percentile"], thresh_df["combined_iou"], label="combined IoU")
        thresh_ax.plot(thresh_df["percentile"], thresh_df["mean_instance_iou"], label="mean instance IoU")
        thresh_ax.set_xlabel("Attribution percentile")
        thresh_ax.set_ylabel("IoU")
        thresh_ax.set_title("Threshold sensitivity")
        thresh_ax.legend()
        thresh_fig.tight_layout()
        plt.close(thresh_fig)
        threshold_fig_output = thresh_fig
    threshold_fig_output


if __name__ == "__main__":
    app.run()
