# xAI Results

Experimental results from explainability runs on the fine-tuned RemoteCLIP roof-material classifier.

---

## 1. Aggregate Spatial Attribution Metrics (Transformer Explainability)

Batch run over the entire holdout split (617 images). Sample size is large enough that patterns here should be treated as stable signals, not small-sample noise.

### Quick table

| Section | Metric | Value | Interpretation |
|---|---|---:|---|
| Sample health | `num_images` | 617 | Entire holdout set processed; statistics now meaningfully stable. |
| Sample health | `zero_sum_images` | 0 | No failed or degenerate heatmaps. |
| Sample health | `mean_negative_mass_ratio` | 0.0 | Transformer Explainability stayed strictly non-negative across holdout. |
| Concentration | `median_mass_center_25_square` | 0.286 | Center quarter holds 28.6% of attribution; only weak center bias over 25% uniform baseline. |
| Concentration | `iqr_mass_center_25_square` | 0.129 | Attention concentration varies meaningfully image to image. |
| Concentration | `median_mass_center_50_square` | 0.524 | Center half holds 52.4% of attribution; model still reads much of image. |
| Concentration | `iqr_mass_center_50_square` | 0.145 | Similar spread as center-quarter metric. |
| Concentration | `fraction_center25_over_50pct` | 0.039 | Only ~3.9% of images put majority of attribution in center quarter. |
| Half-mass area | `median_radius_for_50_mass_square` | 0.693 | Large square crop needed to capture half the attribution. |
| Half-mass area | `iqr_radius_for_50_mass_square` | 0.108 | Pattern fairly consistent across holdout. |
| Half-mass area | `median_radius_for_50_mass_radial` | 0.550 | Radial estimate confirms attribution extends well beyond center. |
| Half-mass area | `iqr_radius_for_50_mass_radial` | 0.088 | Slightly tighter spread than square estimate. |
| Shape | `median_radius_50_gap` | 0.139 | Attribution often has directional lean rather than radial symmetry. |
| Location | `median_centroid_offset_norm` | 0.100 | Average attribution mass stays broadly near image center. |
| Location | `median_peak_offset_norm` | 0.588 | Strongest individual cue usually lies near image periphery. |

### Detailed interpretation

#### Sample health

- `num_images: 617` — entire holdout set processed.
- `zero_sum_images: 0` — every image produced a valid, non-degenerate attribution heatmap; no failed or blank outputs needed exclusion.
- `mean_negative_mass_ratio: 0.0` — Transformer Explainability stayed strictly non-negative across all images, so normalization was clean and consistent throughout.

#### How concentrated is model attention?

- `median_mass_center_25_square: 0.286` — on a typical image, the center quarter of pixels contains 28.6% of total attribution. Uniform-random baseline would be 25%, so center preference exists but is weak.
- `iqr_mass_center_25_square: 0.129` — meaningful image-to-image spread; some samples are more center-focused, many are not.
- `median_mass_center_50_square: 0.524` — the center half of the image captures 52.4% of attribution on median. Against a 50% uniform baseline, this again suggests only weak center bias.
- `iqr_mass_center_50_square: 0.145` — similar spread as center-25 metric; concentration is not uniform across dataset.
- `fraction_center25_over_50pct: 0.039` — only about 3.9% of images (~24/617) place more than half their attribution mass inside the center quarter. Most images show diffuse, spread-out attention.

#### How much area captures half the signal?

- `median_radius_for_50_mass_square: 0.693` — square center-crops must span 69.3% of image half-width to capture 50% of attribution on a typical image. Large crop, weak central concentration.
- `iqr_radius_for_50_mass_square: 0.108` — moderate spread; this pattern is fairly consistent across holdout.
- `median_radius_for_50_mass_radial: 0.550` — radially, 55.0% of max image radius is needed to capture half the attribution mass. Same story: attribution extends well beyond center.
- `iqr_radius_for_50_mass_radial: 0.088` — slightly tighter than square estimate, consistent with radial measure being less sensitive to directional asymmetry.
- `median_radius_50_gap: 0.139` — square and radial estimates differ by ~14 percentage points on median, suggesting attribution is not radially symmetric and often has directional lean.

#### Where is model actually looking?

- `median_centroid_offset_norm: 0.100` — attribution center of mass sits only ~10.0% of image half-width away from geometric center. In aggregate, attention is still broadly centered.
- `median_peak_offset_norm: 0.588` — the single strongest attribution pixel sits ~58.8% of image half-width away from center on median. Strongest individual cue is typically near image periphery, not roof center.

#### Main interpretation

Most important tension in these results:

- **Centroid story:** `0.100` suggests average attribution mass stays roughly centered.
- **Peak story:** `0.588` suggests highest-value individual evidence often lives near image edge.

Interpretation: model appears to use broadly distributed contextual evidence, while its strongest single cue often comes from peripheral content. That peripheral cue could be neighboring structures, shadows, vegetation, or road-edge context rather than rooftop material alone.

---

## 2. Segmentation-Attribution Overlap (Transformer Explainability + GroundingDINO + SAM)

To be populated after running the segmentation overlap batch pipeline:

```bash
.venv/bin/python xAI_notebooks/remoteclip_segmentation_overlap_batch.py
```

Key metrics to report:
- `attribution_mass_inside` / `attribution_mass_outside` — fraction of attribution falling inside vs. outside segmented roof/building regions
- `combined_iou` — spatial overlap between binarized attribution and combined SAM mask
- `inside_ratio` / `coverage_ratio` — attribution-in-mask and mask-covered-by-attribution
- Consistency rates at threshold bands (mass ≥ 0.50/0.70, IoU ≥ 0.10/0.20/0.30)

Results will include:
- Per-image CSV with all metrics
- Summary JSON with aggregate statistics and consistency rates
- Histograms, scatter plots, and threshold bar charts under `xAI_outputs/segmentation/plots/`

---

## Aggregate metrics reference

The batch runner processes each method family's heatmaps through `attribution_helpers/feature_attribution_aggregation.py`. The per-image CSV contains these columns:

### Raw heatmap properties

| Column | Meaning |
|---|---|
| `raw_sum` | Sum of all pixel values before normalization |
| `raw_abs_sum` | Sum of absolute pixel values before normalization |
| `raw_min` / `raw_max` | Min and max pixel value in raw heatmap |
| `negative_mass_ratio` | `\|negative\| / \|total\|` — fraction of absolute mass that is negative. Near 0 → model used mostly positive evidence; near 0.5 → equal positive/negative |
| `is_zero_sum` | True if heatmap is all zeros (attribution failed for that image) |

### Spatial concentration

| Column | Meaning |
|---|---|
| `mass_center_25_square` | Fraction of total attribution mass inside the central 25%-area square |
| `mass_center_50_square` | Fraction of total attribution mass inside the central 50%-area square |
| `radius_for_50_mass_square` | Side fraction (0–1) of the smallest centered square that captures 50% of mass. Smaller → more concentrated |
| `radius_for_50_mass_radial` | Normalized radius (0–1) of the smallest centered circle that captures 50% of mass. Smaller → more concentrated |
| `radius_50_gap` | Square minus radial radius. Positive → mass is more circular than square; negative → mass follows square/edge pattern |

### Centroid and peak location

| Column | Meaning |
|---|---|
| `centroid_x` / `centroid_y` | Attribution-weighted centroid in pixel coordinates |
| `centroid_offset_px` / `centroid_offset_norm` | Euclidean distance from image center to centroid, in pixels / normalized to [0, 1]. Small offset_norm → model focused near center of image |
| `peak_x` / `peak_y` | Coordinates of the single highest-attribution pixel |
| `peak_offset_px` / `peak_offset_norm` | Offset of the peak pixel from center. Compare with centroid offset to distinguish broad (centroid near center, peak off-center) vs. sharp focus |

### Radial profile

| Column | Meaning |
|---|---|
| `radial_profile_00_20` through `radial_profile_80_100` | Attribution mass fraction in each concentric ring (0–20%, 20–40%, …, 80–100% of max radius). Monotonically decreasing → center-focused; flat → diffuse |

### Cross-image summary (`{method_family}_spatial_summary.csv`)

| Metric | Meaning |
|---|---|
| `num_images` | Number of heatmaps processed |
| `zero_sum_images` | Count of all-zero heatmaps |
| `median_mass_center_25_square` / `iqr_*` | Typical fraction of mass in center 25% area, with IQR spread. High median + narrow IQR → consistent center focus |
| `median_mass_center_50_square` / `iqr_*` | Same for center 50% area |
| `median_radius_for_50_mass_square` / `iqr_*` | Typical square crop size to capture half the mass |
| `median_radius_for_50_mass_radial` / `iqr_*` | Typical radial radius to capture half the mass |
| `fraction_center25_over_50pct` | Proportion of images where >50% of mass falls in center 25% area. High → strong center bias |
| `median_centroid_offset_norm` | Typical centroid displacement. Near 0 → consistent center focus |
| `median_peak_offset_norm` | Typical peak displacement. Compare with centroid offset |
| `mean_negative_mass_ratio` | Average negative evidence across batch. If high, consider positive-only aggregation instead |
| `median_radius_50_gap` | Typical gap between square and radial 50% radii. Large positive → mass is more circular than square |

### Generated plots

| File | How to read it |
|---|---|
| `{method_family}_radial_profile.png` | Mean ± 1 std of attribution mass across radial rings. Steep drop → center-concentrated; flat → diffuse |
| `{method_family}_center25_hist.png` | Histogram of `mass_center_25_square`. Right-skewed → most images concentrate in the center |
| `{method_family}_centroid_offset_hist.png` | Histogram of `centroid_offset_norm`. Tight cluster near 0 → model looks at center consistently; spread out → variable focus |
