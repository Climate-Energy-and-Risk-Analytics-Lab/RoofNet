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

Batch run over the entire holdout split (617 images). Each image processed through: RemoteCLIP prediction → Transformer Explainability attribution → GroundingDINO proposals → SAM refinement → overlap metrics.

### Quick table

| Metric | Median | Mean | Std | Interpretation |
|---|---:|---:|---:|---|
| `attribution_mass_inside` | 0.933 | 0.828 | 0.219 | 93% of attribution mass falls inside segmented roof/building regions on a typical image |
| `attribution_mass_outside` | 0.067 | 0.172 | 0.219 | Only 7% leaks outside segmented regions on median |
| `combined_iou` | 0.200 | 0.195 | 0.038 | Moderate spatial overlap between binarized attribution and combined SAM mask |
| `inside_ratio` | 0.930 | 0.846 | 0.198 | Attribution strongly concentrated inside mask |
| `coverage_ratio` | 0.200 | 0.212 | 0.072 | SAM mask is ~5× larger than attribution region — attribution is selective |

### Consistency rates

| Threshold | Rate |
|---|---:|
| Mass inside ≥ 50% | 90.3% |
| Mass inside ≥ 70% | 77.1% |
| Combined IoU ≥ 10% | 99.0% |
| Combined IoU ≥ 20% | 52.0% |
| Combined IoU ≥ 30% | 1.6% |
| High mass (≥50%) AND IoU (≥20%) | 45.7% |

### Health stats

| Stat | Value |
|---|---|
| Images processed | 617 / 617 |
| Failures | 0 |
| Success rate | 100% |
| GroundingDINO fallback box rate | 1.1% |
| Mean GroundingDINO boxes / image | 3.7 |
| Mean SAM masks / image | 3.7 |

### Detailed interpretation

#### Attribution mass

- `median_attribution_mass_inside: 0.933` — on a typical image, 93.3% of attribution mass falls inside the combined SAM mask. Model attention is overwhelmingly directed at the segmented roof/building region.
- `iqr_attribution_mass_inside: 0.717–0.999` — lower quartile still at 71.7%, meaning even the bottom 25% of images have most attribution inside the mask.
- `median_attribution_mass_outside: 0.067` — only 6.7% of attribution leaks outside segmented regions on median. Minimal background distraction.

#### Spatial overlap (IoU)

- `median_combined_iou: 0.200` — IoU is moderate because attribution is selective (concentrated on a subset of the roof) while SAM masks cover the full building footprint. Low IoU does not mean poor alignment — it means attribution is more focused than the mask.
- `iqr_combined_iou: 0.186–0.200` — tight distribution; overlap behavior is consistent across holdout.

#### Inside-ratio vs. coverage-ratio

- `median_inside_ratio: 0.930` — 93% of binarized attribution pixels fall inside the mask. Attribution rarely spills outside the building region.
- `median_coverage_ratio: 0.200` — attribution covers only 20% of the mask area. Model focuses on a subset of the roof rather than the entire building footprint. This is consistent with the spatial-concentration findings from § 1.

#### Consistency

- 90.3% of images have ≥50% attribution mass inside the mask — strong, near-universal alignment between model attention and building location.
- 77.1% have ≥70% mass inside — for most images, the vast majority of attribution is building-localized.
- 99.0% have ≥10% IoU — almost every image shows nonzero spatial overlap.
- Only 52.0% have ≥20% IoU — consistent with attribution being more focused than the mask (low coverage ratio drives IoU down).
- 45.7% meet both high-mass and high-IoU criteria — images where attribution is both well-localized and spatially overlapping.

#### Health

- `success_rate: 1.0` — every holdout image completed without failure.
- `fallback_box_rate: 0.011` — GroundingDINO returned zero proposals on only 1.1% of images (7/617), triggering full-image fallback box. Near-perfect detection.
- `mean_gdino_boxes: 3.7` — GroundingDINO typically proposes ~4 building bounding boxes per image, consistent with satellite crops often containing multiple structures.

#### Main interpretation

The segmentation-overlap analysis confirms the hypothesis that model attribution aligns with actual building/roof regions:

- **Mass inside is high** (median 0.933) — model is not distracted by background context; attention is localized to the building subject.
- **IoU is moderate** (median 0.200) because attribution is selective, not because it misaligns. The model focuses on discriminative roof features rather than the full building footprint.
- **Coverage ratio is low** (median 0.200) — consistent with the spatial-concentration findings from § 1: the model needs only a fraction of the image to make its prediction.
- **GroundingDINO detection is near-perfect** (1.1% fallback rate) — the segmentation pipeline is reliable and the results are not inflated by fallback full-image boxes.

### Pipeline config

| Parameter | Value |
|---|---|
| Split | holdout |
| Attribution method | Transformer Explainability |
| GroundingDINO model | `IDEA-Research/grounding-dino-base` |
| SAM model | `facebook/sam-vit-huge` |
| Grounding text prompt | `building . rooftop . roof` |
| GDINO box threshold | 0.20 |
| GDINO text threshold | 0.15 |
| Attribution percentile | 80 |

### Outputs

Per-image CSV, summary JSON, and plots are committed under `xAI_outputs/segmentation/`:

```text
xAI_outputs/segmentation/
├── tables/segmentation_overlap_results.csv
├── summary/segmentation_overlap_summary.json
└── plots/
    ├── attribution_mass_inside_hist.png
    ├── attribution_mass_outside_hist.png
    ├── combined_iou_hist.png
    ├── mass_inside_vs_iou_scatter.png
    ├── metric_boxplots.png
    ├── consistency_threshold_bars.png
    └── ecdf_mass_inside_and_iou.png
```

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
