# EmbeddingManager Guide

This document focuses on the `EmbeddingManager` class (`src/phenoseeker/embedding_manager.py`) and explains how to use it to align, normalize, and analyze embedding spaces that come out of PhenoSeeker pipelines. The emphasis is on the different normalization strategies that are available so you can pick the right correction scheme for a given batch, plate, or use case.

## 1. Core Concepts

- **Metadata-aware container** - `EmbeddingManager` stores a metadata `DataFrame` plus one or more aligned embedding matrices in `self.embeddings`. The class keeps track of wells, plates, images, compounds, or other entities through the `entity` argument passed to the constructor.
- **Consistent indexing** - `load()` aligns `.npy` embedding tensors with metadata by creating an order key from the relevant columns (`Metadata_Well`, `Metadata_Plate`, etc.). After loading, every row in `self.df` matches a row vector in each stored embedding matrix.
- **Control-awareness** - When working at the well/image level the constructor adds `Metadata_Row_Number`/`Metadata_Col_Number` and attempts to infer DMSO controls (`Metadata_Is_dmso`). Many normalization routines rely on these flags.
- **Plate granularity** - Normalizations are almost always computed plate-by-plate to avoid leaking signal across experimental batches. The helper maps each plate to its row indices before applying transforms.
- **Entity levels** - Choose `entity="image"` when handling per-site (field of view) features, `entity="well"` for well aggregated profiles, and `entity="compound"` (after `grouped_embeddings`) when collapsing wells into structure-level signatures; each level preserves the metadata fields relevant to that resolution.

## 2. Quick Start

```python
from pathlib import Path
import pandas as pd
from phenoseeker.embedding_manager import EmbeddingManager

metadata = pd.read_parquet("metadata.parquet")
em = EmbeddingManager(metadata, entity="well")

em.load(
    embedding_name="Embeddings_Raw",
    embeddings_file=Path("embeddings.npy"),
    metadata_file=None,  # set to CSV/Parquet if different from metadata above
)

# Run normalization + QC
em.apply_robust_Z_score("Embeddings_Raw", new_embeddings_name="Embeddings_rZ")
em.compute_features_stats("Embeddings_rZ", plot=True)
em.plot_dimensionality_reduction("Embeddings_rZ", reduction_method="UMAP", color_by="Metadata_Plate")
```

## 3. Normalization & Correction Toolkit

All normalization helpers live inside the manager so they share the same metadata context and DMSO masks. Here is a condensed reference:

| Method | What it does | When to use | Key parameters |
| --- | --- | --- | --- |
| `apply_robust_Z_score` | Per-plate centering and scaling using robust estimators. | First-pass plate/batch correction; enforce comparable feature ranges. | `use_control`, `center_by` (`mean`/`median`), `reduce_by` (`std`/`iqrs`/`mad`). |
| `apply_inverse_normal_transform` | Applies a rank-based inverse normal transform (INT) feature-wise. | When marginal distributions are heavy-tailed and you want Gaussianized features before distance computations. | `indices` (subset of feature columns), `n_jobs`. |
| `apply_rescale` | Rescales each feature to `0-1` or `-1-1` range per plate. | Useful after Z-scoring when downstream models expect bounded inputs. | `scale` (`"0-1"` or `"-1-1"`). |
| `apply_spherizing_transform` | Learns a whitening transform (PCA/ZCA, optional correlation mode) on each plate (optionally DMSO-only) and reprojects samples; can renormalize to unit length. | When you want to decorrelate features, equalize variance, or prepare data for cosine similarity search. | `method` (`"PCA"`, `"ZCA"`, `"PCA-cor"`, `"ZCA-cor"`), `use_control`, `norm_embeddings`. |
| `apply_median_polish` | Treats each plate as a 2D grid (row/column) and removes spatial biases via Tukey's median polish per component. | Correcting row/column effects (edge wells) or illumination artifacts on high-content plates. | No extra params; relies on `Metadata_Row_Number`/`Metadata_Col_Number`. |

### 3.1 Robust Z-score (`apply_robust_Z_score`, lines 561-623)

- Computes centering (`mean` or `median`) and scaling (`std`, interquartile range, or median absolute deviation) per plate, optionally restricted to DMSO controls.
- The resulting embedding is stored under `new_embeddings_name` (`robust_Z_score` by default).
- Recommended as the first normalization layer before more advanced transforms. Combine with `use_control=True` to anchor plates on the same neutral baseline.

### 3.2 Inverse Normal Transform (`apply_inverse_normal_transform`, lines 1289-1349)

- Runs a rank-based INT for each selected feature column independently on every plate.
- Handles skewed or heavy-tailed distributions so PCA/UMAP, cosine similarity, and statistical tests behave closer to Gaussian assumptions.
- You can limit the transform to a subset of feature indices to save time or avoid discrete channels.

### 3.3 Feature Rescaling (`apply_rescale`, lines 1422-1472)

- Uses `norm_functions.rescale` to map features to `[0, 1]` or `[-1, 1]` intervals per plate.
- Works on top of Z-scored data when you need bounded features (e.g., when training models with activation constraints).
- Because scaling is plate-specific, relative differences inside each plate remain but global extrema are harmonized.

### 3.4 Sphering / Whitening (`apply_spherizing_transform`, lines 1474-1536)

- Fits the `Spherize` transformer (`ZCA` by default) using either all wells or DMSO-only wells on each plate, then applies it to the plate embeddings.
- Optional `norm_embeddings=True` enforces unit-norm rows, which is ideal when computing cosine distances (`pairwise_distances(..., metric="cosine")` later).
- Use `"PCA-cor"`/`"ZCA-cor"` to base the whitening on correlation matrices instead of covariance if features vary wildly in scale.

### 3.5 Median Polish (`apply_median_polish`, lines 1596-1671)

- Rebuilds each plate into a `(rows, cols, features)` tensor using `Metadata_Row_Number` and `Metadata_Col_Number`.
- Runs Tukey median polish per feature to extract additive row and column effects and subtracts them, reducing edge well and illumination bias.
- Writes the corrected embeddings back under `new_embeddings_name` (`Embeddings_MedianPolish` by default).

## 4. Suggested Normalization Recipes

Based on our internal benchmarking (see the `results/` reports for more details) we recommend the following default pipeline:

1. **Mean aggregation at the well level** using `grouped_embeddings(group_by="well", aggregation="mean")`. This dampens image-level noise while preserving per-plate structure.
2. **Sphering fitted on DMSO controls** via `apply_spherizing_transform(..., use_control=True, norm_embeddings=False)`. Training the whitening map on neutral wells stabilizes downstream cosine distances.
3. **Inverse normal transform** with `apply_inverse_normal_transform()` to Gaussianize marginal feature distributions.

Remember that each method writes a new entry into `self.embeddings`. You can branch pipelines by choosing different `new_embeddings_name` values and comparing downstream metrics (mAP, LISI, etc.).

## 5. Diagnostics & Quality Checks

- **Feature statistics** - `compute_features_stats()` aggregates per-plate and global statistics and can plot distribution overlays to spot drifts early.
- **Distribution probing** - `test_feature_distributions()` runs goodness-of-fit checks against a catalog of scipy distributions, which is particularly useful before INT or Z-score steps.
- **Visualization** - `plot_features_distributions()` and `plot_dimensionality_reduction()` (PCA, t-SNE, UMAP) help verify that treated plates mix as expected.
- **Neighborhood metrics** - `compute_lisi()` quantifies label mixing, while `compute_maps()` reports mean Average Precision for retrieval-style evaluations; both support plotting helpers (`plot_lisi_scores`, `plot_maps`).
- **Distance inspection** - `compute_distance_matrix()` plus `plot_distance_matrix()` or `hierarchical_clustering_and_visualization()` reveal duplicate plates, mislabeled wells, or overfitting.
- **Covariance monitoring** - `plot_covariance_and_correlation()` surfaces latent correlations either by sample or by feature to catch leakage after normalization.

## 6. Dataset Manipulation & Export

- **Filtering** - `filter_and_instantiate(**criteria)` returns a brand-new `EmbeddingManager` scoped to the filtered rows, carrying over only the matching slices of every embedding matrix.
- **Grouping** - `grouped_embeddings(group_by=..., aggregation="mean")` can aggregate wells into compounds, genes, or arbitrary custom groupings. Provide `embeddings_to_aggregate` to restrict which matrices are reduced.
- **Exporting** - `save_to_folder(Path("run_001"), embeddings_name="Embeddings_rZ")` writes the aligned metadata as Parquet plus each selected embedding as `.npy` for downstream work or sharing with teammates.

## 7. Automating Transformation Sequences

The helper `apply_transformations()` in `src/phenoseeker/all_norm_functions.py` lets you define ordered normalization recipes as simple dictionaries (or YAML) and execute them programmatically:

```python
from phenoseeker.all_norm_functions import apply_transformations

sequence = {
    "name": "INT_rZ_ZCA",
    "transformations": [
        {"method": "apply_inverse_normal_transform"},
        {
            "method": "apply_robust_Z_score",
            "params": {"center_by": "mean", "reduce_by": "std", "use_control": True},
        },
        {
            "method": "apply_spherizing_transform",
            "params": {"method": "ZCA", "norm_embeddings": True, "use_control": True},
        },
    ],
}

final_embedding = apply_transformations(
    well_em=em,
    sequence=sequence,
    starting_embedding_name="Embeddings_Raw",
)
```

Suffixes are derived automatically (see `get_suffix()` in the same module), so each intermediary embedding is saved with a descriptive name (`Embeddings_Raw__Int__rZMs_C__ZCA_N_C`, etc.).

## 8. Practical Examples

The snippets below showcase common patterns that combine the most useful `EmbeddingManager` APIs.

### 8.1 Loading and aligning embeddings

```python
from pathlib import Path
import pandas as pd
from phenoseeker import EmbeddingManager

metadata = pd.read_parquet("metadata.parquet")
image_em = EmbeddingManager(metadata, entity="image")
image_em.load(
    embedding_name="Embeddings_Raw",
    embeddings_file=Path("image_embeddings.npy"),
    metadata_file=None,  # optionally pass a CSV/Parquet with extra annotations or different number of samples
)
```

### 8.2 Moving across entity levels

```python
# Image -> well aggregation (mean)
well_em = image_em.grouped_embeddings(
    group_by="well",
    embeddings_to_aggregate=["Embeddings_Raw"],
    aggregation="mean",
    n_jobs=-1,
)

# Filter to a single plate and instantiate a new manager
plate_a = well_em.filter_and_instantiate(Metadata_Plate="PLATE_A_01")

# Well -> compound aggregation (median for robustness)
compound_em = well_em.grouped_embeddings(
    group_by="compound",
    embeddings_to_aggregate=["Embeddings_Raw"],
    aggregation="median",
    cols_to_keep=["Metadata_JCP2022", "Metadata_Is_dmso"],
)
```

### 8.3 Normalization passes

```python
# Robust Z-score using DMSO wells only
well_em.apply_robust_Z_score(
    embeddings_name="Embeddings_Raw",
    new_embeddings_name="Embeddings_rZ",
    use_control=True,
    center_by="median",
    reduce_by="mad",
)

# Plate-wise sphering (ZCA) followed by inverse-normal transform
well_em.apply_spherizing_transform(
    embeddings_name="Embeddings_rZ",
    new_embeddings_name="Embeddings_rZ_ZCA",
    method="ZCA",
    norm_embeddings=True,
    use_control=True,
)
well_em.apply_inverse_normal_transform(
    embeddings_name="Embeddings_rZ_ZCA",
    new_embeddings_name="Embeddings_rZ_ZCA_Int",
)

# Optional additional steps
well_em.apply_rescale(
    embeddings_name="Embeddings_rZ_ZCA_Int",
    new_embeddings_name="Embeddings_rZ_ZCA_Int_Res01",
    scale="0-1",
)
well_em.apply_median_polish(
    embeddings_name="Embeddings_Raw",
    new_embeddings_name="Embeddings_MedPol",
)
```

### 8.4 Plate diagnostics and visualization

```python
stats_df = well_em.compute_features_stats(
    embedding_name="Embeddings_rZ",
    plot=True,
    n_jobs=-1,
)

well_em.plot_features_distributions(
    embedding_name="Embeddings_rZ",
    feature_indices=[0, 1, 2],
    bins=30,
    hue_column="Metadata_Is_dmso",
)

well_em.plot_dimensionality_reduction(
    embedding_name="Embeddings_rZ_ZCA_Int",
    reduction_method="UMAP",
    color_by="Metadata_Plate",
    random_state=42,
)
```

### 8.5 Neighborhood quality metrics

```python
lisi_scores = well_em.compute_lisi(
    labels_column="Metadata_Plate",
    embeddings_names=["Embeddings_rZ", "Embeddings_rZ_ZCA_Int"],
    n_neighbors_list=[15, 30, 60],
    plot=True,
)

map_df = well_em.compute_maps(
    labels_column="Metadata_JCP2022",
    embeddings_names=["Embeddings_rZ_ZCA_Int"],
    distance="cosine",
    weighted=True,
    random_maps=True,
    plot=True,
)
```

### 8.6 Distance, clustering, and heatmaps

```python
well_em.compute_distance_matrix(
    embedding_name="Embeddings_rZ_ZCA_Int",
    distance="cosine",
    similarity=False,
)
well_em.plot_distance_matrix(
    embedding_name="Embeddings_rZ_ZCA_Int",
    distance="cosine",
    sort_by="Metadata_Plate",
    label_by="Metadata_Well",
)
well_em.hierarchical_clustering_and_visualization(
    embedding_name="Embeddings_rZ_ZCA_Int",
    distance="cosine",
    threshold=0.4,
)
```

### 8.7 Persistence and export

```python
filtered_em = well_em.filter_and_instantiate(Metadata_Is_dmso=True)
filtered_em.save_to_folder(
    folder_path=Path("outputs/dmso_only"),
    embeddings_name="Embeddings_rZ_ZCA_Int",
)
```

## 9. Tips & Troubleshooting

- **Verify controls** - `EmbeddingManager` auto-detects DMSO by `Metadata_InChI`; if your controls differ, populate `Metadata_Is_dmso` before instantiating or override `find_dmso_controls()`.
- **Track column names** - Methods such as `grouped_embeddings()` expect specific metadata columns; check `self.df.columns` or adjust upstream ETL scripts to keep plate/well identifiers.
- **Watch memory** - Each normalization typically creates a dense float matrix copy. Delete unused entries from `self.embeddings` (e.g., `del em.embeddings["Embeddings_Raw"]`) once you commit to a variant.
- **Parallelism** - Most heavy routines accept `n_jobs`. Use a sensible value for your hardware to avoid oversubscription; set to `1` when debugging to get clearer stack traces.
- **Reproducibility** - `plot_dimensionality_reduction()` exposes `random_state`, and mAP/LISI helpers rely on deterministic numpy operations, so pass fixed seeds when comparing runs.
