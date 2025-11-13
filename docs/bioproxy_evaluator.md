# BioproxyEvaluator Guide

This guide mirrors the `EmbeddingManager` document and focuses on the `BioproxyEvaluator` utility (`src/phenoseeker/bioproxy_evaluator.py`). The class helps you benchmark Cell Painting embeddings against orthogonal bioactivity readouts (e.g., Lit-PCBA, proprietary HTS panels) by computing ligand-to-ligand distances, hit rankings, and enrichment factors.

## 1. When to use it

- **Cross-modal validation** – quantify how well a given embedding space prioritizes known bioactive compounds for each screening panel.
- **Screen-aware subsetting** – keep screen metadata, embeddings, and bioactivity roles aligned in a single container so you can swap embeddings without rebuilding datasets.
- **Standardized metrics** – report enrichment factor (EF), normalized EF, and hit-rate deltas at user-defined thresholds (top N% or distance cut-offs) with a single call.

## 2. Required inputs

1. **Compound metadata + embeddings**  
   - `compounds_metadata`: a `DataFrame` or path containing `Metadata_JCP2022`, `Metadata_InChI`, and any descriptors you want to carry around.  
   - `embeddings_path`: `.npy` or directory holding the compound-level embeddings referenced by `embeddings_name` (`Embeddings` by default).  
   - `embeddings_entity`: entity handled by the stored embeddings (`"compound"` in most cases).

2. **Screen folders**  
   - `screens_folders`: `dict[str, Path]` mapping a source name (e.g., `"lit_pcba"`, `"public_hts"`) to a directory containing one CSV per screen/assay.  
   - Each CSV is expected to include `Metadata_JCP2022` plus a column describing the hit label. If the column is named `role_val` it is automatically renamed to `Metadata_Bioactivity`. Duplicate JCP IDs are collapsed so every compound appears once per screen.

## 3. Quick start

```python
from pathlib import Path
import pandas as pd
from phenoseeker import BioproxyEvaluator

metadata = pd.read_parquet("compounds.parquet")
screens = {"lit_pcba": Path("data/lit_pcba_rankings")}

bio = BioproxyEvaluator(
    compounds_metadata=metadata,
    embeddings_path=Path("embeddings/compounds.npy"),
    screens_folders=screens,
    embeddings_name="Embeddings_rZ_ZCA_Int",
)

# Compute EF at multiple top-percentage cutoffs for all Lit-PCBA screens
thresholds = [0.5, 1, 5, 10]  # percentage of the ranked list
ef_summary = bio.compute_enrichment_factors(
    source="lit_pcba",
    embeddings_name="Embeddings_rZ_ZCA_Int",
    thresholds=thresholds,
    mode="percentage",
)
```

## 4. Screen ingestion details

- Every CSV inside a `screens_folders[source]` directory is treated as a single screen; the filename (without extension) becomes the screen identifier.
- Columns named `Unnamed: 0` are dropped, and `role_val` is renamed to `Metadata_Bioactivity`. Use either `"hit"`/`"non-hit"` or `"hit"`/`"inactive"` labels.
- The evaluator filters the global `EmbeddingManager` to the subset of `Metadata_JCP2022` IDs present in the screen and merges the screen-specific columns into the local metadata; you can add additional label columns (e.g., IC50) and they will be preserved.

## 5. Ranking API

`compute_ranking(source, screen, embeddings_name, JCP2022_id, distance="cosine")`

- Builds (or reuses) the requested distance matrix via the underlying `EmbeddingManager`.
- Returns the sorted list of compounds (excluding the query) along with their distances and bioactivity labels.
- Supported `distance` arguments match the EmbeddingManager distance helpers (`"cosine"`, `"euclidean"`, `"cityblock"`, etc.). If you need fingerprint distances you can precompute them and inject them as a custom matrix before calling the method.

## 6. Enrichment metrics

- `compute_enrichment_factor_for_screen(...)` loops over every hit inside a specific screen and aggregates EF metrics at each threshold you pass.
- `compute_enrichment_factors(...)` iterates over all screens of a given source and returns a tidy `DataFrame` with per-threshold aggregates (mean/median/max EF, normalized EF, hit rates, counts).  
- Threshold interpretation:
  - `mode="percentage"` (default) → treat each integer/float as `% of ranked list`. A value of `1` means “top 1% of candidates”.
  - `mode="distance"` → treat thresholds as absolute distance cut-offs.
- `Normalized_EF` rescales EF so that 100 represents the theoretical optimum given the number of hits available at that cutoff.

Example:

```python
screen_stats = bio.compute_enrichment_factor_for_screen(
    source="lit_pcba",
    screen="ACHE",
    embeddings_name="Embeddings_rZ_ZCA_Int",
    thresholds=[0.5, 1, 2],
)
```

## 7. Updating embeddings on the fly

Use `BioproxyEvaluator.load()` to add a brand-new embedding tensor (or reload an updated one) into the global manager and propagate it to every screen without rebuilding the evaluator:

```python
bio.load(
    embedding_name="Embeddings_new",
    embeddings_file=Path("embeddings/compounds_new.npy"),
    metadata_file=None,  # optional override if metadata rows differ
    dtype=np.float32,
)
```

This is particularly handy when sweeping normalization pipelines: instantiate once, then call `load()` for each embedding you want to benchmark.

## 8. Visual QC

`plot_assays_distribution(source)` plots the hit count and total molecule count per screen for a given source. Use it to spot imbalanced assays before you interpret EF values.

```python
bio.plot_assays_distribution("lit_pcba")
```

## 9. Tips & troubleshooting

- Ensure every screen CSV only contains compounds that exist in the global metadata. The evaluator raises an error if a `Metadata_JCP2022` identifier is missing from the loaded embeddings.
- Distance matrices are cached inside each screen-level `EmbeddingManager` (`manager.distance_matrices`). Delete an entry or restart the session if you need to recompute them with new embeddings.
- For very large screens, pre-filter to a subset of plates or families before computing distance matrices to limit memory usage, or rely on `EmbeddingManager.cleanup_large_pipelines()` after each experiment to reclaim RAM.

