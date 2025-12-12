<div align="center">

# PhenoSeeker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8475.svg)](https://doi.org/10.5281/zenodo.8475)
[![python](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.9-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![preprint](https://img.shields.io/badge/preprint-bioRxiv-red)](https://www.biorxiv.org/content/10.1101/2025.05.16.654292v1.full.pdf)
[![license](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
</div>

> Non-commercial use only. This repository is licensed under CC BY-NC 4.0.

## What is PhenoSeeker?

PhenoSeeker is a Python toolkit for phenotype-based molecule discovery using Cell Painting data. It includes utilities to load, normalize, aggregate, and evaluate embeddings; scripts to extract image features and create profiles; and helper analyses used in the accompanying publication.

## Publication & Citation

- Preprint: ["Large Scale Compound Selection Guided by Cell Painting Reveals Activity Cliffs and Functional Relationships"](https://www.biorxiv.org/content/10.1101/2025.05.16.654292v1.full.pdf).
- Citation: Sanchez, M., Bourriez, N., Bendidi, I., Cohen, E., Svatko, I., Del Nery, E., Tajmouati, H., Bollot, G., Calzone, L., & Genovesio, A. (2025). *Large Scale Compound Selection Guided by Cell Painting Reveals Activity Cliffs and Functional Relationships*. bioRxiv. https://doi.org/10.1101/2025.05.16.654292

## Repository Contents

- `src/phenoseeker/` - Core library (EmbeddingManager for aggregation/normalization/visualization, BioproxyEvaluator, transformation utilities).
- `scripts/` - Task-oriented entrypoints: feature extraction, profile creation, normalization sweeps, ChEMBL label extraction, Lit-PCBA mapping, pathway analysis, and more.
- `configs/` - YAML templates for each script (e.g., `config_extraction.yaml`, `create_profiles.yaml`, `config_test_all_norms.yaml`, `config_pathways_analysis.yaml`).
- `docs/` - How-to guides (`embedding_manager.md`, `bioproxy_evaluator.md`) with practical examples.
- `notebooks/` - Reproducible analysis notebooks (e.g., `notebooks/fig_1_umap.ipynb`).
- `data/` - Cytoscape session files and supporting tables used in the manuscript figures.

## Installation (Python 3.11+)

```bash
git clone https://github.com/mxfly14/2025_sanchez_phenoseeker.git
cd 2025_sanchez_phenoseeker
poetry env use 3.11
poetry install
poetry shell
```

## Quick Workflows

### 1) Extract image features

Edit `configs/config_extraction.yaml` with your data locations and run:

```bash
python scripts/extract_features.py
```

### 2) Create well and compound profiles (recommended pipeline)

`configs/create_profiles.yaml` defines inputs/outputs and the default normalization recipe (image -> well mean -> sphering on DMSO controls -> inverse normal transform):

```bash
python scripts/create_profiles.py -c configs/create_profiles.yaml
```

Outputs:
- Aligned well-level embeddings in `well_output_dir`
- Aggregated compound-level embeddings in `compound_output_dir`

### 3) Evaluate normalization pipelines

Grid-search normalization sequences and log mAP scores by configuring `configs/config_test_all_norms.yaml`, then run:

```bash
python scripts/test_normalisations.py
```

### 4) Add ChEMBL activity labels

After downloading the ChEMBL SQLite DB, set the paths in `scripts/get_chembl_activities.py` (metadata parquet, chembl db, output folder) and run:

```bash
python scripts/get_chembl_activities.py
```

### 5) Map Lit-PCBA ligands to JUMP compounds

Download per-target `actives.smi` / `inactives.smi` files into target-specific folders, then:

```bash
python scripts/explore_lit_PCBA.py \
  --metadata-parquet path/to/metadata_openphenom.parquet \
  --pcba-root path/to/lit_pcba \
  --output-dir path/to/output
```

Follow with `scripts/prepare_csv_lit_pcba.py` (update `SRC_DIR`/`DST_DIR` inside the script) to create per-target CSVs.

### 6) Pathway-level phenotypic similarity

Place `BindingDB_All_202412_tsv.zip` in `./data`, adjust `configs/config_pathways_analysis.yaml` (base path, output folder, metadata parquet with embeddings), then run:

```bash
python scripts/pathways_max.py
```

### 7) Bioproxy evaluations and plots

See `docs/bioproxy_evaluator.md` for wiring screens, computing enrichment factors, and generating QC plots with `BioproxyEvaluator`.

## Documentation

- `docs/embedding_manager.md` - Loading, filtering, normalization recipes, aggregation across entity levels, and QC/metrics examples.
- `docs/bioproxy_evaluator.md` - End-to-end bioproxy evaluation workflows.

## License (Non-commercial)

This project is distributed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Commercial use is not permitted. See `pyproject.toml` and [the license text](https://creativecommons.org/licenses/by-nc/4.0/) for details.

## Acknowledgments

If you use PhenoSeeker in academic work, please cite the preprint above and consider linking back to this repository. Contributions are welcome.
