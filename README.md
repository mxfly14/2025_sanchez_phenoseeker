<div align="center">

# PhenoSeeker

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

## Description
PhenoSeeker - A Python toolkit for phenotype-based molecule discovery using Cell Painting data.
You can test the method directly -> https://www.phenoseeker.bio.ens.psl.eu/ 

## Publication
Read the bioRxiv preprint ["Large Scale Cell Painting Guided Compound Selection Reveals Activity Cliffs and Functional Relationships"](https://www.biorxiv.org/content/10.1101/2025.05.16.654292v1.full.pdf).

## Citation
Sanchez, M., Bourriez, N., Bendidi, I., Cohen, E., Svatko, I., Del Nery, E., Tajmouati, H., Bollot, G., Calzone, L., & Genovesio, A. (2025). *Large Scale Cell Painting Guided Compound Selection Reveals Activity Cliffs and Functional Relationships*. bioRxiv. https://doi.org/10.1101/2025.05.16.654292

## Data artifacts
All network visualizations referenced in the manuscript are bundled under `/data` as Cytoscape-compatible sessions so you can reproduce every network related figure directly in Cytoscape.

## Code Availability

The PhenoSeeker codebase will be publicly released soon!  




## Installation

1. **Clone the Repository**  
   Clone the PhenoSeeker repository to your local machine:
   ```bash
   git clone https://github.com/mxfly14/2025_sanchez_phenoseeker.git
   cd phenoseeker

# TODO link to install poetry 


2. **Set Up a Virtual Environment**  
   Create and activate a Python 3.10 virtual environment:
   ```bash
   poetry env use 3.10

3. **Install Dependencies**  
   Install all required dependencies using Poetry:
   ```bash
   poetry install

4. **Activate Poetry Shell**  
   Run poetry shell to enter the virtual environment:
   ```bash
   poetry shell

## Extracting Image Features

To extract image features using PhenoSeeker, follow these steps:

1. **Prepare the Configuration File**  
   Update the `configs/config_extraction.yaml` file with the appropriate paths and parameters for your dataset and feature extraction settings.

2. **Run the Extraction Script**  
   Execute the `extract_features.py` script located in the `scripts` directory:

   ```bash
   python scripts/extract_features.py

## Downloading ChEMBL Activity Labels

After downloading the official ChEMBL SQLite database, use `scripts/get_chembl_activities.py` to build the activity label matrix that matches your Cell Painting metadata. Update the constants at the bottom of the script with:

- `METADATA_PATH`: path to the full JUMP metadata CSV (e.g., `complete_metadata.csv`).
- `CHEMBL_DB_PATH`: path to the downloaded `chembl_<version>.db` file.
- `base_path`: folder where you want the extracted tables to be saved.

Then run:

```bash
python scripts/get_chembl_activities.py
```

This writes `chembl_activity_data.csv` plus helper tables inside `base_path`.

## Testing Normalisation Pipelines

The repository ships with `scripts/test_normalisations.py`, which enumerates transformation pipelines and logs their MAP scores. Configure your paths, plate selection, and transformation search space inside `configs/config_test_all_norms.yaml`, then launch the evaluation:

```bash
python scripts/test_normalisations.py
```

Results (per-label MAPs and logs) are stored under the experiment folder defined in the config.

## Creating compounds phenotypic profiles

Generate well-level and compound-level profiles with:

```bash
python scripts/create_profiles.py -c configs/create_profiles.yaml
```

The YAML file defines the embedding inputs, output folders, aggregation strategy, and normalisation parameters so you can tailor the profiling pipeline to a new dataset.

## Literature PCBA matching

Use `scripts/explore_lit_PCBA.py` to map Lit-PCBA ligands (download the per-target `actives.smi` / `inactives.smi` files from https://drugdesign.unistra.fr/LIT-PCBA/ and keep them in one folder per assay) to their closest JUMP compounds using Morgan fingerprint similarity. Supply the JUMP metadata parquet, the root directory containing the SMILES folders, and an output directory:

```bash
python scripts/explore_lit_PCBA.py \
  --metadata-parquet path/to/metadata_openphenom.parquet \
  --pcba-root path/to/lit_pcba \
  --output-dir path/to/output
```

Optional flags allow tuning the fingerprint radius and size. Each subfolder produces a `jump_<folder>.parquet` file with the assigned `closest_jcp` identifier and the associated Tanimoto score.

## Documentation

- `docs/embedding_manager.md` — deep dive into loading, filtering, normalising, and aggregating embeddings with practical recipes and naming conventions.
- `docs/bioproxy_evaluator.md` — walk-through of the bioproxy workflow, from wiring screens to computing enrichment factors and visual QC plots.

## Evaluating phenotypic profiles for molecule selection
