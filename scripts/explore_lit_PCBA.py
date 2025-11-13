"""Annotate Lit-PCBA compounds with the closest JUMP reference compound."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the closest Cell Painting compound for each PCBA ligand using "
            "Tanimoto similarity on Morgan fingerprints."
        )
    )
    parser.add_argument(
        "--metadata-parquet",
        type=Path,
        required=True,
        help="Parquet file containing JUMP metadata with Metadata_JCP2022/InChI columns.",
    )
    parser.add_argument(
        "--pcba-root",
        type=Path,
        required=True,
        help="Directory containing subfolders with literature PCBA actives/inactives .smi files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where annotated parquet files will be written.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Radius parameter for Morgan fingerprints.",
    )
    parser.add_argument(
        "--fp-size",
        type=int,
        default=2048,
        help="Bit size for Morgan fingerprints.",
    )
    return parser.parse_args()


def make_generator(radius: int, fp_size: int):
    return AllChem.GetMorganGenerator(
        radius=radius, fpSize=fp_size, includeChirality=False
    )


def inchi_to_fp(inchi: str, generator) -> Any:
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    return generator.GetFingerprint(mol)


def smiles_to_fp(smiles: str, generator) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return generator.GetFingerprint(mol)


def load_jump_metadata(metadata_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(metadata_parquet, engine="pyarrow")
    df = df[["Metadata_JCP2022", "Metadata_InChI"]]
    return df.drop_duplicates().reset_index(drop=True)


def add_jump_fingerprints(df_jump: pd.DataFrame, generator) -> pd.DataFrame:
    fps = []
    keep_rows = []
    for idx, inchi in enumerate(
        tqdm(df_jump["Metadata_InChI"], desc="JUMP fingerprints")
    ):
        fp = inchi_to_fp(inchi, generator)
        if fp is None:
            continue
        fps.append(fp)
        keep_rows.append(idx)
    filtered = df_jump.iloc[keep_rows].reset_index(drop=True)
    filtered["fingerprint"] = fps
    return filtered


def load_pcba_runs(pcba_root: Path) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for folder in sorted(pcba_root.iterdir()):
        if not folder.is_dir():
            continue
        actives_file = folder / "actives.smi"
        inactives_file = folder / "inactives.smi"
        frames = []
        if actives_file.exists():
            df_actives = pd.read_csv(
                actives_file, sep=" ", names=["smiles", "id_lit_pcba"], engine="c"
            )
            df_actives["Active"] = True
            frames.append(df_actives)
        if inactives_file.exists():
            df_inactives = pd.read_csv(
                inactives_file, sep=" ", names=["smiles", "id_lit_pcba"], engine="c"
            )
            df_inactives["Active"] = False
            frames.append(df_inactives)
        if frames:
            datasets[folder.name] = pd.concat(frames, ignore_index=True)
    return datasets


def collect_unique_smiles(datasets: Dict[str, pd.DataFrame]) -> set[str]:
    unique_smiles: set[str] = set()
    for df in datasets.values():
        unique_smiles.update(df["smiles"].dropna().unique())
    return unique_smiles


def map_smiles_to_jump(
    unique_smiles: set[str], generator, jump_fps: list, jump_ids: list
) -> Dict[str, tuple[Any, float]]:
    mapping: Dict[str, tuple[Any, float]] = {}
    for query_smiles in tqdm(
        unique_smiles, desc="Scoring literature molecules", leave=False
    ):
        fp = smiles_to_fp(query_smiles, generator)
        if fp is None:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fp, jump_fps)
        similarities = np.asarray(similarities)
        best_index = int(np.argmax(similarities))
        mapping[query_smiles] = (
            jump_ids[best_index],
            float(similarities[best_index]),
        )
    return mapping


def annotate_and_save(
    datasets: Dict[str, pd.DataFrame],
    mapping: Dict[str, tuple[Any, float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tqdm(datasets.items(), desc="Writing outputs"):
        df = df.copy()
        df["closest_jcp"] = df["smiles"].map(
            lambda smi: mapping.get(smi, (None, np.nan))[0]
        )
        df["tanimoto_similarity"] = df["smiles"].map(
            lambda smi: mapping.get(smi, (None, np.nan))[1]
        )
        df.to_parquet(output_dir / f"jump_{name}.parquet", index=False)


def main() -> None:
    args = parse_args()
    generator = make_generator(args.radius, args.fp_size)
    df_jump = load_jump_metadata(args.metadata_parquet)
    df_jump = add_jump_fingerprints(df_jump, generator)
    if df_jump.empty:
        raise ValueError("No valid JUMP entries with InChI were found.")
    datasets = load_pcba_runs(args.pcba_root)
    if not datasets:
        raise ValueError("No PCBA folders with actives/inactives were found.")
    unique_smiles = collect_unique_smiles(datasets)
    if not unique_smiles:
        raise ValueError("No SMILES strings were loaded from the PCBA folders.")
    mapping = map_smiles_to_jump(
        unique_smiles,
        generator,
        df_jump["fingerprint"].tolist(),
        df_jump["Metadata_JCP2022"].tolist(),
    )
    if not mapping:
        raise ValueError("Fingerprint generation failed for all literature SMILES.")
    annotate_and_save(datasets, mapping, args.output_dir)


if __name__ == "__main__":
    main()
