import argparse
from pathlib import Path
from typing import Any

import yaml

from phenoseeker import EmbeddingManager

DEFAULT_SPHERING_PARAMS = {
    "method": "ZCA",
    "norm_embeddings": True,
    "use_control": True,
    "n_jobs": -1,
}

DEFAULT_INT_PARAMS = {
    "n_jobs": -1,
}

DEFAULT_COMPOUND_COLS = ["Metadata_JCP2022", "Metadata_InChI"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create well-level and compound-level profiles from image embeddings."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a mapping at the root level.")
    return data


def _require_path(config: dict[str, Any], key: str, base_dir: Path) -> Path:
    if key not in config:
        raise ValueError(f"Missing required configuration key: '{key}'.")
    raw_value = config[key]
    if not isinstance(raw_value, str):
        raise ValueError(f"Configuration value for '{key}' must be a string path.")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _merge_params(defaults: dict[str, Any], overrides: Any) -> dict[str, Any]:
    params = defaults.copy()
    if isinstance(overrides, dict):
        params.update(overrides)
    return params


def main() -> None:
    args = _parse_args()
    config_path = args.config
    config_dir = config_path.parent

    config = _load_config(config_path)

    metadata_path = _require_path(config, "metadata_path", config_dir)
    embeddings_path = _require_path(config, "embeddings_path", config_dir)
    well_output_dir = _require_path(config, "well_output_dir", config_dir)
    compound_output_dir = _require_path(config, "compound_output_dir", config_dir)

    starting_embedding = config.get("starting_embedding_name", "Embeddings_Raw")
    sphering_params = _merge_params(
        DEFAULT_SPHERING_PARAMS, config.get("sphering", {})
    )
    sphered_name = sphering_params.pop("new_embeddings_name", "Embeddings__ZCA_C")

    int_params = _merge_params(
        DEFAULT_INT_PARAMS, config.get("inverse_normal_transform", {})
    )
    int_name = int_params.pop("new_embeddings_name", f"{sphered_name}__Int")

    well_aggregation = config.get("well_aggregation", "mean")
    compound_aggregation = config.get("compound_aggregation", "mean")
    well_group_n_jobs = config.get("well_group_n_jobs", -1)
    compound_group_n_jobs = config.get("compound_group_n_jobs", -1)
    compound_cols_to_keep = config.get("compound_cols_to_keep", DEFAULT_COMPOUND_COLS)

    print("Loading image-level embeddings...")
    image_em = EmbeddingManager(metadata_path, entity="image")
    image_em.load(starting_embedding, embeddings_path)

    print("Aggregating image embeddings to wells...")
    well_em = image_em.grouped_embeddings(
        group_by="well",
        embeddings_to_aggregate=[starting_embedding],
        aggregation=well_aggregation,
        n_jobs=well_group_n_jobs,
    )

    print("Applying sphering transform on well embeddings...")
    well_em.apply_spherizing_transform(
        embeddings_name=starting_embedding,
        new_embeddings_name=sphered_name,
        **sphering_params,
    )

    print("Applying inverse normal transform...")
    well_em.apply_inverse_normal_transform(
        embeddings_name=sphered_name,
        new_embeddings_name=int_name,
        **int_params,
    )
    final_well_embedding = int_name

    print(f"Saving well-level embeddings to {well_output_dir}...")
    well_em.save_to_folder(well_output_dir, embeddings_name=final_well_embedding)

    print("Aggregating wells to compounds...")
    compounds_em = well_em.grouped_embeddings(
        group_by="compound",
        embeddings_to_aggregate=[final_well_embedding],
        cols_to_keep=compound_cols_to_keep,
        aggregation=compound_aggregation,
        n_jobs=compound_group_n_jobs,
    )

    print(f"Saving compound-level embeddings to {compound_output_dir}...")
    compounds_em.save_to_folder(
        compound_output_dir, embeddings_name=final_well_embedding
    )


if __name__ == "__main__":
    main()
