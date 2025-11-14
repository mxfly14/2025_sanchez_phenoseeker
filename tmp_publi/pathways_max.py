import numpy as np
import pandas as pd
import os

# import random
import logging
import yaml

# Core libraries
from pathlib import Path
from tqdm import tqdm

# Machine learning & statistics
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

import matplotlib.pyplot as plt

# from joblib import Parallel, delayed
import glob

# Configure logging
logging.basicConfig(
    filename="process_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logging.info(f"Loaded configuration from {config_path}.")
    return config


def compute_phenotypic_similarity(df):
    embeddings = np.stack(df["Embeddings_mean"])
    similarity_matrix = cosine_similarity(embeddings)
    logging.info("Computed phenotypic similarity matrix.")
    return similarity_matrix


def compute_average_precision_for_hit(hit_embedding, all_embeddings, all_labels):
    similarities = cosine_similarity(
        hit_embedding.reshape(1, -1), all_embeddings
    ).flatten()
    ranked_indices = np.argsort(-similarities)
    ranked_labels = all_labels[ranked_indices]
    precisions = []
    num_hits = 0
    for rank, rel_label in enumerate(ranked_labels, start=1):
        if rel_label == 1:
            num_hits += 1
            precisions.append(num_hits / rank)
    logging.debug("Computed average precision for a hit.")
    return np.mean(precisions) if precisions else 0


def compute_enrichment_factor_at_n(
    hit_embedding, all_embeddings, all_labels, n_percent
):
    similarities = cosine_similarity(
        hit_embedding.reshape(1, -1), all_embeddings
    ).flatten()
    ranked_indices = np.argsort(-similarities)
    ranked_labels = all_labels[ranked_indices]
    ranked_indices = ranked_indices[1:]
    ranked_labels = ranked_labels[1:]
    n_top = max(1, int(len(ranked_labels) * (n_percent / 100)))
    hits_in_top_n = np.sum(ranked_labels[:n_top])
    total_hits = np.sum(all_labels)
    logging.debug("Computed enrichment factor at n%.")
    return (
        (hits_in_top_n / n_top) / (total_hits / len(all_labels))
        if total_hits > 0
        else 0.0
    )


def pairwise_similarities(group, metric="cosine", n_jobs=-1):
    """
    Calcul optimisé des distances intra-groupe.
    """
    group_array = np.array(group, dtype=np.float32)  # Convertir en float32
    similarity_matrix = cosine_similarity(group_array, group_array, dense_output=True)
    return similarity_matrix[np.triu_indices(len(group_array), k=1)]


def sample_df(df):
    df_hits = df[df["Metadata_Bioactivity"] == "hit"]
    df_non_hits = df[df["Metadata_Bioactivity"] == "nan"].sample(n=5000)
    test_df = pd.concat([df_hits, df_non_hits], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return test_df


def save_plots(distances_a, distances_b, pathway_name, output_folder):

    data = [distances_a, distances_b]
    labels = [
        "Active Molecules (Intra)",
        "Inactive Molecules (Intra)",
    ]
    colors = ["#1f77b4", "#ff7f0e"]

    plt.figure(figsize=(25, 12))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)

    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("blue")
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")

    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.title(
        f"Distribution of Distances for pathway {pathway_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Cosine Distance", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    save_path_violin = output_folder / f"violin_plot_for_pathway_{pathway_name}.png"
    plt.savefig(save_path_violin)  # Sauvegarde
    plt.close()


def analyze_groups(group_a, group_b):
    group_a = np.array(group_a)
    group_b = np.array(group_b)

    # 1. Test de Levene (égalité des variances)
    _, levene_p = stats.levene(group_a, group_b)

    # 2. Test de Kurtosis (aplatissement des distributions)
    kurtosis_a = stats.kurtosis(group_a)
    kurtosis_b = stats.kurtosis(group_b)

    # 3. Test de Skewness (asymétrie des distributions)
    skewness_a = stats.skew(group_a)
    skewness_b = stats.skew(group_b)

    # 4. Test de Kolmogorov-Smirnov (différences entre distributions)
    _, ks_p = stats.ks_2samp(group_a, group_b)

    # 5. Analyse des valeurs négatives et positives avec seuil ±0.2
    left_threshold = -0.2
    right_threshold = 0.2

    left_prop_a = np.mean(group_a < left_threshold)
    left_prop_b = np.mean(group_b < left_threshold)
    right_prop_a = np.mean(group_a > right_threshold)
    right_prop_b = np.mean(group_b > right_threshold)

    _, left_p = stats.fisher_exact(
        [
            [left_prop_a * len(group_a), (1 - left_prop_a) * len(group_a)],
            [left_prop_b * len(group_b), (1 - left_prop_b) * len(group_b)],
        ]
    )
    _, right_p = stats.fisher_exact(
        [
            [right_prop_a * len(group_a), (1 - right_prop_a) * len(group_a)],
            [right_prop_b * len(group_b), (1 - right_prop_b) * len(group_b)],
        ]
    )

    # 6. Tests sur les distributions des valeurs positives et négatives
    negative_values_a = group_a[group_a < 0]
    negative_values_b = group_b[group_b < 0]
    positive_values_a = group_a[group_a > 0]
    positive_values_b = group_b[group_b > 0]

    _, ks_negative_p = stats.ks_2samp(negative_values_a, negative_values_b)
    _, ks_positive_p = stats.ks_2samp(positive_values_a, positive_values_b)

    # Résultats sous forme de dictionnaire
    results = {
        "variance_test": levene_p,
        "kurtosis": {"group_a": kurtosis_a, "group_b": kurtosis_b},
        "skewness": {"group_a": skewness_a, "group_b": skewness_b},
        "ks_test": ks_p,
        "ks_negative_test": ks_negative_p,
        "ks_positive_test": ks_positive_p,
        "extreme_values": {
            "left": {"group_a": left_prop_a, "group_b": left_prop_b, "p_value": left_p},
            "right": {
                "group_a": right_prop_a,
                "group_b": right_prop_b,
                "p_value": right_p,
            },
        },
    }

    return results


def process_json_file(json_path, df_phenom, new_df, n_percent, output_folder):
    logging.info(f"Processing JSON file: {json_path}")
    gsea_data = pd.read_json(json_path)
    genes_list = gsea_data.transpose()["geneSymbols"].iloc[0]

    filtered_df = new_df[
        new_df["gene_symbol"].isin(genes_list)
        & (
            new_df["Target Source Organism According to Curator or DataSource"]
            == "Homo sapiens"
        )
    ]

    df_final = filtered_df.drop_duplicates().merge(
        df_phenom, left_on="Ligand InChI", right_on="Metadata_InChI"
    )

    if df_final.empty:
        logging.warning(f"No data in df_final for pathway in {json_path}. Skipping.")
        return None

    inchi_list = df_final["Metadata_InChI"].to_list()
    df_phenom["Metadata_Bioactivity"] = df_phenom["Metadata_InChI"].apply(
        lambda x: "hit" if x in inchi_list else "nan"
    )
    df_phenom_test = sample_df(df_phenom)

    group_a = (
        df_phenom_test[df_phenom_test["Metadata_Bioactivity"] == "hit"][
            "Embeddings_mean"
        ]
        .apply(lambda x: np.array(x, dtype=np.float32))
        .tolist()
    )

    group_b = (
        df_phenom_test[df_phenom_test["Metadata_Bioactivity"] == "nan"][
            "Embeddings_mean"
        ]
        .apply(lambda x: np.array(x, dtype=np.float32))
        .tolist()
    )

    distances_a = pairwise_similarities(group_a, metric="cosine", n_jobs=-1)
    distances_b = pairwise_similarities(group_b, metric="cosine", n_jobs=-1)
    results = analyze_groups(distances_a, distances_b)

    logging.info(f"Completed processing for {json_path}.")

    pathway_name = os.path.basename(json_path).split(".")[0].replace("HALLMARK_", "")
    save_plots(distances_a, distances_b, pathway_name, output_folder)

    return {
        "Pathway": pathway_name,
        "Number of Genes": len(genes_list),
        "Number of Impacted Genes": len(set(df_final["gene_symbol"])),
        "Number of Molecules": len(inchi_list),
        "Mean Similarity Active": np.mean(distances_a),
        "Mean Similarity non-Active": np.mean(distances_b),
        "Mean Variance Active": np.var(distances_a),
        "Mean Variance non-Active": np.var(distances_b),
        "Kurtosis Active": results["kurtosis"]["group_a"],
        "Kurtosis Inactive": results["kurtosis"]["group_b"],
        "Skewness Active": results["skewness"]["group_a"],
        "Skewness Inactive": results["skewness"]["group_b"],
        "P-Value Levene Test": results["variance_test"],
        "P-Value Kolmogorov-Smirnov Test": results["ks_test"],
        "P-Value Kolmogorov-Smirnov Test Left": results["ks_negative_test"],
        "P-Value Kolmogorov-Smirnov Test Right": results["ks_positive_test"],
        "P-Value extreme values Test (Fisher exact) Left": results["extreme_values"][
            "left"
        ]["p_value"],
        "P-Value extreme values Test (Fisher exact) Right": results["extreme_values"][
            "right"
        ]["p_value"],
        "Proportion extreme values Left Active": results["extreme_values"]["left"][
            "group_a"
        ],
        "Proportion extreme values Left Inactive": results["extreme_values"]["left"][
            "group_b"
        ],
        "Proportion extreme values Right Active": results["extreme_values"]["right"][
            "group_a"
        ],
        "Proportion extreme values Right Inactive": results["extreme_values"]["right"][
            "group_b"
        ],
    }


def process_all_jsons(json_paths, df_phenom, new_df, n_percent, output_folder):
    logging.info("Starting to process all JSON files.")
    results = []
    for json_path in tqdm(json_paths, desc="Processing JSON files"):
        result = process_json_file(
            json_path, df_phenom, new_df, n_percent, output_folder
        )
        if result is not None:
            results.append(result)
    logging.info("Completed processing all JSON files.")
    return pd.DataFrame(results)


def save_results_to_csv(results_df, output_path):
    results_df.to_csv(output_path, index=False)
    logging.info(f"Saved results to CSV at {output_path}.")


def main():
    config = load_config("config.yaml")

    base_path = Path(config["base_path"])
    n_percent = config["enrichment_factor_percentage"]
    output_folder = Path(config["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    binding_db_path = base_path / Path("BindingDB_All_202412_tsv(1).zip")
    df_bd = pd.read_csv(
        binding_db_path, sep="\t", on_bad_lines="skip", low_memory=False
    )
    logging.info("Loaded BindingDB data.")

    paths_to_jsons = glob.glob(str(base_path / "gsea/*.json"))
    logging.info(f"Found {len(paths_to_jsons)} JSON files for processing.")

    df_phenom = pd.read_parquet(
        "/projects/synsight/data/openphenom/norm_2_compounds_embeddings.parquet"
    )
    logging.info("Loaded phenotypic embeddings.")

    columns_to_keep = [
        "Ligand InChI",
        "UniProt (SwissProt) Entry Name of Target Chain",
        "UniProt (SwissProt) Recommended Name of Target Chain",
        "Target Source Organism According to Curator or DataSource",
    ]

    new_df = df_bd[columns_to_keep]
    genes = []
    for i in new_df["UniProt (SwissProt) Entry Name of Target Chain"]:
        try:
            if isinstance(i, str):
                genes.append(i.split("_")[0])
            else:
                genes.append(None)
        except Exception as e:
            logging.error(f"Error processing gene: {e}")
            genes.append(None)
    new_df["gene_symbol"] = genes

    results_df = process_all_jsons(
        paths_to_jsons,
        df_phenom,
        new_df,
        n_percent,
        output_folder,
    )

    save_results_to_csv(results_df, output_folder / "results_summary.csv")

    logging.info("Main execution completed.")


if __name__ == "__main__":
    main()
