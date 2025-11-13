import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from phenoseeker import load_config, MultiChannelImageDataset, modify_first_layer


# ---------------------------
# Logging
# ---------------------------
log_file = Path("./tmp/log_extraction.txt")
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("extract")


# ---------------------------
# Validation helpers
# ---------------------------
def validate_config(config: dict, required_keys: list):
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    return config


def validate_columns(
    df: pd.DataFrame, metadata_cols: list[str], images_cols: list[str]
):
    missing_metadata_cols = [c for c in metadata_cols if c not in df.columns]
    missing_images_cols = [c for c in images_cols if c not in df.columns]
    if missing_metadata_cols:
        raise ValueError(f"Missing metadata columns: {missing_metadata_cols}")
    if missing_images_cols:
        raise ValueError(f"Missing image columns: {missing_images_cols}")


# ---------------------------
# Tensor-only transforms (for CHW tensors)
# ---------------------------
class ResizeTensor:
    """Resize a CHW float tensor to (H,W) using bilinear interpolation."""

    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ResizeTensor expects CHW, got {tuple(x.shape)}")
        x = x.unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return x.squeeze(0).contiguous()  # [C,H,W]


def make_tensor_transform(n_channels: int):
    return transforms.Compose(
        [
            ResizeTensor((224, 224)),
        ]
    )


# ---------------------------
# Model loader
# ---------------------------
def load_backbone_from_config(config: dict, in_channels: int, device: torch.device):
    """
    Load DINOv2 or DINOv3 backbone based on config.
    Config keys:
      - model_family: "dinov2" | "dinov3"
      - model_variant: e.g. "dinov2_vitg14" | "dinov3_vit7b16"
      - dinov3_ckpt_path: local .pth for DINOv3 (recommended)
    """
    family = "dinov2"
    LOGGER.info(f"Requested model_family={family}")

    name = "dinov2_vitg14"
    LOGGER.info(f"Loading DINOv2 hub model: {name}")
    model = torch.hub.load("facebookresearch/dinov2", name)  # downloads small hub code

    try:
        if in_channels != 3:
            modify_first_layer(model, in_channels=in_channels)
            LOGGER.info(
                f"Expanded DINOv2 first layer to {in_channels} channels via modify_first_layer"  # noqa
            )
    except Exception as e:
        LOGGER.warning(
            f"modify_first_layer failed on DINOv2; keeping 3 channels. Error: {e}"
        )

    return model.to(device).eval(), family


# ---------------------------
# Main worker
# ---------------------------
def main_worker(model, model_family: str, df: pd.DataFrame, config: dict):
    gpu_id = int(config.get("gpu_id", 0))
    has_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_id}" if has_cuda else "cpu")
    if has_cuda:
        torch.cuda.set_device(gpu_id)
        torch.backends.cudnn.benchmark = True

    num_workers = int(config.get("num_workers", max(1, (os.cpu_count() or 4) // 2)))
    batch_size = int(config.get("batch_size", 32))
    output_folder = Path(config.get("output_folder", "./tmp"))
    output_folder.mkdir(parents=True, exist_ok=True)

    metadata_cols = config["metadata_cols"]
    images_cols = config["images_cols"]

    validate_columns(df, metadata_cols, images_cols)

    in_channels = len(images_cols)
    transform = make_tensor_transform(in_channels)

    dataset = MultiChannelImageDataset(
        df, metadata_cols=metadata_cols, images_cols=images_cols, transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=has_cuda,
        prefetch_factor=8 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    LOGGER.info(f"Using device: {device}")
    LOGGER.info(f"Dataset size: {len(dataset)} | Batches: {len(dataloader)}")

    features_list, metadata_list = [], []
    LOGGER.info("Starting feature extraction...")

    amp_enabled = has_cuda
    with torch.no_grad():
        for images, metadata in tqdm(dataloader, desc="Feature Extraction"):
            images = images.to(device, non_blocking=True).float()
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(images)

                feats = (
                    out
                    if isinstance(out, torch.Tensor)
                    else next(v for v in out.values() if isinstance(v, torch.Tensor))
                )
            features_list.append(feats.detach().cpu().numpy())
            metadata_list.append(metadata)

    features = np.concatenate(features_list, axis=0)
    metadata_df = pd.concat([pd.DataFrame(m) for m in metadata_list], ignore_index=True)

    np.save(output_folder / "extracted_features.npy", features)
    metadata_df.to_csv(output_folder / "metadata.csv", index=False)
    LOGGER.info(
        f"Feature extraction completed. Saved {features.shape} to {output_folder}"
    )


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    # Load and validate config
    config_path = Path("./configs/config_extraction.yaml")
    config = load_config(config_path)
    validate_config(
        config,
        required_keys=["parquet_path", "output_folder", "metadata_cols", "images_cols"],
    )

    parquet_path = config.get("parquet_path")
    if not Path(parquet_path).is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Device (we’ll pass it to loader)
    gpu_id = int(config.get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading dataset...")
    df = pd.read_parquet(parquet_path)

    LOGGER.info("Initializing backbone...")
    in_channels = len(config["images_cols"])
    model, family = load_backbone_from_config(
        config, in_channels=in_channels, device=device
    )

    main_worker(model, family, df, config)
