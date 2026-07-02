"""Quick verification for combine_data and CombinedLandCover loader."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.combined_landcover import CombinedLandCover
from utils import ext_transforms as et


def verify_files(data_root: Path):
    stats_path = data_root / "stats.json"
    train_txt = data_root / "train.txt"
    val_txt = data_root / "val.txt"

    assert stats_path.exists(), f"missing {stats_path}"
    assert train_txt.exists(), f"missing {train_txt}"
    assert val_txt.exists(), f"missing {val_txt}"

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    print(f"[OK] stats.json: train={stats['train_samples']}, val={stats['val_samples']}")
    print(f"     tile_size={stats['tile_size']}, stride={stats['stride']}")

    for split in ("train", "val"):
        lines = (data_root / f"{split}.txt").read_text(encoding="utf-8").strip().splitlines()
        for line in lines[:3]:
            img_rel, mask_rel = line.split()
            img_path = data_root / img_rel
            mask_path = data_root / mask_rel
            assert img_path.exists(), img_path
            assert mask_path.exists(), mask_path
            img = Image.open(img_path)
            mask = np.array(Image.open(mask_path))
            assert img.size == (1024, 1024), f"bad image size {img.size} for {img_path}"
            assert mask.shape == (1024, 1024), f"bad mask shape {mask.shape} for {mask_path}"
            uniq = set(np.unique(mask).tolist())
            assert uniq.issubset(set(range(8)) | {255}), f"unexpected labels {uniq} in {mask_path}"
        print(f"[OK] {split}: {len(lines)} samples, first tiles valid 1024x1024")


def verify_dataloader(data_root: Path):
    transform = et.ExtCompose([
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = CombinedLandCover(root=str(data_root), split="train", transform=transform)
    loader = data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    images, labels = next(iter(loader))
    assert images.shape[1:] == (3, 1024, 1024), images.shape
    assert labels.shape[1:] == (1024, 1024), labels.shape
    print(f"[OK] DataLoader batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")

    import network

    model = network.modeling.deeplabv3plus_mobilenet(num_classes=8, output_stride=16)
    model.eval()
    with torch.no_grad():
        out = model(images)
    assert out.shape == (2, 8, 1024, 1024), out.shape
    print(f"[OK] Forward pass: output={tuple(out.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"D:\kty\目标球智能体\datasets\combine_data",
    )
    args = parser.parse_args()
    data_root = Path(args.data_root)

    print("=== Verify combine_data ===")
    verify_files(data_root)
    verify_dataloader(data_root)
    print("=== All checks passed ===")


if __name__ == "__main__":
    main()
