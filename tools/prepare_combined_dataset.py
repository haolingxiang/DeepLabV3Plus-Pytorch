"""
Merge DeepGlobe + LoveDA into combine_data with unified labels and 1024 tiles.

Default stride=1024 (tile size): 2048->2x2=4 tiles, 2448->3x3=9 tiles, full coverage.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DEEPGLOBE_LUT = {
    (rgb_val[0] << 16) | (rgb_val[1] << 8) | rgb_val[2]: cls_id
    for rgb_val, cls_id in {
        (0, 0, 0): 0,
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (255, 0, 255): 3,
        (255, 255, 0): 4,
        (255, 255, 255): 5,
        (0, 255, 255): 6,
    }.items()
}

LOVEDA_LUT = np.full(256, 255, dtype=np.uint8)
for _src, _dst in {0: 255, 1: 0, 2: 6, 3: 7, 4: 1, 5: 5, 6: 2, 7: 4}.items():
    LOVEDA_LUT[_src] = _dst

CLASS_INFO = [
    {"id": 0, "name": "background", "name_zh": "背景", "color": [0, 0, 0]},
    {"id": 1, "name": "water", "name_zh": "水体", "color": [0, 0, 255]},
    {"id": 2, "name": "forest", "name_zh": "森林", "color": [0, 255, 0]},
    {"id": 3, "name": "grassland", "name_zh": "草地", "color": [255, 0, 255]},
    {"id": 4, "name": "farmland", "name_zh": "农田", "color": [255, 255, 0]},
    {"id": 5, "name": "barren", "name_zh": "裸地", "color": [200, 200, 200]},
    {"id": 6, "name": "building", "name_zh": "建筑", "color": [0, 255, 255]},
    {"id": 7, "name": "road", "name_zh": "道路", "color": [255, 128, 0]},
]


def tile_starts(size: int, tile_size: int, stride: int) -> list[int]:
    """Return tile start positions that fully cover [0, size) without gaps."""
    if size <= tile_size:
        return [0]

    starts = [0]
    while starts[-1] + tile_size < size:
        next_start = starts[-1] + stride
        last_start = size - tile_size
        prev_end = starts[-1] + tile_size

        if next_start >= last_start:
            if last_start > starts[-1]:
                starts.append(last_start)
            break

        # Large stride can leave uncovered pixels; abut the next tile instead.
        if next_start >= prev_end:
            next_start = prev_end

        if next_start <= starts[-1]:
            next_start = starts[-1] + 1

        if next_start + tile_size > size:
            if last_start > starts[-1]:
                starts.append(last_start)
            break

        starts.append(next_start)

    return sorted(set(starts))


def preview_tile_grid(size: int, tile_size: int, stride: int) -> tuple[int, list[int]]:
    starts = tile_starts(size, tile_size, stride)
    return len(starts), starts


def deepglobe_rgb_to_mask(rgb: np.ndarray) -> np.ndarray:
    packed = (
        rgb[..., 0].astype(np.uint32) << 16
        | rgb[..., 1].astype(np.uint32) << 8
        | rgb[..., 2].astype(np.uint32)
    )
    flat = packed.ravel()
    unique, inv = np.unique(flat, return_inverse=True)
    mapped = np.fromiter(
        (DEEPGLOBE_LUT.get(int(u), 0) for u in unique),
        dtype=np.uint8,
        count=len(unique),
    )
    return mapped[inv].reshape(rgb.shape[:2])


def loveda_to_mask(arr: np.ndarray) -> np.ndarray:
    return LOVEDA_LUT[arr]


def valid_ratio(mask: np.ndarray) -> float:
    return float((mask != 255).sum()) / mask.size


def update_stats_from_mask(stats: Counter, mask: np.ndarray):
    valid = mask.ravel()
    valid = valid[valid != 255]
    if valid.size == 0:
        return
    counts = np.bincount(valid.astype(np.int64), minlength=8)
    for cls_id, count in enumerate(counts):
        if count:
            stats[cls_id] += int(count)


def save_pair(img_arr, mask_arr, img_path: Path, mask_path: Path, compress_level: int = 1):
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_arr).save(img_path, compress_level=compress_level)
    Image.fromarray(mask_arr, mode="L").save(mask_path, compress_level=compress_level)


def process_deepglobe(
    root: Path,
    output_root: Path,
    tile_size: int,
    stride: int,
    min_valid_ratio: float,
    split_map: dict[str, str],
    stats: Counter,
    manifest: dict[str, list],
):
    metadata = root / "metadata.csv"
    if not metadata.exists():
        print(f"[WARN] DeepGlobe metadata not found: {metadata}")
        return

    with metadata.open(newline="", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["split"] in split_map and row.get("mask_path")
        ]

    for row in tqdm(rows, desc="DeepGlobe"):
        out_split = split_map[row["split"]]
        sat_rel = row["sat_image_path"].replace("\\", "/")
        mask_rel = row["mask_path"].replace("\\", "/")

        img_path = root / sat_rel
        mask_path = root / mask_rel
        if not img_path.exists() or not mask_path.exists():
            continue

        with Image.open(img_path) as img_f, Image.open(mask_path) as mask_f:
            image = np.asarray(img_f.convert("RGB"))
            mask_full = deepglobe_rgb_to_mask(np.asarray(mask_f.convert("RGB")))
        h, w = image.shape[:2]

        xs = tile_starts(w, tile_size, stride)
        ys = tile_starts(h, tile_size, stride)
        image_id = row["image_id"]

        for y in ys:
            for x in xs:
                tile_img = image[y : y + tile_size, x : x + tile_size]
                tile_mask = mask_full[y : y + tile_size, x : x + tile_size]

                if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                    continue
                if valid_ratio(tile_mask) < min_valid_ratio:
                    continue

                name = f"dg_{image_id}_x{x}_y{y}.png"
                out_img = output_root / "images" / out_split / name
                out_mask = output_root / "masks" / out_split / name
                save_pair(tile_img, tile_mask, out_img, out_mask)

                rel_img = f"images/{out_split}/{name}"
                rel_mask = f"masks/{out_split}/{name}"
                manifest[out_split].append((rel_img, rel_mask))
                update_stats_from_mask(stats, tile_mask)


def process_loveda(
    root: Path,
    output_root: Path,
    min_valid_ratio: float,
    split_map: dict[str, str],
    stats: Counter,
    manifest: dict[str, list],
):
    for loveda_split, out_split in split_map.items():
        split_dir = root / loveda_split
        if not split_dir.exists():
            print(f"[WARN] LoveDA split not found: {split_dir}")
            continue

        for scene in ("Rural", "Urban"):
            img_dir = split_dir / scene / "images_png"
            mask_dir = split_dir / scene / "masks_png"
            if not img_dir.exists() or not mask_dir.exists():
                continue

            images = sorted(img_dir.glob("*.png"))
            for img_path in tqdm(images, desc=f"LoveDA {loveda_split}/{scene}"):
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    continue

                image = np.asarray(Image.open(img_path).convert("RGB"))
                mask = loveda_to_mask(np.asarray(Image.open(mask_path)))
                if valid_ratio(mask) < min_valid_ratio:
                    continue

                stem = img_path.stem
                name = f"loveda_{loveda_split.lower()}_{scene.lower()}_{stem}.png"
                out_img = output_root / "images" / out_split / name
                out_mask = output_root / "masks" / out_split / name
                save_pair(image, mask, out_img, out_mask)

                rel_img = f"images/{out_split}/{name}"
                rel_mask = f"masks/{out_split}/{name}"
                manifest[out_split].append((rel_img, rel_mask))
                update_stats_from_mask(stats, mask)


def write_manifest(output_root: Path, manifest: dict[str, list]):
    for split, pairs in manifest.items():
        txt_path = output_root / f"{split}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for img_rel, mask_rel in pairs:
                f.write(f"{img_rel} {mask_rel}\n")


def write_meta(output_root: Path, stats: Counter, manifest: dict[str, list], args):
    class_names = {
        "num_classes": 8,
        "ignore_index": 255,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "classes": CLASS_INFO,
    }
    with (output_root / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    total_pixels = sum(stats.values())
    class_stats = {}
    for cls in range(8):
        cnt = stats.get(cls, 0)
        class_stats[str(cls)] = {
            "name": CLASS_INFO[cls]["name"],
            "pixels": cnt,
            "ratio": round(cnt / total_pixels, 6) if total_pixels else 0.0,
        }

    summary = {
        "train_samples": len(manifest["train"]),
        "val_samples": len(manifest["val"]),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "class_pixels": class_stats,
    }
    with (output_root / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare combined DeepGlobe + LoveDA dataset")
    parser.add_argument(
        "--deepglobe_root",
        type=str,
        default=r"D:\kty\目标球智能体\datasets\DeepGlobe-Land-cover",
    )
    parser.add_argument(
        "--loveda_root",
        type=str,
        default=r"D:\kty\目标球智能体\datasets\LoveDA",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=r"D:\kty\目标球智能体\datasets\combine_data",
    )
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024,
                        help="sliding window stride (default: 1024, same as tile_size)")
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.0,
        help="skip tiles with less valid labeled pixels (0~1)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_map_dg = {"train": "train", "valid": "val"}
    split_map_ld = {"Train": "train", "Val": "val"}

    stats = Counter()
    manifest = defaultdict(list)

    process_deepglobe(
        Path(args.deepglobe_root),
        output_root,
        args.tile_size,
        args.stride,
        args.min_valid_ratio,
        split_map_dg,
        stats,
        manifest,
    )
    process_loveda(
        Path(args.loveda_root),
        output_root,
        args.min_valid_ratio,
        split_map_ld,
        stats,
        manifest,
    )

    write_manifest(output_root, manifest)
    write_meta(output_root, stats, manifest, args)

    n2048, s2048 = preview_tile_grid(2048, args.tile_size, args.stride)
    n2448, s2448 = preview_tile_grid(2448, args.tile_size, args.stride)
    print("\n=== Done ===")
    print(f"Output: {output_root}")
    print(f"Tile size: {args.tile_size}, stride: {args.stride}")
    print(f"2048 image grid: {n2048}x{n2048}={n2048 ** 2} tiles, starts={s2048}")
    print(f"2448 image grid: {n2448}x{n2448}={n2448 ** 2} tiles, starts={s2448}")
    print(f"Train samples: {len(manifest['train'])}")
    print(f"Val samples:   {len(manifest['val'])}")
    print(f"Grassland pixels: {stats.get(3, 0)} ({stats.get(3, 0) / max(sum(stats.values()), 1):.4%})")


if __name__ == "__main__":
    main()
