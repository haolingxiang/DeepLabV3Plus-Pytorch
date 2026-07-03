"""Export inference results with per-pixel class codes and Chinese labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_combined_class_info(data_root: str | None) -> list[dict]:
    from datasets.combined_landcover import DEFAULT_CLASSES

    if not data_root:
        return DEFAULT_CLASSES

    meta_path = Path(data_root) / "class_names.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("classes", DEFAULT_CLASSES)
    return DEFAULT_CLASSES


def build_class_legend(class_info: list[dict]) -> dict:
    legend = {}
    for item in class_info:
        cls_id = int(item["id"])
        legend[str(cls_id)] = {
            "code": cls_id,
            "name": item.get("name", ""),
            "name_zh": item.get("name_zh", item.get("name", "")),
            "color": item.get("color", [0, 0, 0]),
        }
    return {
        "description": "像素值即类别码 code；name_zh 为中文类别名",
        "classes": legend,
    }


def pred_to_zh_grid(pred: np.ndarray, class_info: list[dict]) -> np.ndarray:
    id_to_zh = {int(item["id"]): item.get("name_zh", item.get("name", "")) for item in class_info}
    lookup = np.vectorize(lambda x: id_to_zh.get(int(x), "未知"), otypes=[object])
    return lookup(pred)


def summarize_prediction(pred: np.ndarray, class_info: list[dict]) -> dict:
    total = int(pred.size)
    counts = np.bincount(pred.ravel().astype(np.int64), minlength=256)
    by_class = []
    for item in class_info:
        cls_id = int(item["id"])
        cnt = int(counts[cls_id])
        by_class.append(
            {
                "code": cls_id,
                "name": item.get("name", ""),
                "name_zh": item.get("name_zh", item.get("name", "")),
                "pixels": cnt,
                "ratio": round(cnt / total, 6) if total else 0.0,
            }
        )
    return {"total_pixels": total, "by_class": by_class}


def write_class_legend(out_dir: Path, class_info: list[dict]):
    legend_path = out_dir / "class_legend.json"
    if legend_path.exists():
        return
    with legend_path.open("w", encoding="utf-8") as f:
        json.dump(build_class_legend(class_info), f, ensure_ascii=False, indent=2)


def export_prediction(
    pred: np.ndarray,
    img_name: str,
    out_dir: Path,
    decode_fn,
    class_info: list[dict] | None = None,
    save_pixel_zh: bool = True,
):
    """
    Save:
      color/{name}.png           - 彩色可视化
      class_codes/{name}.png     - 单通道类别码 (0~num_classes-1)
      pixel_class_zh/{name}.npz  - codes + 每个像素中文类别名
      summaries/{name}.json      - 各类像素统计（含中文名）
    """
    out_dir = Path(out_dir)
    color_dir = out_dir / "color"
    code_dir = out_dir / "class_codes"
    zh_dir = out_dir / "pixel_class_zh"
    summary_dir = out_dir / "summaries"
    for d in (color_dir, code_dir, zh_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)

    if class_info is not None:
        write_class_legend(out_dir, class_info)

    color = Image.fromarray(decode_fn(pred).astype("uint8"))
    color.save(color_dir / f"{img_name}.png")

    Image.fromarray(pred.astype(np.uint8), mode="L").save(code_dir / f"{img_name}.png")

    if class_info is not None:
        summary = summarize_prediction(pred, class_info)
        with (summary_dir / f"{img_name}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if save_pixel_zh:
            zh_grid = pred_to_zh_grid(pred, class_info)
            np.savez_compressed(
                zh_dir / f"{img_name}.npz",
                codes=pred.astype(np.uint8),
                class_zh=zh_grid,
            )
