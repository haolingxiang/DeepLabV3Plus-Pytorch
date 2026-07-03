"""Pad/slide-window inference helpers for semantic segmentation."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

PAD_FILL = (0, 0, 0)


def tile_starts(size: int, tile_size: int, stride: int) -> list[int]:
    """Same grid logic as prepare_combined_dataset.py."""
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


def extract_tile(img: Image.Image, x: int, y: int, tile_size: int) -> tuple[Image.Image, int, int]:
    """Crop a tile and pad to tile_size x tile_size when near image borders."""
    w, h = img.size
    x_end = min(x + tile_size, w)
    y_end = min(y + tile_size, h)
    valid_w = x_end - x
    valid_h = y_end - y

    crop = img.crop((x, y, x_end, y_end))
    if valid_w < tile_size or valid_h < tile_size:
        padded = Image.new("RGB", (tile_size, tile_size), PAD_FILL)
        padded.paste(crop, (0, 0))
        return padded, valid_h, valid_w
    return crop, valid_h, valid_w


def _accumulate_logits(
    acc: np.ndarray,
    counts: np.ndarray,
    logits: np.ndarray,
    y: int,
    x: int,
    valid_h: int,
    valid_w: int,
):
    acc[:, y : y + valid_h, x : x + valid_w] += logits[:, :valid_h, :valid_w]
    counts[y : y + valid_h, x : x + valid_w] += 1.0


def _logits_to_pred(acc: np.ndarray, counts: np.ndarray) -> np.ndarray:
    counts = np.maximum(counts, 1.0)
    return (acc / counts[np.newaxis, :, :]).argmax(axis=0).astype(np.uint8)


@torch.no_grad()
def _forward_logits(model, transform, tile: Image.Image, device) -> np.ndarray:
    tensor = transform(tile).unsqueeze(0).to(device)
    return model(tensor)[0].cpu().numpy()


def infer_padded(
    model,
    img: Image.Image,
    transform,
    device,
    tile_size: int = 1024,
) -> np.ndarray:
    """Pad image to tile_size x tile_size, infer, crop back to original size."""
    w, h = img.size
    padded = Image.new("RGB", (tile_size, tile_size), PAD_FILL)
    padded.paste(img, (0, 0))
    logits = _forward_logits(model, transform, padded, device)
    return logits[:, :h, :w].argmax(axis=0).astype(np.uint8)


def infer_sliding_window(
    model,
    img: Image.Image,
    transform,
    device,
    num_classes: int,
    tile_size: int = 1024,
    stride: int = 1024,
) -> np.ndarray:
    """
    Slide tile_size windows with given stride and fuse overlapping regions
    by averaging logits (standard for semantic segmentation; NMS not needed).
    """
    w, h = img.size
    xs = tile_starts(w, tile_size, stride)
    ys = tile_starts(h, tile_size, stride)

    acc = np.zeros((num_classes, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in ys:
        for x in xs:
            tile, valid_h, valid_w = extract_tile(img, x, y, tile_size)
            logits = _forward_logits(model, transform, tile, device)
            _accumulate_logits(acc, counts, logits, y, x, valid_h, valid_w)

    return _logits_to_pred(acc, counts)


def infer_image(
    model,
    img: Image.Image,
    transform,
    device,
    num_classes: int,
    tile_size: int = 1024,
    stride: int = 1024,
) -> np.ndarray:
    """
    Route by image size:
    - both sides <= tile_size: pad to tile_size, infer, crop back
    - either side > tile_size: sliding window + logit fusion + stitch
    """
    w, h = img.size
    if h > tile_size or w > tile_size:
        return infer_sliding_window(
            model, img, transform, device, num_classes, tile_size, stride
        )
    return infer_padded(model, img, transform, device, tile_size)
