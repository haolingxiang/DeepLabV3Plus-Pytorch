import json
import os

import numpy as np
import torch.utils.data as data
from PIL import Image


DEFAULT_CLASSES = [
    {"id": 0, "name": "background", "color": [0, 0, 0]},
    {"id": 1, "name": "water", "color": [0, 0, 255]},
    {"id": 2, "name": "forest", "color": [0, 255, 0]},
    {"id": 3, "name": "grassland", "color": [255, 0, 255]},
    {"id": 4, "name": "farmland", "color": [255, 255, 0]},
    {"id": 5, "name": "barren", "color": [200, 200, 200]},
    {"id": 6, "name": "building", "color": [0, 255, 255]},
    {"id": 7, "name": "road", "color": [255, 128, 0]},
]


def _build_cmap(class_info=None):
    cmap = np.zeros((256, 3), dtype=np.uint8)
    classes = class_info or DEFAULT_CLASSES
    for item in classes:
        cmap[item["id"]] = item["color"]
    return cmap


class CombinedLandCover(data.Dataset):
    """Dataset for combine_data/ produced by prepare_combined_dataset.py."""

    num_classes = 8
    ignore_index = 255
    cmap = _build_cmap()

    def __init__(self, root, split="train", transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform

        meta_path = os.path.join(self.root, "class_names.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            CombinedLandCover.cmap = _build_cmap(meta.get("classes"))
            self.num_classes = meta.get("num_classes", 8)
            self.ignore_index = meta.get("ignore_index", 255)

        list_path = os.path.join(self.root, f"{split}.txt")
        if not os.path.isfile(list_path):
            raise RuntimeError(f"Split file not found: {list_path}")

        self.images = []
        self.masks = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel, mask_rel = line.split()
                self.images.append(os.path.join(self.root, img_rel.replace("/", os.sep)))
                self.masks.append(os.path.join(self.root, mask_rel.replace("/", os.sep)))

        if not self.images:
            raise RuntimeError(f"No samples found in {list_path}")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.int64)
        else:
            mask = np.array(mask, dtype=np.int64)
        return cls.cmap[mask]
