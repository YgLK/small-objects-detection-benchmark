"""
Offline COCO-style dataset augmentation script using Albumentations.

This script performs offline augmentation of a dataset formatted in COCO JSON format.
Each image in the dataset is augmented multiple times using a predefined augmentation
pipeline that simulates common real-world transformations. The script outputs new
images and updates the COCO annotations accordingly.

Augmentation operations applied:
- Random rotation (±5 degrees)
- Random affine transformations including:
    - Scaling (range: 0.8 to 1.2)
    - Translation (up to 10%)
- Horizontal flip (50% chance)
- Vertical flip (20% chance)
- Padding (if needed) to ensure image is at least `img_size` x `img_size`
- Random cropping to fixed size (`img_size` x `img_size`)

Features:
- Keeps original dataset intact and copies unmodified images to output directory.
- Preserves annotation consistency (bounding boxes, category IDs).
- Filters out boxes with width or height ≤ 1 after augmentation.
- Generates unique filenames for augmented images using UUIDs.

Usage:
Update the call to `augment_dataset_coco()` at the bottom of the script with the
desired dataset paths and parameters. This script is designed to run standalone
and save the augmented dataset to a new folder.

Dependencies:
- albumentations
- OpenCV (cv2)
- tqdm
"""

import json
import os
from pathlib import Path
import shutil
import uuid

import albumentations as A
import cv2
from tqdm import tqdm


def load_coco_annotations(json_path: str) -> dict:
    """Load COCO annotations from a JSON file."""
    with open(json_path) as f:
        return json.load(f)


def save_coco_annotations(annotations: dict, output_path: str) -> None:
    """Save COCO annotations to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)


def build_augmentations(img_size: int = 640) -> A.Compose:
    """Create an augmentation pipeline similar to YOLOv8 config (without mosaic)."""
    return A.Compose(
        [
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),  # degrees=5.0
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), shear=0.0, p=0.7),  # translate=0.1, scale=0.2
            A.HorizontalFlip(p=0.5),  # fliplr=0.5
            A.VerticalFlip(p=0.2),  # flipud=0.2
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomCrop(height=img_size, width=img_size),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )


def augment_dataset_coco(
    input_img_dir: str,
    input_json_path: str,
    output_img_dir: str,
    output_json_path: str,
    num_aug_per_image: int = 3,
    img_size: int = 640,
) -> None:
    """
    Augment a COCO dataset offline and save augmented images + annotations.

    Args:
        input_img_dir: Path to original images.
        input_json_path: Path to original COCO .json annotation.
        output_img_dir: Path to save augmented images.
        output_json_path: Path to save new COCO .json with augmented annotations.
        num_aug_per_image: Number of augmentations to apply per image.
        img_size: Output image size (square).
    """
    os.makedirs(output_img_dir, exist_ok=True)

    coco_data = load_coco_annotations(input_json_path)
    aug = build_augmentations(img_size)
    next_image_id = max(img["id"] for img in coco_data["images"]) + 1
    next_ann_id = max(ann["id"] for ann in coco_data["annotations"]) + 1

    new_images = []
    new_annotations = []

    image_id_map = {img["file_name"]: img["id"] for img in coco_data["images"]}
    image_anns_map = {img_id: [] for img_id in image_id_map.values()}
    for ann in coco_data["annotations"]:
        image_anns_map[ann["image_id"]].append(ann)

    for img_dict in tqdm(coco_data["images"], desc="Augmenting"):
        img_path = os.path.join(input_img_dir, img_dict["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            continue

        anns = image_anns_map[img_dict["id"]]
        bboxes = [ann["bbox"] for ann in anns]
        cat_ids = [ann["category_id"] for ann in anns]

        for i in range(num_aug_per_image):
            try:
                augmented = aug(image=image, bboxes=bboxes, category_ids=cat_ids)
                aug_img = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_cats = augmented["category_ids"]
            except Exception:
                continue  # skip if augmentation fails (e.g. empty boxes)

            # save image
            new_filename = f"{Path(img_dict['file_name']).stem}_aug{i}_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(os.path.join(output_img_dir, new_filename), aug_img)

            # add image entry
            new_img_entry = {"id": next_image_id, "file_name": new_filename, "width": img_size, "height": img_size}
            new_images.append(new_img_entry)

            # add annotations
            for bbox, cat_id in zip(aug_bboxes, aug_cats):
                x, y, w, h = bbox
                if w <= 1 or h <= 1:
                    continue  # skip tiny boxes
                new_ann = {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
                new_annotations.append(new_ann)
                next_ann_id += 1

            next_image_id += 1

    # save original images as-is
    for img_dict in coco_data["images"]:
        shutil.copy(os.path.join(input_img_dir, img_dict["file_name"]), output_img_dir)

    all_images = coco_data["images"] + new_images
    all_annotations = coco_data["annotations"] + new_annotations

    coco_data["images"] = all_images
    coco_data["annotations"] = all_annotations
    save_coco_annotations(coco_data, output_json_path)


augment_dataset_coco(
    input_img_dir="/home/yglk/coding/dpm3-skyfusion_v3/dpm3/datasets/SkyFusion/train",
    input_json_path="/home/yglk/coding/dpm3-skyfusion_v3/dpm3/datasets/SkyFusion/train/_annotations.coco.json",
    output_img_dir="/home/yglk/coding/dpm3-skyfusion_v3/dpm3/datasets/SkyFusion/train_augmented",
    output_json_path="/home/yglk/coding/dpm3-skyfusion_v3/dpm3/datasets/SkyFusion/train_augmented/_annotations.coco.json",
    num_aug_per_image=3,
    img_size=640,
)
