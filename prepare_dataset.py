"""Prepare a Roboflow COCO export on one common 640px evaluation grid."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from statistics import median
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as coco_mask

OFFICIAL_SPLITS = {"train": 397, "valid": 166, "test": 359}
SOURCE = "https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5"
ORIENTATION_FIXES = {
    "182169107_jpg.rf.c9b1bbecf1474a5b36a6df197c30dfed.jpg": Image.Transpose.ROTATE_270,
    "182169121_jpg.rf.eb0ccb71d0ea38adf1e38bf9c02a4a15.jpg": Image.Transpose.ROTATE_270,
    "182169109_jpg.rf.0531e60eb6e404d9633ed0c23a381fe7.jpg": Image.Transpose.ROTATE_90,
    "182169108_jpg.rf.205d9f1ed8f888006b8f97fc29eb8790.jpg": Image.Transpose.ROTATE_270,
}


def encode(mask):
    rle = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def letterbox(image, size):
    width, height = image.size
    scale = size / max(width, height)
    fit_w, fit_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - fit_w) // 2, (size - fit_h) // 2
    result = Image.new("RGB", (size, size), (114,) * 3)
    result.paste(image.resize((fit_w, fit_h), Image.Resampling.BILINEAR), (left, top))
    return result, (fit_w, fit_h, left, top)


def read_annotation_files(archive):
    result = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.name == "_annotations.coco.json":
            split = path.parent.name
            if split in result:
                raise ValueError(f"More than one export JSON for {split}")
            result[split] = (path.parent, json.loads(archive.read(name)))
    if not result:
        raise ValueError("COCO export did not contain annotation files.")
    return result


def change_mask(segmentation, width, height, placement, size):
    fit_w, fit_h, left, top = placement
    if isinstance(segmentation, list):
        polygons = []
        for polygon in segmentation:
            if len(polygon) < 6 or len(polygon) % 2:
                continue
            points = np.array(polygon, dtype=np.float64).reshape(-1, 2)
            if not np.isfinite(points).all():
                continue
            points[:, 0] = np.clip(points[:, 0] * (fit_w / width), 0, fit_w) + left
            points[:, 1] = np.clip(points[:, 1] * (fit_h / height), 0, fit_h) + top
            polygons.append(points.reshape(-1).round(4).tolist())
        if not polygons:
            return None, []
        mask = coco_mask.merge(coco_mask.frPyObjects(polygons, size, size))
        mask["counts"] = mask["counts"].decode("ascii")
        return mask, polygons
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = coco_mask.frPyObjects(rle, height, width)
        raw = coco_mask.decode(rle)
        if raw.shape != (height, width):
            raise ValueError(f"RLE shape {raw.shape} differs from {(height, width)}.")
        padded = np.zeros((size, size), np.uint8)
        resized = cv2.resize(raw, (fit_w, fit_h), interpolation=cv2.INTER_NEAREST)
        padded[top:top+fit_h, left:left+fit_w] = resized
        contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.reshape(-1).tolist() for contour in contours if len(contour) >= 3]
        return encode(padded), polygons
    return None, []


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source, destination, size, categories_path=None):
    personal = categories_path is not None
    audit = {
        "source": "private Roboflow COCO export" if personal else SOURCE,
        "raw_sha256": hash_file(source),
        "size": size,
        "splits": {},
        "cross_split_exact_duplicates": [],
        "orientation_fixes": {},
    }
    hashes = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    with ZipFile(source) as raw, ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=5) as packed:
        splits = read_annotation_files(raw)
        if personal:
            if len(splits) != 1:
                raise ValueError("Supply one manually reviewed split for the external test.")
            splits = {"private": next(iter(splits.values()))}
            names = json.loads(categories_path.read_text(encoding="utf-8"))["categories"]
        else:
            counts = {split: len(data["images"]) for split, (_, data) in splits.items()}
            if counts != OFFICIAL_SPLITS:
                raise ValueError(f"Wrong dataset or official split counts: {counts}.")
            active = set()
            for _, data in splits.values():
                source_names = {x["id"]: x["name"] for x in data["categories"]}
                active.update(source_names[x["category_id"]] for x in data["annotations"])
            names = sorted(active)
        category_ids = {name: i for i, name in enumerate(names, start=1)}
        audit["categories"] = names

        for split, (prefix, data) in splits.items():
            source_names = {x["id"]: x["name"] for x in data["categories"]}
            by_image = defaultdict(list)
            for ann in data["annotations"]:
                by_image[ann["image_id"]].append(ann)
            gt = {"info": {"description": audit["source"]}, "images": [], "annotations": []}
            gt["categories"] = [{"id": i, "name": n} for n, i in category_ids.items()]
            counts = Counter()
            image_counts = Counter()
            missing_names = Counter()
            areas = defaultdict(list)
            skipped = Counter()
            for image in data["images"]:
                filename = image["file_name"]
                url_path = (prefix / filename).as_posix()
                pixels = raw.read(url_path)
                sha = hashlib.sha256(pixels).hexdigest()
                other = hashes.setdefault(sha, (split, filename))
                if other[0] != split:
                    audit["cross_split_exact_duplicates"].append([*other, split, filename])
                width, height = int(image["width"]), int(image["height"])
                with Image.open(BytesIO(pixels)) as opened:
                    if opened.size != (width, height):
                        opened = ImageOps.exif_transpose(opened)
                    if opened.size != (width, height) and not personal and filename in ORIENTATION_FIXES:
                        correction = ORIENTATION_FIXES[filename]
                        opened = opened.transpose(correction)
                        audit["orientation_fixes"][filename] = correction.name
                    if opened.size != (width, height):
                        raise ValueError(f"Image and annotation dimensions differ for {filename}.")
                    processed, placement = letterbox(opened.convert("RGB"), size)
                item_name = f'{image["id"]:06d}.jpg'
                buf = BytesIO()
                processed.save(buf, format="JPEG", quality=89, optimize=True)
                packed.writestr(f"{split}/images/{item_name}", buf.getvalue(), compress_type=ZIP_STORED)
                gt["images"].append({
                    "id": image["id"], "file_name": item_name, "width": size, "height": size,
                    "source_name": filename,
                })
                present = set()
                for ann in by_image[image["id"]]:
                    name = source_names[ann["category_id"]]
                    if name not in category_ids:
                        missing_names[name] += 1
                        continue
                    mask, polygons = change_mask(ann.get("segmentation"), width, height, placement, size)
                    if mask is None or int(coco_mask.area(mask)) == 0:
                        skipped[name] += 1
                        continue
                    area = int(coco_mask.area(mask))
                    gt["annotations"].append({
                        "id": len(gt["annotations"]) + 1, "image_id": image["id"],
                        "category_id": category_ids[name], "segmentation": mask, "polygons": polygons,
                        "area": area, "bbox": coco_mask.toBbox(mask).tolist(), "iscrowd": int(ann.get("iscrowd", 0)),
                    })
                    counts[name] += 1
                    present.add(name)
                    areas[name].append(area / (placement[0] * placement[1]))
                image_counts.update(present)
            if not gt["annotations"]:
                raise ValueError(f"No valid masks in {split}.")
            audit["splits"][split] = {
                "images": len(gt["images"]), "instances": len(gt["annotations"]),
                "class_instances": dict(counts), "class_images": dict(image_counts),
                "median_mask_fraction": {name: median(values) for name, values in areas.items()},
                "invalid_masks": dict(skipped), "classes_outside_taxonomy": dict(missing_names),
                "images_without_mapped_mask": sum(not by_image[x["id"]] or not any(
                    source_names[a["category_id"]] in category_ids for a in by_image[x["id"]]
                ) for x in data["images"]),
            }
            packed.writestr(f"{split}/gt.json", json.dumps(gt, separators=(",", ":")))
            print(f'{split}: {len(gt["images"])} images, {len(gt["annotations"])} masks', flush=True)
        packed.writestr("manifest.json", json.dumps(audit, separators=(",", ":")))
    temporary.replace(destination)
    manifest = destination.with_suffix(".json")
    manifest.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"{destination.stat().st_size:,} bytes. Prepared SHA256: {hash_file(destination)}.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--size", type=int, default=640)
    cli.add_argument("--external-taxonomy", type=Path)
    options = cli.parse_args()
    prepare(options.source, options.destination, options.size, options.external_taxonomy)
