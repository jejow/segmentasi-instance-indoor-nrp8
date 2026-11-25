"""Build report panels from archived COCO masks and predictions."""
from collections import defaultdict
from io import BytesIO
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask

from audit_private_test import PHOTO
from inspect_failures import match, overlay
from prepare_dataset import hash_file


PUBLIC = Path("bukti/visual")
LOCAL = Path("output/report/visual")
EVAL = Path("output/test-pribadi/eval-v3")
LABELS = ("YOLOv8n-seg", "YOLO11n-seg", "Mask R-CNN")
MODELS = ("yolov8n", "yolo11n", "maskrcnn")
FULL = ("full-yolov8n-r2", "full-yolo11n-r3", "full-maskrcnn-eval")
THREE = ("three-yolov8n", "three-yolo11n", "three-maskrcnn-r3")
CASES = (372, 548, 183, 106, 200)
RESAMPLE = Image.Resampling.LANCZOS
REFERENCE = "#008e7d"
PREDICTION = "#d95116"


def read(archive, member):
    return json.loads(archive.read(member))


def get_predictions(archives, dataset_hash, split, image_ids):
    result = []
    for path, model in zip(archives, MODELS, strict=True):
        with ZipFile(path) as archive:
            summary = read(archive, "summary.json")
            if summary["dataset_sha256"] != dataset_hash or summary["model"] != model:
                raise ValueError(f"Unexpected dataset or model in {path}")
            by_image = defaultdict(list)
            for row in read(archive, f"predictions-{split}.json"):
                if row["image_id"] in image_ids:
                    by_image[row["image_id"]].append(row)
            result.append(by_image)
    return result


def photo(archive, split, row):
    with Image.open(BytesIO(archive.read(f"{split}/images/{row['file_name']}"))) as pixels:
        return pixels.convert("RGB")


def grid(destination, rows, size=250):
    columns = len(rows[0])
    row_height = size + 57
    gap = 10
    canvas = Image.new("RGB", (columns * (size + gap) - gap, len(rows) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.load_default(size=19)
    detail = ImageFont.load_default(size=17)
    for row_index, row in enumerate(rows):
        for column_index, (title, subtitle, pixels) in enumerate(row):
            x = column_index * (size + gap)
            y = row_index * row_height
            draw.text((x + 2, y + 2), title, font=heading, fill="#181818")
            canvas.paste(pixels.resize((size, size), RESAMPLE), (x, y + 27))
            draw.text((x + 2, y + size + 31), subtitle, font=detail, fill="#373737")
    canvas.save(destination, quality=92, optimize=True)


def comparison(reference, prediction):
    truth = coco_mask.decode(reference["segmentation"]).astype(bool)
    pred = np.zeros_like(truth) if prediction is None else coco_mask.decode(prediction["segmentation"]).astype(bool)
    pixels = np.full((*truth.shape, 3), 245, dtype=np.uint8)
    pixels[truth & ~pred] = (0, 116, 102)
    pixels[pred & ~truth] = (210, 77, 19)
    pixels[truth & pred] = (28, 53, 106)
    return Image.fromarray(pixels)


def result_note(stats):
    if not stats.get("shown_class"):
        return "Tak ada overlap"
    return f"{stats['shown_class']} / {stats['shown_iou']:.2f}"


def public_examples():
    dataset = Path("data/panoptic-indoor-v5-640.zip")
    digest = hash_file(dataset)
    with ZipFile(dataset) as archive:
        ground_truth = read(archive, "test/gt.json")
        names = {row["id"]: row["name"] for row in ground_truth["categories"]}
        images = {row["id"]: row for row in ground_truth["images"]}
        targets = {}
        for target in ("knife", "light", "spoon"):
            targets[target] = max(
                (row for row in ground_truth["annotations"]
                 if names[row["category_id"]] == target and not row["iscrowd"]),
                key=lambda row: (row["area"], -row["id"]),
            )
        ids = {row["image_id"] for row in targets.values()}
        full, three = [
            get_predictions([Path("output/runs") / run / "result.zip" for run in runs],
                            digest, "test", ids)
            for runs in (FULL, THREE)
        ]
        for label, reference in targets.items():
            image_id = reference["image_id"]
            pixels = photo(archive, "test", images[image_id])
            rows = [[("Citra uji v5", f"Image ID {image_id}", pixels)],
                    [("Mask COCO", f"{label} / instance {reference['id']}",
                      overlay(pixels, reference, REFERENCE))]]
            for suffix, results, row in zip(("semua", "3 kelas"), (full, three), rows, strict=True):
                for model, predictions in zip(LABELS, results, strict=True):
                    candidates = [pred for pred in predictions[image_id]
                                  if pred["category_id"] == reference["category_id"]]
                    stats, chosen = match(reference, candidates, names)
                    tile = overlay(pixels, chosen, PREDICTION)
                    note = (f"IoU {stats['shown_iou']:.2f} / skor {stats['shown_score']:.3f}"
                            if chosen is not None else "Tak ada overlap sekelas")
                    row.append((f"{model} / {suffix}", note, tile))
            grid(PUBLIC / f"v5-{label}.jpg", rows, size=260)


def phone_examples():
    case_eval = EVAL.with_name("eval-v2")
    dataset = case_eval / "photo-test-640.zip"
    digest = hash_file(dataset)
    difficult = set(json.loads(Path("difficult-photos.json").read_text(encoding="utf-8")))
    with Path("bukti/hasil/kasus-iou.csv").open(newline="", encoding="utf-8") as stream:
        expected = {(int(row["case"]), row["model"]): row for row in csv.DictReader(stream)}
    with ZipFile(dataset) as archive:
        ground_truth = read(archive, "private/gt.json")
        names = {row["id"]: row["name"] for row in ground_truth["categories"]}
        images = {row["id"]: row for row in ground_truth["images"]}
        annotations = {row["id"]: row for row in ground_truth["annotations"]}
        ids = {annotations[annotation_id]["image_id"] for annotation_id in CASES}
        results = get_predictions([case_eval / f"{run}.zip" for run in FULL], digest, "private", ids)
        for case, annotation_id in enumerate(CASES, 1):
            reference = annotations[annotation_id]
            image_id = reference["image_id"]
            label = names[reference["category_id"]]
            pixels = photo(archive, "private", images[image_id])
            private = [("Foto HP", f"Kasus {case}", pixels),
                       ("Mask COCO", label, overlay(pixels, reference, REFERENCE))]
            silhouette = [("Mask COCO", label, comparison(reference, None))]
            for model, display, predictions in zip(MODELS, LABELS, results, strict=True):
                stats, chosen = match(reference, predictions[image_id], names)
                check = expected[case, model]
                if (check["reference_label"] != label or check["shown_label"] != stats["shown_class"]
                        or abs(float(check["shown_iou"]) - stats["shown_iou"]) > 1e-8
                        or abs(float(check["same_class_iou"]) - stats["same_class_iou"]) > 1e-8):
                    raise ValueError(f"Case {case} disagrees with recorded IoU for {model}")
                note = result_note(stats)
                private.append((display, note, overlay(pixels, chosen, PREDICTION)))
                silhouette.append((display, note, comparison(reference, chosen)))
            grid(LOCAL / f"kasus-{case}.jpg", [private], size=250)
            grid(PUBLIC / f"kasus-{case}-mask.jpg", [silhouette], size=250)

    with ZipFile(EVAL / "photo-test-640.zip") as archive:
        ground_truth = read(archive, "private/gt.json")
        counts = defaultdict(int)
        for annotation in ground_truth["annotations"]:
            counts[annotation["image_id"]] += 1
        catalogue = []
        for order, image in enumerate(sorted(ground_truth["images"], key=lambda row: row["id"]), 1):
            source_id = PHOTO.search(image["source_name"]).group()
            condition = "Sulit" if source_id in difficult else "Reguler"
            catalogue.append((f"Foto {order:02d} / {condition}", f"{counts[image['id']]} mask acuan",
                              photo(archive, "private", image)))
        if len(catalogue) != 40 or sum("Sulit" in row[0] for row in catalogue) != 12:
            raise ValueError("Unexpected phone test set count")
        for start in range(0, len(catalogue), 8):
            tiles = catalogue[start:start + 8]
            grid(LOCAL / f"galeri-{start // 8 + 1}.jpg", [tiles[:4], tiles[4:]], size=225)


if __name__ == "__main__":
    PUBLIC.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    public_examples()
    phone_examples()
