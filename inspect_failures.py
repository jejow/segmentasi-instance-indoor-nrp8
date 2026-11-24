"""Find and render inspectable failure cases on labeled objects."""
import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask

from audit_private_test import PHOTO
from metrics import write_json
from prepare_dataset import hash_file


def match(annotation, predictions, names):
    if not predictions:
        return {"same_class_iou": 0.0, "nearest_class": None, "nearest_iou": 0.0}, None
    overlaps = coco_mask.iou(
        [row["segmentation"] for row in predictions], [annotation["segmentation"]], [0]
    )[:, 0]
    best = max(range(len(predictions)), key=lambda idx: (overlaps[idx], predictions[idx]["score"]))
    same = [i for i, row in enumerate(predictions) if row["category_id"] == annotation["category_id"]]
    same_best = max(same, key=lambda idx: (overlaps[idx], predictions[idx]["score"])) if same else None
    chosen = same_best if same_best is not None and overlaps[same_best] else best
    stats = {
        "same_class_iou": float(overlaps[same_best]) if same_best is not None else 0.0,
        "nearest_class": names[predictions[best]["category_id"]] if overlaps[best] else None,
        "nearest_iou": float(overlaps[best]),
        "shown_class": names[predictions[chosen]["category_id"]] if overlaps[chosen] else None,
        "shown_iou": float(overlaps[chosen]),
        "shown_score": predictions[chosen]["score"] if overlaps[chosen] else None,
    }
    return stats, predictions[chosen] if overlaps[chosen] else None


def overlay(image, annotation, color):
    if annotation is None:
        return image
    binary = Image.fromarray(coco_mask.decode(annotation["segmentation"])).convert("L")
    alpha = binary.point(lambda val: 105 if val else 0)
    layer = image.copy()
    layer.paste(color, (0, 0), alpha)
    return layer


def inspect(dataset, model_archives, output, requested, render_ids):
    outputs = {}
    dataset_sha256 = hash_file(dataset)
    for model in model_archives:
        by_photo = defaultdict(list)
        with ZipFile(model) as archive:
            summary = json.loads(archive.read("summary.json"))
            for prediction in json.loads(archive.read("predictions-private.json")):
                by_photo[prediction["image_id"]].append(prediction)
        if summary["dataset_sha256"] != dataset_sha256 or summary["model"] in outputs:
            raise ValueError(f"Wrong evaluation snapshot or duplicate model: {model}")
        outputs[summary["model"]] = by_photo
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(dataset) as archive:
        gt = json.loads(archive.read("private/gt.json"))
        names = {c["id"]: c["name"] for c in gt["categories"]}
        items = {row["id"]: row for row in gt["images"]}
        candidates = []
        for annotation in gt["annotations"]:
            file = items[annotation["image_id"]]
            photo = PHOTO.search(file["source_name"]).group()
            if requested and photo not in requested or annotation["iscrowd"]:
                continue
            label = names[annotation["category_id"]]
            models, examples = {}, {}
            for model, predictions in outputs.items():
                stats, example = match(annotation, predictions.get(file["id"], []), names)
                models[model], examples[model] = stats, example
            if min(row["same_class_iou"] for row in models.values()) < 0.50:
                candidates.append({
                    "photo": photo, "annotation_id": annotation["id"], "class": label,
                    "area": annotation["area"], "bbox": annotation["bbox"], "models": models,
                })
            if annotation["id"] in render_ids:
                pixels = Image.open(BytesIO(archive.read(f"private/images/{file['file_name']}"))).convert("RGB")
                panels = [("Anotasi COCO", label, overlay(pixels, annotation, "#10a68d"))]
                for model, row in models.items():
                    model_label = row.get("shown_class") or "tanpa overlap"
                    panels.append((
                        model, f"{model_label} | IoU {row.get('shown_iou', 0):.2f}",
                        overlay(pixels, examples[model], "#eb6e38"),
                    ))
                width, height = 300, 398
                canvas = Image.new("RGB", (width * len(panels), height), "#f4f7fa")
                draw = ImageDraw.Draw(canvas)
                heading = ImageFont.load_default(size=20)
                caption_font = ImageFont.load_default(size=16)
                draw.text((10, 10), f"{photo} | {label} | instance {annotation['id']}",
                          fill="#172c45", font=heading)
                for i, (name, caption, panel) in enumerate(panels):
                    canvas.paste(panel.resize((width, width), Image.Resampling.LANCZOS), (i * width, 41))
                    draw.text((i * width + 9, 348), name, fill="#172c45", font=heading)
                    draw.text((i * width + 9, 374), caption, fill="#172c45", font=caption_font)
                canvas.save(output / f"{photo}-{annotation['id']}.jpg", quality=92)
    write_json(output / "candidates.json", candidates)
    print(f"{len(candidates)} kandidat IoU kelas yang sama < 0,50 pada minimal satu model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--render", nargs="*", type=int, default=[])
    options = parser.parse_args()
    photos = set(json.loads(options.photos.read_text(encoding="utf-8"))) if options.photos else set()
    inspect(options.dataset, options.models, options.output, photos, set(options.render))
