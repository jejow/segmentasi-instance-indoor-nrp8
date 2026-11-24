"""Measure a fixed gamma perturbation on the four preidentified dark photos."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image
import torch

from audit_private_test import PHOTO
from metrics import score_masks, write_json
from train import category_table, predict_maskrcnn


DARK = {"1000865809", "1000865813", "1000865828", "1000865829"}


def experiment(dataset, trained_run, baseline_run, metadata, output, gamma):
    original_sizes = {
        Path(row["file"]).stem: (row["width"], row["height"])
        for row in json.loads(metadata.read_text(encoding="utf-8"))
    }
    lookup = [round(255 * (value / 255) ** gamma) for value in range(256)]
    with TemporaryDirectory(prefix="indoor-gamma-") as work, ZipFile(dataset) as images:
        gt = json.loads(images.read("private/gt.json"))
        manifest = json.loads(images.read("manifest.json"))
        used = {
            image["id"]: (image, PHOTO.search(image["source_name"]).group())
            for image in gt["images"] if PHOTO.search(image["source_name"]).group() in DARK
        }
        if {photo for _, photo in used.values()} != DARK:
            raise ValueError("Missing preselected low-light photo.")
        gt["images"] = [image for image, _ in used.values()]
        gt["annotations"] = [row for row in gt["annotations"] if row["image_id"] in used]
        size = manifest["size"]
        directory = Path(work) / "images"
        directory.mkdir()
        for image, photo in used.values():
            name = image["file_name"]
            width, height = original_sizes[photo]
            scale = size / max(width, height)
            fit_w, fit_h = round(scale * width), round(scale * height)
            left, top = (size - fit_w) // 2, (size - fit_h) // 2
            pixels = Image.open(BytesIO(images.read(f"private/images/{name}"))).convert("RGB")
            interior = pixels.crop((left, top, left + fit_w, top + fit_h))
            pixels.paste(interior.point(lookup * 3), (left, top))
            pixels.save(directory / name, format="PNG")
        checkpoint = Path(work) / "checkpoint.pt"
        with ZipFile(trained_run) as training, ZipFile(baseline_run) as evaluation:
            baseline = json.loads(evaluation.read("summary.json"))
            summary = json.loads(training.read("summary.json"))
            if any(baseline[k] != summary[k] for k in ("seed", "class_names", "model", "input_size")):
                raise ValueError("Pilot and control do not use the same checkpoint settings.")
            with training.open("checkpoint.pt") as weights, checkpoint.open("wb") as sink:
                copyfileobj(weights, sink)
            control = [
                row for row in json.loads(evaluation.read("predictions-private.json"))
                if row["image_id"] in used
            ]
        model = argparse.Namespace(model=summary["model"], device="cpu")
        categories = category_table(manifest, summary["class_names"])
        brightened = list(predict_maskrcnn(model, directory, gt, checkpoint, categories, size))
        report = {
            "gamma": gamma, "photos": sorted(DARK), "instances": len(gt["annotations"]),
            "model": summary["model"], "dataset_sha256": baseline["dataset_sha256"],
            "control_predictions": len(control), "brightened_predictions": len(brightened),
            "control": score_masks(gt, control, categories),
            "brightened": score_masks(gt, brightened, categories),
        }
        write_json(output, report)
        print(f"Gamma {gamma}: AP {report['control']['ap']:.4f} -> {report['brightened']['ap']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("trained_run", type=Path)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gamma", type=float, default=0.65)
    options = parser.parse_args()
    if options.gamma <= 0:
        parser.error("Gamma must be positive.")
    torch.set_num_threads(4)
    experiment(options.dataset, options.trained_run, options.baseline_run,
               options.metadata, options.output, options.gamma)
