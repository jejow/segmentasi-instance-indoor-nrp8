"""Prepare an indoor-only snapshot and evaluate fixed checkpoints locally."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from audit_private_test import PHOTO
from metrics import write_json
from prepare_dataset import prepare, read_annotation_files


def select_indoor_images(source, target, excluded):
    with source.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    with ZipFile(source) as raw:
        splits = read_annotation_files(raw)
        if len(splits) != 1:
            raise ValueError("Expected a single Roboflow job.")
        prefix, coco = next(iter(splits.values()))
        annotations = Counter(
            a["image_id"] for a in coco["annotations"] if a.get("segmentation")
        )
        rejected, selected, seen = {}, {}, set()
        for row in coco["images"]:
            camera_id = PHOTO.search(row["file_name"])
            if camera_id is None or camera_id.group() in seen:
                raise ValueError(f"Unrecognized or duplicate camera file: {row['file_name']}")
            photo = camera_id.group()
            seen.add(photo)
            if photo in excluded:
                rejected[photo] = "outside indoor scope"
            elif annotations[row["id"]] == 0:
                rejected[photo] = "no segmentation"
            else:
                selected[row["id"]] = row
        if excluded - seen:
            raise ValueError(f"Exclusion identifiers missing from export: {excluded - seen}")
        filtered = {
            **coco,
            "images": list(selected.values()),
            "annotations": [a for a in coco["annotations"] if a["image_id"] in selected],
        }
        with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr((prefix / "_annotations.coco.json").as_posix(), json.dumps(filtered))
            for row in selected.values():
                member = (prefix / row["file_name"]).as_posix()
                archive.writestr(member, raw.read(member), compress_type=ZIP_STORED)
    return {"source_sha256": digest, "eligible_images": len(selected), "excluded": rejected}


def evaluate(source, taxonomy, runs, output, excluded, device, threads, reuse=None, prepared=False):
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "photo-test-640.zip"
    with TemporaryDirectory(prefix="indoor-snapshot-") as tmp:
        temporary = Path(tmp)
        filtered = temporary / "selected.zip"
        receipt = output / "selection.json"
        if prepared:
            selection = json.loads(receipt.read_text(encoding="utf-8"))
            if selection["source_sha256"] != sha256(source) or not dataset.is_file():
                raise ValueError("Prepared split does not match this export.")
        else:
            selection = select_indoor_images(source, filtered, excluded)
            prepare(filtered, dataset, 640, taxonomy)
            write_json(receipt, selection)
        for run in runs:
            if reuse:
                update_run(run, dataset, output, temporary, device, threads, reuse)
            else:
                evaluate_run(run, dataset, output, temporary, device, threads)


def evaluate_run(run, dataset, output, temporary, device, threads):
    name = run.parent.name
    checkpoint = temporary / "checkpoint.pt"
    classes = temporary / "classes.json"
    with ZipFile(run) as archive, checkpoint.open("wb") as sink, archive.open("checkpoint.pt") as weights:
        copyfileobj(weights, sink)
        summary = json.loads(archive.read("summary.json"))
    write_json(classes, {"classes": summary["class_names"]})
    runner = (
        "import runpy, torch\n"
        f"torch.set_num_threads({threads})\n"
        "runpy.run_path('train.py', run_name='__main__')"
    )
    subprocess.run(
        [
            sys.executable, "-u", "-c", runner,
            "--model", summary["model"],
            "--seed", str(summary["seed"]),
            "--classes", str(classes),
            "--dataset", str(dataset),
            "--checkpoint", str(checkpoint),
            "--eval-splits", "private",
            "--device", device,
            "--output", str(output / f"{name}.zip"),
        ],
        check=True,
        cwd=Path(__file__).resolve().parent,
    )


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def update_run(run, dataset, output, temporary, device, threads, reuse):
    """Re-score cached predictions only when the prepared input bytes match."""
    import torch
    from train import (
        CONFIDENCE, MAX_DETECTIONS, YOLO_WEIGHTS, category_table, now,
        predict_maskrcnn, predict_yolo, score_masks,
    )

    torch.set_num_threads(threads)
    started = now()
    name = run.parent.name
    checkpoint = temporary / "checkpoint.pt"
    with ZipFile(run) as archive, checkpoint.open("wb") as sink:
        summary = json.loads(archive.read("summary.json"))
        with archive.open("checkpoint.pt") as weights:
            copyfileobj(weights, sink)
    prior_dataset = reuse / "photo-test-640.zip"
    with ZipFile(reuse / f"{name}.zip") as prior_run:
        prior_summary = json.loads(prior_run.read("summary.json"))
        previous = json.loads(prior_run.read("predictions-private.json"))
    for field in ("class_names", "model", "seed", "input_size", "confidence", "max_detections"):
        if prior_summary[field] != summary[field]:
            raise ValueError(f"{name} has different cached settings for {field}.")
    if prior_summary["dataset_sha256"] != sha256(prior_dataset):
        raise ValueError("Cached predictions and prepared snapshot hash differ.")
    with ZipFile(prior_dataset) as prior_zip, ZipFile(dataset) as new_zip:
        prior_gt = json.loads(prior_zip.read("private/gt.json"))
        current = json.loads(new_zip.read("private/gt.json"))
        manifest = json.loads(new_zip.read("manifest.json"))
        if prior_gt["categories"] != current["categories"]:
            raise ValueError("Taxonomy changed since cached inference.")
        old_rows = {row["source_name"]: row for row in prior_gt["images"]}
        old_to_new, additional = {}, []
        images = temporary / "private-extra"
        images.mkdir(exist_ok=True)
        for row in current["images"]:
            old = old_rows.get(row["source_name"])
            current_bytes = new_zip.read(f"private/images/{row['file_name']}")
            if old and current_bytes == prior_zip.read(f"private/images/{old['file_name']}"):
                old_to_new[old["id"]] = row["id"]
            else:
                additional.append(row)
                (images / row["file_name"]).write_bytes(current_bytes)
    predictions = [
        {**prediction, "image_id": old_to_new[prediction["image_id"]]}
        for prediction in previous
        if prediction["image_id"] in old_to_new
    ]
    model = summary["model"]
    classes = category_table(manifest, summary["class_names"])
    if additional:
        args = argparse.Namespace(model=model, device=device)
        predict = predict_yolo if model in YOLO_WEIGHTS else predict_maskrcnn
        predictions.extend(predict(args, images, {**current, "images": additional},
                                   checkpoint, classes, manifest["size"]))
    scores = score_masks(current, predictions, classes)
    updated = {
        **prior_summary,
        "started_at_utc": started, "finished_at_utc": now(), "dataset_sha256": sha256(dataset),
        "confidence": CONFIDENCE, "max_detections": MAX_DETECTIONS, "scores": {"private": scores},
        "unchanged_images": len(old_to_new), "inferred_images": len(additional),
        "prior_evaluation_sha256": sha256(reuse / f"{name}.zip"), "training_artifact_sha256": sha256(run),
    }
    destination = output / f"{name}.zip"
    partial = destination.with_suffix(".part")
    with ZipFile(partial, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(updated, allow_nan=False, indent=2) + "\n")
        archive.writestr("predictions-private.json", json.dumps(predictions, allow_nan=False))
    partial.replace(destination)
    print(f"{name}: AP={scores['ap']:.5f}, {len(old_to_new)} reused, {len(additional)} new.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--exclude", nargs="*", default=["1000865815"])
    parser.add_argument("--taxonomy", type=Path, default=Path("data/panoptic-indoor-v5-640.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "0"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse", type=Path, help="Prior snapshot and evaluations of the same fixed checkpoints.")
    parser.add_argument("--prepared", action="store_true", help="Use a prepared snapshot from the same export.")
    options = parser.parse_args()
    evaluate(
        options.source.resolve(), options.taxonomy.resolve(),
        [run.resolve() for run in options.runs], options.output.resolve(),
        set(options.exclude), options.device, options.threads, options.reuse, options.prepared,
    )
