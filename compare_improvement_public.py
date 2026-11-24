"""Rescore the control and validation-selected model on the exact same public test set."""
import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from collect_improvement import METRICS, TARGETS, sha256
from metrics import score_masks, write_json


def rescore(artifact, dataset_hash, gt, classes, size):
    with ZipFile(artifact) as archive:
        summary = json.loads(archive.read("summary.json"))
        predictions = json.loads(archive.read("predictions-test.json"))
    expected = {
        "model": "maskrcnn-v2", "seed": 97, "dataset_sha256": dataset_hash,
        "confidence": 0.005, "max_detections": 100, "input_size": 640, "class_names": classes,
    }
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError(f"Different model, dataset, or evaluation parameters in {artifact}.")
    if (summary.get("rcnn_size") or 640) != size or set(summary["scores"]) != {"test"}:
        raise ValueError(f"Different internal input size or evaluation split in {artifact}.")
    if summary["completed_epochs"] is not None:
        raise ValueError(f"The public test run trained a model in {artifact}.")
    scores = score_masks(gt, predictions, [c["id"] for c in gt["categories"]])
    recorded = summary["scores"]["test"]
    if any(not isclose(recorded[key], scores[key], abs_tol=1e-9) for key in ("ap", "ap50", "ap75")):
        raise ValueError(f"Could not reproduce the recorded test score in {artifact}.")
    scores["ap_target"] = mean(scores["per_class"][label]["ap"] for label in TARGETS)
    return {
        "rcnn_size": size, "scores": scores, "artifact": str(artifact), "artifact_sha256": sha256(artifact),
    }


def compare(dataset, control, evaluated, selection_file, output):
    select = json.loads(selection_file.read_text(encoding="utf-8"))
    choice = select["selection"]
    if not choice["ready_for_public_test"]:
        raise ValueError("The validation run has not qualified a new checkpoint for public testing.")
    with ZipFile(dataset) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        gt = json.loads(archive.read("test/gt.json"))
    if manifest["parent_archive_sha256"] != select["dataset_sha256"] or set(manifest["splits"]) != {"test"}:
        raise ValueError("The test bundle does not refer to the selected model's original split archive.")
    if len(gt["images"]) != 359 or len(gt["annotations"]) != 5_344:
        raise ValueError("The test bundle has a different image or mask count.")
    classes = manifest["categories"]
    dataset_hash = sha256(dataset)
    before = rescore(control, dataset_hash, gt, classes, 640)
    after = rescore(evaluated, dataset_hash, gt, classes, choice["rcnn_size"])
    paired = {key: after["scores"][key] - before["scores"][key] for key in METRICS}
    result = {
        "control": before, "improved_configuration": after,
        "condition": choice["condition"], "checkpoint": choice["checkpoint"],
        "test_dataset_sha256": dataset_hash, "difference": paired,
        "active_classes": sum(row["ap"] is not None for row in before["scores"]["per_class"].values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    for key, change in paired.items():
        print(f"{key}: {before['scores'][key]:.2%} -> {after['scores'][key]:.2%} ({100*change:+.2f} pp)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluated", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("output/tuning/test-v5.zip"))
    parser.add_argument("--control", type=Path, default=Path("output/research/maskrcnn-v2-97/test-public-colab.zip"))
    parser.add_argument("--selection", type=Path, default=Path("output/improvement/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("output/improvement/public-test.json"))
    args = parser.parse_args()
    compare(args.dataset, args.control, args.evaluated, args.selection, args.output)
