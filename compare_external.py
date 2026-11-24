"""Compare stored public and private scores on the exact same class support."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from audit_private_test import PHOTO
from metrics import score_masks, write_json
from prepare_dataset import hash_file


METRICS = ("ap", "ap50", "ap75")
TARGETS = ("knife", "light", "spoon")


def summary(path):
    with ZipFile(path) as archive:
        return json.loads(archive.read("summary.json"))


def supported(scores):
    return {name for name, result in scores["per_class"].items() if result["ap"] is not None}


def subset(scores, names):
    if not names or not set(names) <= supported(scores):
        return dict.fromkeys(METRICS)
    return {key: mean(scores["per_class"][name][key] for name in sorted(names)) for key in METRICS}


def write_table(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, fieldnames=list(records[0]))
        table.writeheader()
        table.writerows(records)


def support_details(gt):
    category_names = {c["id"]: c["name"] for c in gt["categories"]}
    photos, masks = defaultdict(set), Counter()
    for annotation in gt["annotations"]:
        if not annotation["iscrowd"]:
            name = category_names[annotation["category_id"]]
            photos[name].add(annotation["image_id"])
            masks[name] += 1
    return {name: {"masks": masks[name], "photos": len(photos[name])} for name in sorted(masks)}


def difficult_scores(gt, predictions, challenging, categories):
    photo_ids = {row["id"]: PHOTO.search(row["source_name"]).group() for row in gt["images"]}
    by_difficulty = {}
    shared = None
    for name, is_challenging in (("difficult", True), ("other", False)):
        image_ids = {k for k, v in photo_ids.items() if (v in challenging) == is_challenging}
        if not image_ids:
            return None
        restricted_gt = {
            **gt,
            "images": [row for row in gt["images"] if row["id"] in image_ids],
            "annotations": [row for row in gt["annotations"] if row["image_id"] in image_ids],
        }
        restricted_predictions = [row for row in predictions if row["image_id"] in image_ids]
        scores = score_masks(restricted_gt, restricted_predictions, categories)
        shared = supported(scores) if shared is None else shared & supported(scores)
        by_difficulty[name] = scores
    return {
        "classes": sorted(shared),
        **{key: {"photos": result["images"], **subset(result, shared)} for key, result in by_difficulty.items()},
    }


def plot_comparison(rows, destination):
    import matplotlib.pyplot as plt
    import numpy as np

    full = [row for row in rows if row["trained_classes"] > 3 and row["common_classes"]]
    names = {
        "yolov8n": "YOLOv8n-seg", "yolo11n": "YOLO11n-seg",
        "maskrcnn": "Mask R-CNN", "maskrcnn-v2": "Mask R-CNN v2",
    }
    positions = np.arange(len(full))
    fig, axis = plt.subplots(figsize=(8.3, 3.65), layout="constrained")
    for field, label, shift, color in (
        ("public_common_ap", "Test publik", -.18, "#215988"),
        ("private_common_ap", "Foto ponsel", .18, "#128577"),
    ):
        heights = [100 * row[field] for row in full]
        bars = axis.bar(positions + shift, heights, .32, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_xticks(positions, [names[row["model"]] for row in full])
    axis.set_ylabel("Mask AP50–95 (%)")
    axis.set_ylim(bottom=0, top=1.28 * max(100 * row[field] for row in full
                                         for field in ("public_common_ap", "private_common_ap")))
    axis.legend(loc="upper left", frameon=False)
    axis.grid(axis="y", alpha=.22)
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_axisbelow(True)
    fig.savefig(destination, dpi=195)
    plt.close(fig)


def compare(public_paths, private_directory, difficult_file):
    prepared_dataset = private_directory / "photo-test-640.zip"
    fingerprint = hash_file(prepared_dataset)
    with ZipFile(prepared_dataset) as dataset:
        gt = json.loads(dataset.read("private/gt.json"))
    difficult_ids = set(json.loads(difficult_file.read_text(encoding="utf-8"))) if difficult_file else None
    rows, detail, comparisons = [], [], {}
    fingerprints = set()
    for public_path in public_paths:
        key = public_path.parent.name
        public = summary(public_path)
        private_path = private_directory / f"{key}.zip"
        with ZipFile(private_path) as archive:
            private = json.loads(archive.read("summary.json"))
            predictions = json.loads(archive.read("predictions-private.json")) if difficult_ids else None
        fingerprints.add((private["dataset_sha256"], private["scores"]["private"]["images"]))
        for field in ("model", "class_names", "seed", "input_size", "confidence", "max_detections"):
            if public[field] != private[field]:
                raise ValueError(f"{key} uses incompatible {field}.")
        if public["model"].startswith("maskrcnn"):
            public_size = public.get("rcnn_size") or public["input_size"]
            private_size = private.get("rcnn_size") or private["input_size"]
            if public_size != private_size:
                raise ValueError(f"{key} uses incompatible internal input sizes.")
        native, external = public["scores"]["test"], private["scores"]["private"]
        common = supported(native) & supported(external)
        native_ap, external_ap = subset(native, common), subset(external, common)
        rows.append({
            "run": key, "model": public["model"], "trained_classes": len(public["class_names"]),
            "public_images": native["images"], "private_images": external["images"],
            "common_classes": len(common),
            "public_supported_classes": len(supported(native)),
            "private_supported_classes": len(supported(external)),
            "public_native_ap": native["ap"], "private_native_ap": external["ap"],
            **{f"public_common_{k}": v for k, v in native_ap.items()},
            **{f"private_common_{k}": v for k, v in external_ap.items()},
            **{f"public_target_{k}": v for k, v in subset(native, TARGETS).items()},
            **{f"private_target_{k}": v for k, v in subset(external, TARGETS).items()},
        })
        for name in public["class_names"]:
            detail.append({
                "run": key, "class": name, "common": name in common,
                **{f"public_{k}": v for k, v in native["per_class"][name].items()},
                **{f"private_{k}": v for k, v in external["per_class"][name].items()},
            })
        comparisons[key] = {"common_classes": sorted(common)}
        if difficult_ids:
            comparisons[key]["difficulty"] = difficult_scores(
                gt, predictions, difficult_ids, {int(k) for k in private["categories"]}
            )
    if fingerprints != {(fingerprint, len(gt["images"]))}:
        raise ValueError("Mixed test snapshots in comparison.")
    report = {
        "dataset_sha256": fingerprint,
        "class_support": support_details(gt),
        "models": rows,
        "comparisons": comparisons,
    }
    write_json(private_directory / "comparison.json", report)
    write_table(private_directory / "comparison.csv", rows)
    write_table(private_directory / "per-class.csv", detail)
    if any(row["trained_classes"] > 3 and row["common_classes"] for row in rows):
        plot_comparison(rows, private_directory / "perbandingan-domain.png")
    for row in rows:
        pub, own = row["public_common_ap"], row["private_common_ap"]
        if pub is not None and own is not None:
            print(f"{row['model']} ({row['trained_classes']}): {100 * pub:.2f}% / {100 * own:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_directory", type=Path)
    parser.add_argument("public_artifacts", type=Path, nargs="+")
    parser.add_argument("--difficult", type=Path)
    options = parser.parse_args()
    compare(options.public_artifacts, options.private_directory, options.difficult)
