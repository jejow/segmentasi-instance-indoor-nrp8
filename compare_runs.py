"""Build the NRP-8 comparison table and figure from completed Colab artifacts."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from metrics import load_json

FRIENDLY_NAMES = {"yolov8n": "YOLOv8n-seg", "yolo11n": "YOLO11n-seg", "maskrcnn": "Mask R-CNN R50-FPN"}


def subset_ap(record, split, names, key="ap"):
    details = record["scores"][split]["per_class"]
    values = [details[name][key] for name in names]
    if any(value is None for value in values):
        return None
    return 100 * mean(values)


def read_runs(paths, classes, checksum):
    runs = {}
    for path in paths:
        with ZipFile(path) as archive:
            record = json.loads(archive.read("summary.json"))
        if record["pilot"] or record["dataset_sha256"] != checksum:
            raise ValueError(f"Run used a different dataset or was a pilot: {path}")
        if record["model"] not in FRIENDLY_NAMES:
            raise ValueError(f"Unexpected model in {path}")
        if record["completed_epochs"] is None or record["completed_epochs"] < 1:
            raise ValueError(f"No completed training epoch: {path}")
        trained_classes = set(record["class_names"])
        if trained_classes == set(classes):
            condition = "three"
        elif trained_classes.issuperset(classes) and len(trained_classes) > 3:
            condition = "full"
        else:
            raise ValueError(f"Run does not use the declared class set: {path}")
        run = record["model"], condition
        if run in runs:
            raise ValueError(f"Two artifacts claim the same model/condition: {run}")
        runs[run] = record
    return runs


def compare(run_paths, selection_path, output):
    selection = load_json(selection_path)
    classes = selection["classes"]
    runs = read_runs(run_paths, classes, selection["dataset_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    paired = []
    for name, label in FRIENDLY_NAMES.items():
        pair = {}
        for condition in ("full", "three"):
            record = runs.get((name, condition))
            if record is None:
                continue
            ap = subset_ap(record, "test", classes)
            if ap is not None:
                pair[condition] = ap
            rows.append({
                "model": label, "condition": condition, "classes": len(record["class_names"]),
                "seed": record["seed"], "epoch": record["completed_epochs"],
                "gpu": record["gpu"], "training_minutes": record["training_seconds"] / 60,
                "test_images": record["scores"]["test"]["images"],
                "test_ap_all_supported": 100 * record["scores"]["test"]["ap"] if condition == "full" else None,
                "test_ap_target": ap, "test_ap50_target": subset_ap(record, "test", classes, "ap50"),
                "test_ap75_target": subset_ap(record, "test", classes, "ap75"),
                **{f"test_ap_{class_name}": subset_ap(record, "test", [class_name]) for class_name in classes},
            })
        if set(pair) == {"full", "three"}:
            first, second = runs[name, "full"], runs[name, "three"]
            if (first["seed"], first["input_size"]) != (second["seed"], second["input_size"]):
                raise ValueError(f"Train conditions do not share seed and input size for {label}.")
            paired.append((label, pair["full"], pair["three"]))
    if not rows:
        raise ValueError("No completed experiment artifacts.")
    with (output / "hasil-test-v5.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not paired:
        return
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    fig, ax = plt.subplots(figsize=(9.2, 4.9), layout="constrained")
    positions = np.arange(len(paired))
    specs = [(1, -.18, "Semua kelas", "#21578b"), (2, .18, "Tiga kelas", "#008875")]
    peak = max(max(row[1:]) for row in paired)
    for column, offset, label, color in specs:
        bars = ax.bar(positions + offset, [row[column] for row in paired], .32, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_title("Efek jumlah kelas pada test set asli Roboflow v5", loc="left", weight="bold")
    ax.set_ylabel("Mask AP50–95 pada kelas target (%)")
    ax.set_xlabel(", ".join(classes) + " · 359 gambar test", labelpad=10)
    ax.set_xticks(positions, [row[0] for row in paired])
    ax.set_ylim(0, min(110, max(.1, peak * 1.44)))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(title="Kondisi training", frameon=False)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "perbandingan-nrp8-test-v5.png", dpi=200)
    fig.savefig(output / "perbandingan-nrp8-test-v5.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--classes", type=Path, default=Path("hard-classes.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    compare(args.runs, args.classes, args.output)
