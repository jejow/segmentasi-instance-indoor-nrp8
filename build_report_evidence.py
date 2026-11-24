"""Collect traceable report metrics, original epoch logs, and figures."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import file_digest
import json
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile


BASE_NAMES = {"yolov8n": "YOLOv8n-seg", "yolo11n": "YOLO11n-seg", "maskrcnn": "Mask R-CNN"}
CASES = (372, 548, 183, 106, 200)
TARGETS = ("knife", "light", "spoon")
METRICS = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")
OUTCOMES = ("below_iou", "other_class_only", "correct_class")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def mask_diagnosis(test, evaluations):
    """Count ground truths by best candidate overlap, before COCO matching."""
    import numpy as np
    from pycocotools import mask

    with ZipFile(test) as archive:
        gt = json.loads(archive.read("test/gt.json"))
    truth = defaultdict(list)
    for row in gt["annotations"]:
        if not row["iscrowd"]:
            truth[row["image_id"]].append(row)
    fingerprint = digest(test)
    result = []
    for condition, artifact in evaluations.items():
        estimates = defaultdict(list)
        with ZipFile(artifact) as archive:
            summary = json.loads(archive.read("summary.json"))
            if (summary["dataset_sha256"], summary["confidence"], summary["max_detections"]) != (fingerprint, .005, 100):
                raise ValueError(f"Evaluation settings do not match for {condition}.")
            for row in json.loads(archive.read("predictions-test.json")):
                estimates[row["image_id"]].append(row)
        counts = Counter()
        for image_id, labels in truth.items():
            predictions = estimates[image_id]
            if predictions:
                overlaps = mask.iou(
                    [p["segmentation"] for p in predictions],
                    [a["segmentation"] for a in labels],
                    [0] * len(labels),
                ) >= .50
                classes = np.array([p["category_id"] for p in predictions])
            for i, reference in enumerate(labels):
                outcome = "below_iou"
                if predictions and overlaps[:, i].any():
                    same = classes == reference["category_id"]
                    outcome = "correct_class" if (same & overlaps[:, i]).any() else "other_class_only"
                area = reference["area"]
                size = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
                counts["all", outcome] += 1
                counts[size, outcome] += 1
        for size in ("all", "small", "medium", "large"):
            row = {outcome: counts[size, outcome] for outcome in OUTCOMES}
            result.append({"condition": condition, "size": size, "masks": sum(row.values()), **row})
    return result


def original_logs(root, target, manifest, replays, improvement):
    groups = [
        ("inti", row["run"], root / "runs" / row["run"], row["epochs"], row["training_profile"])
        for row in manifest
    ]
    groups += [
        ("replikasi", row["run"], root / "research/log-runs" / row["run"], row["epochs"], row["profile"])
        for row in replays["runs"]
    ]
    groups += [
        ("perbaikan", row["run"], root / "improvement/log-runs" / row["run"], row["epochs"], "")
        for row in improvement["runs"] if row["head"] != "random" or row["rcnn_size"] != 640
    ]
    rows = []
    for group, run, directory, epochs, profile in groups:
        summary = read(directory / "summary.json")
        source = directory / "results.csv"
        with source.open(encoding="utf-8-sig", newline="") as stream:
            entries = list(csv.DictReader(stream))
        if len(entries) != epochs or summary["completed_epochs"] != epochs:
            raise ValueError(f"Incomplete epoch log for {run}.")
        destination = target / group / run / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)
        rows.append({
            "experiment": group, "run": run, "model": summary["model"], "seed": summary["seed"],
            "profile": profile, "gpu": summary["gpu"], "epochs": epochs,
            "training_minutes": summary["training_seconds"] / 60,
            "started_at_utc": summary["started_at_utc"],
            "training_clock": summary.get("training_clock", "training_seconds"),
            "epoch_log_sha256": digest(destination),
        })
    return rows


def collect(project, destination):
    output = project / "output"
    private = output / "test-pribadi"
    selected = output / "improvement"
    public = output / "perbandingan-v5" / "hasil-test-v5.csv"
    artifact = selected / read(selected / "selection.json")["selection"]["checkpoint"]
    comparison = read(private / "eval-v3/comparison.json")
    phone_data = private / "eval-v3/photo-test-640.zip"
    gamma = read(private / "gamma-065-kasus-v3.json")
    if digest(phone_data) != comparison["dataset_sha256"] or gamma["dataset_sha256"] != comparison["dataset_sha256"]:
        raise ValueError("The phone test and the lighting check must use the same annotated snapshot.")
    with public.open(encoding="utf-8-sig", newline="") as stream:
        original_results = {(r["model"], r["condition"]): r for r in csv.DictReader(stream)}
    main = []
    domain = []
    for run in comparison["models"]:
        model, classes = run["model"], run["trained_classes"]
        if model not in BASE_NAMES:
            continue
        condition = "three" if classes == 3 else "full"
        public_row = next(r for (name, mode), r in original_results.items()
                          if name.startswith(BASE_NAMES[model]) and mode == condition)
        if abs(float(public_row["test_ap_target"]) - run["public_target_ap"] * 100) > 1e-9:
            raise ValueError(f"Conflicting public scores for {model} {condition}.")
        main.append({
            "model": model, "condition": condition, "classes": classes,
            "public_images": run["public_images"], "phone_images": run["private_images"],
            "public_native_pct": 100 * run["public_native_ap"] if condition == "full" else "",
            "public_targets_pct": 100 * run["public_target_ap"],
            "phone_targets_pct": 100 * run["private_target_ap"],
            "training_minutes": float(public_row["training_minutes"]),
        })
        if condition == "full":
            domain.append({"model": BASE_NAMES[model], "classes": run["common_classes"],
                           "public_pct": 100 * run["public_common_ap"], "phone_pct": 100 * run["private_common_ap"]})
    if len(main) != 6 or len(original_results) != 6:
        raise ValueError("The three full/three-class model pairs are required.")
    improvement = read(selected / "selection.json")
    replays = read(output / "research/research.json")
    phone = read(selected / "phone-test.json")
    public_test = read(selected / "public-test.json")
    if phone["dataset_sha256"] != comparison["dataset_sha256"]:
        raise ValueError("The extra test uses a different phone snapshot.")
    for label, key in (("Mask R-CNN v2", "control"), ("Mask R-CNN v2, COCO 800", "selected")):
        domain.append({"model": label, "classes": len(phone["classes"]),
                       "public_pct": 100 * phone[key]["public_common"]["ap"], "phone_pct": 100 * phone[key]["private_common"]["ap"]})
    validation = [
        {"run": row["run"], "condition": ("coco" if row["head"] != "random" else "random") + f"-{row['rcnn_size']}",
         "seed": row["seed"], "ap_pct": 100 * row["ap"], "small_pct": 100 * row["ap_small"],
         "targets_pct": 100 * row["ap_target"]}
        for row in improvement["runs"]
    ]
    lighting = [{"class": label, "instances": gamma["instances"] if label == "Seluruh kelas"
                 else gamma["control"]["per_class"][label]["instances"],
                 "control_pct": 100 * (gamma["control"]["ap"] if label == "Seluruh kelas"
                                       else gamma["control"]["per_class"][label]["ap"]),
                 "gamma_pct": 100 * (gamma["brightened"]["ap"] if label == "Seluruh kelas"
                                     else gamma["brightened"]["per_class"][label]["ap"])}
                for label in ("Seluruh kelas", "person", "light")]
    cases = {row["annotation_id"]: row for row in read(output / "report/cases/candidates.json")}
    case_rows = []
    for number, key in enumerate(CASES, 1):
        entry = cases[key]
        for model, row in entry["models"].items():
            case_rows.append({"case": number, "reference_label": entry["class"],
                              "reference_area_px": entry["area"], "model": model,
                              "same_class_iou": row["same_class_iou"], "best_label": row["nearest_class"],
                              "best_iou": row["nearest_iou"], "shown_label": row["shown_class"],
                              "shown_iou": row["shown_iou"]})
    evaluation_paths = {
        "random-640": output / "research/maskrcnn-v2-97/test-public-colab.zip",
        "coco-800": artifact / "public-test-colab.zip",
    }
    diagnosis = mask_diagnosis(output / "tuning/test-v5.zip", evaluation_paths)
    if any(row["masks"] != 5344 for row in diagnosis if row["size"] == "all"):
        raise ValueError("Unexpected public test denominator.")
    prepared_public = read(project / "data/panoptic-indoor-v5-640.json")["splits"]
    prepared_phone = read(private / "eval-v3/photo-test-640.json")["splits"]
    split_sizes = {
        key: {"images": split["images"], "masks": split["instances"], "classes": len(split["class_instances"])}
        for key, split in {**prepared_public, **prepared_phone}.items()
    }
    release = {
        "dataset_splits": split_sizes,
        "nrp8": main,
        "domain": domain,
        "replay": [{"model": r["model"], "seed": r["seed"], "ap_pct": 100 * r["ap"],
                    "minutes": r["minutes"]} for r in replays["runs"]],
        "validation": validation,
        "public_test": {condition: {m: 100 * public_test[key]["scores"][m]
                                     for m in (*METRICS, "ap_target")}
                        for condition, key in (("random-640", "control"), ("coco-800", "improved_configuration"))},
        "phone_test": {
            condition: {
                "native": {m: 100 * phone[key]["private_native"][m] for m in METRICS},
                "common": {m: 100 * phone[key]["private_common"][m] for m in METRICS[:3]},
            }
            for condition, key in (("random-640", "control"), ("coco-800", "selected"))
        },
        "gamma": lighting,
        "gamma_examples": [
            {"case": {216: 3, 192: 5}[row["annotation_id"]], "class": row["class"],
             **{mode: {key: row[mode][key] for key in ("same_class_iou", "nearest_iou", "nearest_class")}
                for mode in ("control", "brightened")}}
            for row in gamma["examples"]
        ],
        "cases": case_rows,
        "diagnosis": diagnosis,
        "source_sha256": {
            "dataset_public_640": improvement["dataset_sha256"], "dataset_test_public": public_test["test_dataset_sha256"],
            "dataset_phones_v3": comparison["dataset_sha256"],
            "analysis_gamma_v3": digest(private / "gamma-065-kasus-v3.json"),
            "public_predictions": {k: digest(path) for k, path in evaluation_paths.items()},
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("nrp8-dua-test.csv", main), ("domain-shift.csv", domain),
                           ("diagnostik-mask.csv", diagnosis), ("kasus-iou.csv", case_rows)):
        write_csv(destination / filename, rows)
    copyfile(private / "eval-v3/per-class.csv", destination / "ap-per-kelas.csv")
    logs = original_logs(output, destination.parent / "log-training",
                         read(output / "manifest-hasil.json"), replays, improvement)
    write_csv(destination / "waktu-gpu.csv", logs)
    (destination / "ringkasan.json").write_text(
        json.dumps(release, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return release


def number(value):
    return f"{value:.2f}".replace(".", ",")


def render_charts(data, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": "#171717",
        "axes.labelcolor": "#171717", "axes.edgecolor": "#424242",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "grid.color": "#dedede", "grid.linewidth": .65,
    })

    def save(fig, name):
        fig.savefig(destination / name, dpi=220, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.48), layout="constrained", sharey=True)
    for ax, field, title in zip(axes, ("public_targets_pct", "phone_targets_pct"),
                                ("Test publik · 359 foto", "Foto HP · 40 foto"), strict=True):
        for shift, condition, style in (
            (-.18, "full", {"color": "#315E86", "label": "Semua kelas"}),
            (.18, "three", {"color": "#A5D6CB", "edgecolor": "#236657", "hatch": "///", "label": "Tiga kelas"}),
        ):
            rows = [r for r in data["nrp8"] if r["condition"] == condition]
            bars = ax.bar([x + shift for x in range(3)], [r[field] for r in rows], .33, **style)
            ax.bar_label(bars, labels=[number(r[field]) for r in rows], padding=3, fontsize=8)
        ax.set_xticks(range(3), ["YOLOv8n", "YOLO11n", "Mask R-CNN"])
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 1.38)
        ax.grid(axis="y")
    axes[0].set_ylabel("Mask AP50-95 tiga target (%)")
    axes[1].legend(frameon=False, loc="upper left")
    save(fig, "nrp8-dua-test.png")

    fig, ax = plt.subplots(figsize=(7.25, 2.42), layout="constrained")
    for pos, row in enumerate(data["domain"]):
        public, phone = row["public_pct"], row["phone_pct"]
        ax.plot((public, phone), (pos, pos), color="#888888", linewidth=1.25, zorder=2)
        ax.scatter(public, pos, marker="o", color="#315E86", s=35, zorder=3,
                   label="Publik" if pos == 0 else None)
        ax.scatter(phone, pos, marker="s", facecolors="#A5D6CB", edgecolors="#236657", s=36, zorder=3,
                   label="Foto HP" if pos == 0 else None)
    ax.set_yticks(range(len(data["domain"])), [r["model"] for r in data["domain"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 9.05)
    ax.set_xlabel("Mask AP50-95 pada 37 kelas bersama (%)")
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="upper right", ncols=2)
    save(fig, "domain-shift.png")

    fig, ax = plt.subplots(figsize=(7.25, 1.7), layout="constrained")
    rows = [r for r in data["diagnosis"] if r["size"] == "all"]
    for pos, row in enumerate(rows):
        left = 0
        for key, label, shade, ink in zip(
            OUTCOMES, ("IoU di bawah 0,50", "Kelas lain saja", "Sekelas"),
            ("#E7EDF2", "#E7BE84", "#315E86"), ("#161616", "#171717", "#ffffff"), strict=True
        ):
            share = 100 * row[key] / row["masks"]
            ax.barh(pos, share, left=left, height=.46, color=shade, edgecolor="#454545", linewidth=.45,
                    label=label if pos == 0 else None)
            ax.text(left + share / 2, pos, f"{share:.1f}%".replace(".", ","), ha="center", va="center", color=ink)
            left += share
    ax.set_yticks([0, 1], ["Acak 640", "COCO 800"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Bagian dari 5.344 mask acuan (%)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncols=3, borderaxespad=0)
    save(fig, "diagnostik-mask.png")

    ordering = ("random-640", "coco-640", "random-800", "coco-800")
    labels = ("Acak\n640", "COCO\n640", "Acak\n800", "COCO\n800")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), layout="constrained")
    for ax, field, title in zip(axes, ("ap_pct", "small_pct"), ("Seluruh kelas", "Objek kecil"), strict=True):
        for i, condition in enumerate(ordering):
            points = sorted((r for r in data["validation"] if r["condition"] == condition),
                            key=lambda r: r["seed"])
            for row, dx, marker, fill in zip(points, (-.1, .1), ("o", "s"), ("#315E86", "#A5D6CB"), strict=True):
                ax.scatter(i+dx, row[field], marker=marker, edgecolors="#171717", facecolors=fill, s=32,
                           label=f"Seed {row['seed']}" if i == 0 else None, zorder=3)
            midpoint = sum(r[field] for r in points) / len(points)
            ax.hlines(midpoint, i-.17, i+.17, color="#636363", label="Rata-rata" if i == 0 else None)
        ax.set_xticks(range(4), labels)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Mask AP validasi (%)")
        ax.margins(y=.16)
        ax.grid(axis="y")
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, frameon=False, loc="outside lower center", ncols=3, fontsize=8)
    save(fig, "validasi-maskrcnn.png")

    fig, ax = plt.subplots(figsize=(7.25, 1.86), layout="constrained")
    for pos, row in enumerate(data["gamma"]):
        before, after = row["control_pct"], row["gamma_pct"]
        ax.plot((before, after), (pos, pos), color="#777777", linewidth=1.2, zorder=2)
        ax.scatter(before, pos, marker="o", facecolors="white", edgecolors="#315E86", s=35, zorder=3,
                   label="Foto asal" if pos == 0 else None)
        ax.scatter(after, pos, marker="s", color="#236657", s=31, zorder=3,
                   label="Gamma 0,65" if pos == 0 else None)
    ax.set_yticks(range(3), [f"{r['class']} · {r['instances']} mask" for r in data["gamma"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 23.5)
    ax.set_xlabel("Mask AP50-95 (%)")
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="upper right", ncols=2)
    save(fig, "uji-gamma.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true", help="Rebuild figures from the checked-in aggregate data.")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    destination = project / "bukti/hasil"
    results = read(destination / "ringkasan.json") if args.figures_only else collect(project, destination)
    render_charts(results, destination)
    print(f"Verified report evidence: {destination}")
