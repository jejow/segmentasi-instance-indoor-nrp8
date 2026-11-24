"""Collect verified Colab ablations, preserve logs, and rank by validation AP."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile
import shutil

EXPECTED = {
    "full-uniform": "Semua kelas, kontrol",
    "full-repeat": "Semua kelas, repeat factor",
    "full-repeat-mask2": "Repeat factor, mask ratio 2",
    "full-repeat-mask2-light": "Repeat factor, mask ratio 2, HSV",
    "hard-uniform": "Tiga kelas, kontrol",
    "hard-foreground": "Tiga kelas, oversampling positif",
}
CLASSES = ("knife", "light", "spoon")


def collect(registry, output):
    output.mkdir(exist_ok=True, parents=True)
    rows, incomplete = [], {}
    checksum = None
    for run, label in EXPECTED.items():
        candidates = sorted(
            [p for p in registry.iterdir() if p.name == run or p.name.startswith(run + "-retry")],
            key=lambda folder: folder.name,
        )
        latest = None
        for directory in candidates:
            state = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
            latest = state["phase"]
            if latest == "complete":
                break
        else:
            incomplete[run] = latest or "unregistered"
            continue
        artifact = directory / "artifacts/output/result.zip"
        with ZipFile(artifact) as bundle:
            record = json.loads(bundle.read("summary.json"))
            epochs = list(csv.DictReader(io.StringIO(bundle.read("results.csv").decode())))
            if bundle.testzip() or len(epochs) != 32 or record["completed_epochs"] != 32:
                raise ValueError(f"Archive is invalid or does not have 32 epochs: {directory.name}")
            if checksum is None:
                checksum = record["dataset_sha256"]
            if (
                record["dataset_sha256"] != checksum or record["epoch_images"] != 640
                or record["seed"] != 42 or set(record["scores"]) != {"valid"}
            ):
                raise ValueError(f"Incomparable protocol in {directory.name}")
            scores = record["scores"]["valid"]
            if scores["images"] != 166:
                raise ValueError("Unexpected validation split.")
            dest = output / "runs" / directory.name
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("summary.json", "results.csv"):
                bundle.extract(name, dest)
        copied = shutil.copy2(artifact, dest / "result.zip")
        with artifact.open("rb") as original, Path(copied).open("rb") as clone:
            sha = hashlib.file_digest(original, "sha256").hexdigest()
            if sha != hashlib.file_digest(clone, "sha256").hexdigest():
                raise ValueError("Local archive copy hash differs.")
        for name in ("training.log", "monitor.log"):
            if (directory / name).is_file():
                shutil.copy2(directory / name, dest / name)
        rows.append({
            "run": run, "label": label, "profile": state["colab_profile"],
            "output_run": directory.name, "epochs": len(epochs),
            "scope": "three" if len(record["class_names"]) == 3 else "full",
            "ap": scores["ap"] * 100, "ap50": scores["ap50"] * 100, "ap75": scores["ap75"] * 100,
            "target_ap": mean(scores["per_class"][name]["ap"] for name in CLASSES) * 100,
            "training_minutes": record["training_seconds"] / 60,
            "sha256": sha,
        })
    if rows:
        with (output / "hasil-validasi.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    winners = {}
    for scope in ("full", "three"):
        candidates = [row for row in rows if row["scope"] == scope]
        if candidates:
            winners[scope] = max(candidates, key=lambda row: row["ap"])
    report = {"completed": rows, "incomplete": incomplete, "best_by_scope": winners}
    (output / "ringkasan-validasi.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['run']}: AP {row['ap']:.3f}%, target AP {row['target_ap']:.3f}%, "
              f"{row['training_minutes']:.2f} min")
    print("Incomplete:", incomplete)
    for scope, row in winners.items():
        print(f"Best {scope}: {row['output_run']}, {row['ap']:.3f}%")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/tuning"))
    args = parser.parse_args()
    collect(args.registry, args.output)
