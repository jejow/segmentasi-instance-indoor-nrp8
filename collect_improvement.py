"""Check paired validation artifacts and select one seed-97 public test checkpoint."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import file_digest
import json
from math import isclose
from pathlib import Path
from shutil import copyfileobj
from statistics import mean
from struct import error as ZipHeaderError
from zipfile import ZipFile
import zlib

from metrics import score_masks, write_json
from summarize_research import read_partial_metrics


SEEDS = (42, 97)
CONFIGS = ("random-640", "coco-640", "random-800", "coco-800")
METRICS = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ap_target")
TARGETS = ("knife", "light", "spoon")


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def unpack_progress(directory):
    with (directory / "progress.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def candidate_artifact(registry, key):
    directories = sorted(
        (d for d in registry.glob(f"{key}*") if d.name == key or d.name.startswith(f"{key}-retry")),
        key=lambda d: (d.name != key, d.name),
    )
    ready = []
    for directory in directories:
        if not (directory / "progress.json").is_file():
            continue
        progress = unpack_progress(directory)
        outputs = list((directory / "artifacts").rglob("result.zip"))
        if progress["phase"] in {"complete", "interrupted"} and len(outputs) == 1:
            ready.append((directory, outputs[0], False))
        elif progress["phase"] == "interrupted" and progress.get("epoch") == 20:
            parts = sorted((directory / "artifacts").rglob("result.zip.*.part"))
            if len(parts) == 1:
                try:
                    read_partial_metrics(parts[0])
                except (OSError, ValueError, ZipHeaderError, zlib.error):
                    continue
                ready.append((directory, parts[0], True))
    return min(ready, key=lambda item: item[2]) if ready else None


def make_row(artifact, run, seed, head, size, public_hash, training_hash, record,
             scores, class_names, timeline, checkpoint=True):
    expected = ("maskrcnn-v2", seed, 20, 20, 640, "Tesla T4", False)
    keys = ("model", "seed", "target_epochs", "completed_epochs", "input_size", "gpu", "pilot")
    observed_epochs = [(int(row["epoch"]), int(row["batches"])) for row in csv.DictReader(timeline.splitlines())]
    if tuple(record[k] for k in keys) != expected or observed_epochs != [(e, 199) for e in range(1, 21)]:
        raise ValueError(f"Model, seed, epoch, or evaluation grid differs in {artifact}.")
    if record["dataset_sha256"] != training_hash or record.get("parent_dataset_sha256", record["dataset_sha256"]) != public_hash:
        raise ValueError(f"Dataset receipt differs in {artifact}.")
    if (record.get("rcnn_size") or 640, record.get("rcnn_head_init") or "random") != (size, head):
        raise ValueError(f"Model configuration differs in {artifact}.")
    if scores["images"] != 166 or record["class_names"] != class_names or set(scores["per_class"]) != set(class_names):
        raise ValueError(f"Validation split differs in {artifact}.")
    return {
        "run": run, "model": record["model"], "seed": seed, "head": head, "rcnn_size": size,
        "epochs": record["completed_epochs"], "gpu": record["gpu"],
        "started_at_utc": record["started_at_utc"], "finished_at_utc": record["finished_at_utc"],
        "minutes": record["training_seconds"] / 60,
        **{key: scores[key] for key in METRICS if key != "ap_target"},
        "ap_target": mean(scores["per_class"][label]["ap"] for label in TARGETS),
        "artifact": str(artifact), "checkpoint_available": checkpoint,
        "artifact_sha256": sha256(artifact),
    }


def collect(registry, original, train_valid, prior_file, output):
    public_hash, training_hash = sha256(original), sha256(train_valid)
    with ZipFile(train_valid) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        gt = json.loads(archive.read("valid/gt.json"))
    if manifest["parent_archive_sha256"] != public_hash or set(manifest["splits"]) != {"train", "valid"}:
        raise ValueError("Training bundle does not correspond to the original split archives.")
    class_names = manifest["categories"]
    category_ids = [category["id"] for category in gt["categories"]]
    if [c["name"] for c in gt["categories"]] != class_names or len(class_names) != 74:
        raise ValueError("The category order differs from the official training manifest.")
    control = {
        row["seed"]: Path(row["artifact"])
        for row in json.loads(prior_file.read_text(encoding="utf-8"))["runs"]
        if row["model"] == "maskrcnn-v2" and row["seed"] in SEEDS
    }
    rows, missing = [], []
    for config in CONFIGS:
        option, size_text = config.split("-")
        size = int(size_text)
        head = "coco-overlap" if option == "coco" else "random"
        for seed in SEEDS:
            key = f"{config}-{seed}"
            if config == "random-640":
                path = control[seed]
                with ZipFile(path) as archive:
                    record = json.loads(archive.read("summary.json"))
                    timeline = archive.read("results.csv").decode("utf-8")
                    predictions = json.loads(archive.read("predictions-valid.json"))
                scores = score_masks(gt, predictions, category_ids)
                if not isclose(scores["ap"], record["scores"]["valid"]["ap"], abs_tol=1e-9):
                    raise ValueError("Archived control no longer reproduces the recorded score.")
                rows.append(make_row(path, key, seed, head, size, public_hash, public_hash,
                                     record, scores, class_names, timeline))
                continue
            received = candidate_artifact(registry, key)
            if not received:
                missing.append(key)
                continue
            directory, path, recovered = received
            if recovered:
                entries = read_partial_metrics(path)
                record = json.loads(entries["summary.json"])
                timeline = entries["results.csv"].decode("utf-8")
                scores = record["scores"]["valid"]
            else:
                with ZipFile(path) as archive:
                    record = json.loads(archive.read("summary.json"))
                    timeline = archive.read("results.csv").decode("utf-8")
                    predictions = json.loads(archive.read("predictions-valid.json"))
                    if archive.getinfo("checkpoint.pt").file_size < 1_000_000:
                        raise ValueError(f"{directory.name} has an incomplete checkpoint.")
                scores = score_masks(gt, predictions, category_ids)
                if any(not isclose(scores[k], record["scores"]["valid"][k], abs_tol=1e-9)
                       for k in METRICS if k != "ap_target"):
                    raise ValueError(f"{directory.name} validation scores could not be reproduced.")
            if set(record["scores"]) != {"valid"}:
                raise ValueError(f"{directory.name} evaluated outside validation.")
            rows.append(make_row(
                path, directory.name, seed, head, size, public_hash, training_hash,
                record, scores, class_names, timeline, checkpoint=not recovered,
            ))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation.csv").open("w", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)
    if missing:
        print("Pending comparable artifacts:", ", ".join(missing), flush=True)
        return
    grouped = {cfg: [r for r in rows if r["run"].startswith(f"{cfg}-")] for cfg in CONFIGS}
    aggregates = {
        cfg: {
            key: mean(r[key] for r in runs)
            for key in METRICS
        }
        for cfg, runs in grouped.items()
    }
    peak = max(data["ap"] for data in aggregates.values())
    competitive = [cfg for cfg, data in aggregates.items() if peak - data["ap"] < 0.0005]
    winner = min(competitive, key=lambda c: (grouped[c][0]["rcnn_size"], -aggregates[c]["ap"]))
    elected = next(r for r in grouped[winner] if r["seed"] == 97)
    improvement = aggregates[winner]["ap"] > aggregates["random-640"]["ap"]
    checkpoint_ready = elected["checkpoint_available"] and improvement
    checkpoint_sha256 = None
    if checkpoint_ready:
        directory = output / elected["run"]
        directory.mkdir(exist_ok=True)
        checkpoint = directory / "checkpoint.pt"
        temporary = checkpoint.with_suffix(".part")
        with ZipFile(elected["artifact"]) as archive, archive.open("checkpoint.pt") as source:
            with temporary.open("wb") as sink:
                copyfileobj(source, sink)
        temporary.replace(checkpoint)
        checkpoint_sha256 = sha256(checkpoint)
    result = {
        "dataset_sha256": public_hash, "train_valid_sha256": training_hash, "runs": rows,
        "aggregate": aggregates, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "variant": elected["model"], "condition": winner, "checkpoint": elected["run"],
            "mean_validation_gain": aggregates[winner]["ap"] - aggregates["random-640"]["ap"],
            "rcnn_size": elected["rcnn_size"], "ready_for_public_test": checkpoint_ready,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    write_json(output / "selection.json", result)
    for cfg, scores in aggregates.items():
        print(f"{cfg}: Mask AP {100*scores['ap']:.2f}% | AP small {100*scores['ap_small']:.2f}%", flush=True)
    print(f"Selected {elected['run']}; ready for public test: {checkpoint_ready}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--original", type=Path, default=Path("data/panoptic-indoor-v5-640.zip"))
    parser.add_argument("--train-valid", type=Path, default=Path("data/panoptic-indoor-v5-train-valid-640.zip"))
    parser.add_argument("--prior", type=Path, default=Path("output/research/research.json"))
    parser.add_argument("--output", type=Path, default=Path("output/improvement"))
    args = parser.parse_args()
    collect(args.registry, args.original, args.train_valid, args.prior, args.output)
