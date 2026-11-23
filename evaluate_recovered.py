"""Evaluate a SHA-verified checkpoint and original Mask R-CNN training CSV."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from train import run
from metrics import write_json


def evaluate_recovered(options):
    datetime.fromisoformat(options.training_start)
    with TemporaryDirectory(prefix="indoor-recovered-") as folder:
        temp = Path(folder)
        with ZipFile(options.recovered) as source:
            source.extract("checkpoint.pt", temp)
            source.extract("results.csv", temp)
        with (temp / "results.csv").open(encoding="utf-8", newline="") as log:
            rows = list(csv.DictReader(log))
        if [int(row["epoch"]) for row in rows] != list(range(1, 17)):
            raise ValueError("Recovered checkpoint must have sixteen logged epochs.")
        if any(int(row["batches"]) != 199 for row in rows):
            raise ValueError("Recovered CSV has an incomplete training epoch.")
        evaluation = temp / "evaluation.zip"
        checkpoint = temp / "checkpoint.pt"
        run(SimpleNamespace(
            device="0", seed=42, dataset=options.dataset, classes=None, checkpoint=checkpoint,
            model="maskrcnn", pilot=False, eval_splits=["valid", "test"], output=evaluation,
            epochs=16,
        ))
        options.output.parent.mkdir(parents=True, exist_ok=True)
        pending = options.output.with_suffix(".part")
        with ZipFile(evaluation) as source, ZipFile(pending, "w", ZIP_DEFLATED) as output:
            summary = json.loads(source.read("summary.json"))
            summary.update({
                "evaluation_gpu": summary["gpu"], "gpu": options.training_gpu,
                "evaluation_started_at_utc": summary["started_at_utc"],
                "started_at_utc": options.training_start,
                "target_epochs": 16, "completed_epochs": 16,
                "training_seconds": sum(float(row["epoch_seconds"]) for row in rows),
                "training_clock": "sum of completed epoch times in the original results.csv",
                "recovered_training_run": "full-maskrcnn",
            })
            metadata = temp / "summary.json"
            write_json(metadata, summary)
            output.write(metadata, "summary.json")
            output.write(temp / "results.csv", "results.csv")
            output.write(checkpoint, "checkpoint.pt", compress_type=ZIP_STORED)
            for name in source.namelist():
                if name.startswith("predictions-"):
                    output.writestr(name, source.read(name))
        pending.replace(options.output)
        print(f"Evaluation and recovered training evidence saved in {options.output}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", required=True, type=Path)
    parser.add_argument("--training-start", required=True)
    parser.add_argument("--training-gpu", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/panoptic-indoor-v5-640.zip"))
    parser.add_argument("--output", type=Path, default=Path("output/result.zip"))
    evaluate_recovered(parser.parse_args())
