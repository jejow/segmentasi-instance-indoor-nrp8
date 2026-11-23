"""Freeze the three most difficult classes using pilot validation predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from metrics import load_json, write_json


def rank_classes(dataset, pilots):
    with dataset.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    with ZipFile(dataset) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    baseline = {}
    for pilot in pilots:
        with ZipFile(pilot) as archive:
            record = json.loads(archive.read("summary.json"))
        if not record["pilot"] or record["dataset_sha256"] != checksum:
            raise ValueError(f"Non-pilot or different dataset: {pilot}")
        if record["model"] in baseline:
            raise ValueError(f'Duplicate architecture {record["model"]}')
        if record["completed_epochs"] != 3:
            raise ValueError(f"Pilot did not finish three epochs: {pilot}")
        baseline[record["model"]] = record["scores"]["valid"]["per_class"]
    train = manifest["splits"]["train"]
    valid = manifest["splits"]["valid"]
    rows = []
    for name in manifest["categories"]:
        count_train = train["class_images"].get(name, 0)
        count_valid = valid["class_images"].get(name, 0)
        if count_train < 5 or count_valid < 3:
            continue
        scores = {model: per_class[name]["ap"] for model, per_class in baseline.items()}
        if any(value is None for value in scores.values()):
            continue
        rows.append({
            "name": name, "mean_valid_ap": mean(scores.values()),
            "train_images": count_train, "valid_images": count_valid,
            "valid_median_mask_fraction": valid["median_mask_fraction"][name], **scores,
        })
    rows.sort(key=lambda row: (row["mean_valid_ap"], row["valid_median_mask_fraction"],
                               row["train_images"], row["name"]))
    if len(rows) < 3:
        raise ValueError("Fewer than three classes satisfy the predeclared support requirements.")
    return rows, checksum, list(baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilots", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/panoptic-indoor-v5-640.zip"))
    parser.add_argument("--output", type=Path, default=Path("hard-classes.json"))
    parser.add_argument("--minimum-pilots", type=int, default=3)
    args = parser.parse_args()
    if len(args.pilots) < args.minimum_pilots:
        parser.error(f"At least {args.minimum_pilots} completed pilot results are required.")
    ranking, checksum, models = rank_classes(args.dataset, args.pilots)
    selected = {
        "classes": [row["name"] for row in ranking[:3]], "dataset_sha256": checksum,
        "source_split": "valid", "pilot_models": models,
        "method": "lowest mean pilot mask AP50-95; support >=5 train and >=3 validation images",
        "tie_break": "smaller median relative mask area in validation, then fewer training images",
        "top_three": ranking[:3],
    }
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, selected)
    with path.with_name("peringkat-kesulitan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
