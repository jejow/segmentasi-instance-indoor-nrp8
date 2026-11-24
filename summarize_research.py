"""Verify and summarize the paired 20-epoch Mask R-CNN replications."""
import argparse
from collections import defaultdict
import csv
from hashlib import file_digest
import json
from pathlib import Path
from shutil import copyfileobj
from statistics import mean, stdev
from struct import Struct
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

from metrics import write_json


MODELS = ("maskrcnn", "maskrcnn-v2")
SEEDS = (42, 97, 123)
TARGETS = ("knife", "light", "spoon")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_partial_metrics(partial):
    """Recover only intact metric members of a truncated ZIP download."""
    header = Struct("<4s5H3L2H")
    entries = {}
    with partial.open("rb") as stream:
        for expected in ("summary.json", "results.csv"):
            signature, _, flags, method, _, _, crc, length, decoded_length, name_size, extra = (
                header.unpack(stream.read(header.size))
            )
            if signature != b"PK\x03\x04" or flags & 9 or length > 2_000_000 or decoded_length > 2_000_000:
                raise ValueError(f"Unexpected ZIP layout in {partial.name}.")
            name = stream.read(name_size).decode("utf-8")
            stream.seek(extra, 1)
            compressed = stream.read(length)
            if method not in (ZIP_STORED, ZIP_DEFLATED) or len(compressed) != length:
                raise ValueError(f"Truncated or unsupported member: {name}.")
            data = zlib.decompress(compressed, -15) if method == ZIP_DEFLATED else compressed
            if name != expected or len(data) != decoded_length or zlib.crc32(data) != crc:
                raise ValueError(f"CRC, member name, or size mismatch: {name}.")
            entries[name] = data
    return entries


def collect(registry, dataset, output):
    fingerprint = digest(dataset)
    rows, losses = [], {}
    recovery = output / "metrics-only"
    for model in MODELS:
        for seed in SEEDS:
            run = registry / f"{model}-{seed}"
            status = json.loads((run / "progress.json").read_text(encoding="utf-8"))
            files = list((run / "artifacts").rglob("result.zip"))
            if len(files) == 1 and status["phase"] == "complete":
                artifact = files[0]
                with ZipFile(artifact) as archive:
                    entries = {key: archive.read(key) for key in ("summary.json", "results.csv")}
                verification = "complete verified archive"
            elif status["phase"] == "interrupted":
                partial = sorted((run / "artifacts").rglob("result.zip.*.part"))
                if len(partial) != 1:
                    raise RuntimeError(f"{run.name}: no unique recoverable archive.")
                artifact = partial[0]
                entries = read_partial_metrics(artifact)
                verified = recovery / run.name
                verified.mkdir(parents=True, exist_ok=True)
                for name, data in entries.items():
                    (verified / name).write_bytes(data)
                verification = "metrics recovered from partial archive, member CRC verified"
            else:
                raise RuntimeError(f"{run.name}: verified download pending ({status['phase']}).")
            record = json.loads(entries["summary.json"])
            calibration = list(csv.DictReader(entries["results.csv"].decode("utf-8").splitlines()))
            expected = model, seed, fingerprint, 20
            observed = tuple(record[key] for key in ("model", "seed", "dataset_sha256", "completed_epochs"))
            if observed != expected or [int(entry["epoch"]) for entry in calibration] != list(range(1, 21)):
                raise ValueError(f"{run.name}: model, seed, dataset, or budget mismatch.")
            if set(record["scores"]) != {"valid"} or record["scores"]["valid"]["images"] != 166:
                raise ValueError(f"{run.name}: validation split differs from protocol.")
            valid = record["scores"]["valid"]
            rows.append({
                "run": run.name, "model": model, "seed": seed, "epochs": 20,
                "gpu": record["gpu"], "profile": status["colab_profile"],
                "minutes": record["training_seconds"] / 60,
                "ap": valid["ap"], "ap50": valid["ap50"], "ap75": valid["ap75"],
                "ap_target": mean(valid["per_class"][label]["ap"] for label in TARGETS),
                "artifact": str(artifact), "artifact_sha256": digest(artifact), "verification": verification,
            })
            losses[run.name] = [float(entry["train_loss"]) for entry in calibration]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validasi-enam-run.csv").open("w", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    aggregate = {
        model: {
            metric: {"mean": mean(row[metric] for row in group), "sd": stdev(row[metric] for row in group)}
            for metric in ("ap", "ap50", "ap75", "ap_target", "minutes")
        }
        for model, group in groups.items()
    }
    winner = max(MODELS, key=lambda model: aggregate[model]["ap"]["mean"])
    representative = sorted(groups[winner], key=lambda row: (row["ap"], row["seed"]))[1]
    selected = representative["run"]
    if representative["verification"] != "complete verified archive":
        raise RuntimeError(f"The selected checkpoint could not be recovered: {selected}.")
    paired = [groups[MODELS[1]][i]["ap"] - groups[MODELS[0]][i]["ap"] for i in range(len(SEEDS))]
    report = {
        "dataset_sha256": fingerprint, "runs": rows, "aggregate": aggregate,
        "v2_minus_v1_per_seed": dict(zip(SEEDS, paired)),
        "selection": {"variant": winner, "checkpoint": selected, "rule": "highest mean AP, median AP seed"},
    }
    write_json(output / "research.json", report)
    checkpoint = output / selected / "checkpoint.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    with ZipFile(representative["artifact"]) as archive:
        with archive.open("checkpoint.pt") as source, checkpoint.open("wb") as target:
            copyfileobj(source, target)
    plot(aggregate, groups, losses, output / "replikasi-maskrcnn.png")
    for model, metrics in aggregate.items():
        print(f"{model}: AP {100 * metrics['ap']['mean']:.2f}% +/- {100 * metrics['ap']['sd']:.2f} pp")
    print(f"Selected: {selected}")


def plot(aggregate, groups, losses, destination):
    import matplotlib.pyplot as plt

    fig, (scores, curves) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ("#25629c", "#16887c")
    labels = ("R50-FPN", "R50-FPN v2")
    for index, (model, color, label) in enumerate(zip(MODELS, colors, labels)):
        points = [100 * row["ap"] for row in groups[model]]
        stats = aggregate[model]["ap"]
        scores.scatter([index + offset for offset in (-.08, 0, .08)], points, color=color, alpha=.85)
        scores.errorbar(index, 100 * stats["mean"], yerr=100 * stats["sd"], color="#172631",
                        fmt="_", capsize=5, markersize=12, linewidth=1.25)
        for seed in SEEDS:
            curves.plot(range(1, 21), losses[f"{model}-{seed}"], color=color, alpha=.72,
                        label=label if seed == SEEDS[0] else None)
    scores.set_xticks([0, 1], labels)
    scores.set_ylabel("Mask AP50–95 validasi (%)")
    scores.set_title("Tiga seed per varian", loc="left")
    curves.set_xlabel("Epoch")
    curves.set_ylabel("Loss train rata-rata per batch")
    curves.set_title("Kurva training di Colab T4", loc="left")
    curves.legend(frameon=False)
    for chart in (scores, curves):
        chart.grid(axis="y", alpha=.2)
        chart.spines[["right", "top"]].set_visible(False)
    fig.savefig(destination, dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/panoptic-indoor-v5-640.zip"))
    parser.add_argument("--output", type=Path, default=Path("output/research"))
    options = parser.parse_args()
    collect(options.registry, options.dataset, options.output)
