"""Plot additional report figures from audited metrics and mask-match counts."""
from collections import Counter
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from metrics import load_json


SOURCE = Path("bukti/hasil")
DEST = Path("bukti/visual")
EVALUATION = Path("output/test-pribadi/eval-v3/comparison.json")
RUNS = ("full-yolov8n-r2", "full-yolo11n-r3", "full-maskrcnn-eval", "maskrcnn-v2-97", "rcnn-v2-coco800")
THREE = ("three-yolov8n", "three-yolo11n", "three-maskrcnn-r3")
MODELS = ("YOLOv8n-seg", "YOLO11n-seg", "Mask R-CNN", "Mask R-CNN v2 640", "Mask R-CNN v2 COCO 800")
SHORT = ("Y8", "Y11", "R-CNN", "v2 640", "v2 800")
PALETTE = ("#32618c", "#118276", "#b25826", "#655299", "#823c56")
LINESTYLES = ("-", "--", "-.", ":", "-")
FULL_COLOR = "#235d87"
THREE_COLOR = "#b35b28"
PHONE_COLOR = "#087b70"
INK = "#1c2935"
BG = "#edf2f5"
METRICS = ("ap", "ap50", "ap75")
TARGETS = ("knife", "light", "spoon")
CMAP = colors.LinearSegmentedColormap.from_list("mask_ap", ["#f8fbfd", "#c0dbeb", "#5a9ac2", "#17507b"])
NA = "#ededeb"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(figure, name):
    figure.savefig(DEST / f"{name}.png", dpi=178, facecolor="white")
    plt.close(figure)


def fmt(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def ap_label(value, digits=2):
    if value is None:
        return "–"
    if value == 0:
        return "0"
    if value < 10 ** -digits:
        return "<" + fmt(10 ** -digits, digits)
    return fmt(value, digits)


def polished(axis, grid="y"):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis=grid, color="#dce4ea", linewidth=.7)


def class_scores():
    rows = read_csv(SOURCE / "ap-per-kelas.csv")
    comparison = load_json(EVALUATION)
    improvement = load_json("output/improvement/public-test.json")
    phone_improvement = load_json("output/improvement/phone-test.json")
    selected = {
        "public": improvement["improved_configuration"]["scores"]["per_class"],
        "private": phone_improvement["selected"]["private_native"]["per_class"],
    }
    for name in sorted(selected["public"]):
        row = {"run": RUNS[-1], "class": name}
        for scope in selected:
            detail = selected[scope][name]
            row.update({f"{scope}_{key}": "" if detail[key] is None else str(detail[key])
                        for key in (*METRICS, "instances")})
        row["common"] = str(all(int(row[f"{scope}_instances"]) > 0 for scope in selected))
        rows.append(row)

    classes = {row["class"]: row for row in rows if row["run"] == RUNS[0]}
    shared = {name for name, row in classes.items() if row["common"] == "True"}
    mapping = {(row["run"], row["class"]): row for row in rows}
    if len(shared) != 37 or len(mapping) != len(rows):
        raise ValueError("Unexpected class support or duplicate source rows.")
    if any(sum(int(row[f"{domain}_instances"]) > 0 for row in classes.values()) != count
           for domain, count in (("public", 69), ("private", 39))):
        raise ValueError("Unexpected number of annotated test categories.")
    for run in RUNS:
        model_row = next(row for row in comparison["models"] if row["run"] == run) if run != RUNS[-1] else None
        for scope in selected:
            for key in METRICS:
                actual = mean(float(mapping[run, name][f"{scope}_{key}"]) for name in sorted(shared))
                expected = (model_row[f"{scope}_common_{key}"] if model_row else
                            phone_improvement["selected"][f"{scope}_common"][key])
                if abs(actual - expected) > 1e-10:
                    raise ValueError(f"Shared class AP disagrees with the archived evaluation: {run} {scope} {key}.")
    nrp8 = load_json(SOURCE / "ringkasan.json")["nrp8"]
    for index, run in enumerate((*RUNS[:3], *THREE)):
        row = next(row for row in nrp8 if row["model"] == ("yolov8n", "yolo11n", "maskrcnn")[index % 3]
                   and row["condition"] == ("full" if index < 3 else "three"))
        for domain, field in (("public", "public_targets_pct"), ("private", "phone_targets_pct")):
            result = mean(100 * float(mapping[run, target][f"{domain}_ap"]) for target in TARGETS)
            if abs(result - row[field]) > 1e-10:
                raise ValueError(f"Target AP disagrees with NRP 8 summary: {run} {domain}.")

    with (DEST / "ap-kelas-lengkap.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("run", "class", "common", *(f"{scope}_{key}" for scope in selected
                                             for key in (*METRICS, "instances")))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return classes, shared, mapping


def class_heatmaps(classes, mapping):
    recorded = sorted(name for name, row in classes.items()
                      if max(int(row["public_instances"]), int(row["private_instances"])) > 0)
    limit = max(100 * float(row[f"{scope}_ap"]) for row in mapping.values() for scope in ("public", "private")
                if row["run"] in RUNS and row[f"{scope}_ap"])
    norm = colors.PowerNorm(.55, vmin=0, vmax=limit)
    for start in range(0, len(recorded), 18):
        names = recorded[start:start + 18]
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 1.5 + len(names) * .30), layout="constrained")
        for axis, (scope, title) in zip(axes, (("public", "Test resmi · 359 foto"), ("private", "Foto HP · 40 foto")), strict=True):
            matrix = []
            counts = [int(classes[name][f"{scope}_instances"]) for name in names]
            for name in names:
                matrix.append([
                    100 * float(mapping[run, name][f"{scope}_ap"])
                    if mapping[run, name][f"{scope}_ap"] else np.nan for run in RUNS
                ])
            cells = CMAP(norm(np.nan_to_num(matrix)))
            cells[np.isnan(matrix)] = colors.to_rgba(NA)
            counts_col = np.tile(colors.to_rgba(BG), (len(names), 1, 1))
            axis.imshow(np.concatenate((counts_col, cells), axis=1), aspect="auto")
            for row_index, row in enumerate(matrix):
                axis.text(0, row_index, str(counts[row_index]), ha="center", va="center", fontsize=8.3, color=INK)
                for col_index, value in enumerate(row, 1):
                    color = "white" if not np.isnan(value) and norm(value) >= .75 else INK
                    axis.text(col_index, row_index, ap_label(None if np.isnan(value) else value),
                              ha="center", va="center", fontsize=8.1, color=color)
            axis.set_xticks(range(6), ["n", *SHORT])
            axis.xaxis.tick_top()
            axis.tick_params(length=0, labelsize=9)
            axis.set_yticks(range(len(names)), names if scope == "public" else [""] * len(names))
            axis.set_title(title, loc="left", fontsize=12, weight="bold", pad=15)
            axis.set_xticks(np.arange(-.5, 6, 1), minor=True)
            axis.set_yticks(np.arange(-.5, len(names), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=1)
            axis.tick_params(which="minor", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
        scale = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes, orientation="horizontal",
                             shrink=.57, pad=.045, aspect=45)
        scale.set_label("Mask AP50-95 (%)", size=9)
        scale.ax.tick_params(labelsize=8)
        save(fig, f"ap-kelas-{start // 18 + 1}")


def target_heatmap(classes, mapping):
    rows = [(model, run, condition) for model_index, model in enumerate(MODELS[:3])
            for run, condition in ((RUNS[model_index], "semua"), (THREE[model_index], "3 kelas"))]
    matrices = [[[100 * float(mapping[run, target][f"{scope}_ap"]) for target in TARGETS]
                 for _, run, _ in rows] for scope in ("public", "private")]
    vmax = max(np.max(matrix) for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.6), layout="constrained")
    for axis, matrix, (scope, title) in zip(axes, matrices, (("public", "Test resmi"), ("private", "Foto HP")), strict=True):
        axis.imshow(matrix, cmap=CMAP, vmin=0, vmax=vmax, aspect="auto")
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                axis.text(x, y, ap_label(value, 3), ha="center", va="center",
                          color="white" if value / vmax > .70 else INK, fontsize=9.5)
        axis.set_xticks(range(3), [f"{target}\nn={classes[target][f'{scope}_instances']}" for target in TARGETS])
        axis.set_yticks(range(6), [f"{model} · {condition}" for model, _, condition in rows] if scope == "public" else [])
        axis.tick_params(length=0, labelsize=9)
        axis.axhline(1.5, color="white", lw=2)
        axis.axhline(3.5, color="white", lw=2)
        axis.set_title(title, loc="left", weight="bold")
        for spine in axis.spines.values():
            spine.set_visible(False)
    scale = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, vmax), cmap=CMAP),
                         ax=axes, orientation="horizontal", shrink=.62, aspect=42, pad=.06)
    scale.set_label("Mask AP50-95 (%)", size=9)
    save(fig, "ap-target-nrp8")


def aggregate_scores(shared, mapping):
    values = {
        (scope, key): [100 * mean(float(mapping[run, cls][f"{scope}_{key}"]) for cls in sorted(shared))
                       for run in RUNS]
        for scope in ("public", "private") for key in METRICS
    }
    xmax = max(max(row) for row in values.values()) * 1.19
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), sharey=True, layout="constrained")
    y = np.arange(len(RUNS))
    for axis, key, title in zip(axes, METRICS, ("AP50-95", "AP50", "AP75"), strict=True):
        for scope, delta, color, symbol in (("public", -.155, FULL_COLOR, "o"), ("private", .155, PHONE_COLOR, "s")):
            row = values[scope, key]
            axis.scatter(row, y + delta, color=color, marker=symbol, s=26, zorder=3)
            for position, value in zip(y, row, strict=True):
                axis.annotate(fmt(value), (value, position + delta), xytext=(4, 0), textcoords="offset points",
                              va="center", fontsize=7.8, color=INK)
        axis.set_title(title, weight="bold", loc="left")
        axis.set_xlim(0, xmax)
        axis.set_xlabel("Mask AP (%)")
        polished(axis, grid="x")
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, MODELS)
    axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], color=FULL_COLOR, marker="o", ls="", label="Test resmi"),
                        Line2D([], [], color=PHONE_COLOR, marker="s", ls="", label="Foto HP")],
               ncol=2, frameon=False, loc="outside lower center")
    save(fig, "ap-tiga-ambang")


def coverage(classes, overlay):
    coverage_rows = overlay["cakupan_hp"]
    count = Counter(entry["sulit"] for entry in coverage_rows)
    if count != {False: 28, True: 12} or sum(row["mask"] for row in coverage_rows) != 619:
        raise ValueError("Photo coverage differs from the evaluated v3 split.")
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": (1.5, 1)},
                                      layout="constrained")
    for hard, label, color, border, hatch in [(False, "Reguler · 28 foto", FULL_COLOR, FULL_COLOR, ""),
                                              (True, "Sulit · 12 foto", THREE_COLOR, "#6f391e", "//")]:
        rows = [row for row in coverage_rows if row["sulit"] == hard]
        left.bar([r["n"] for r in rows], [r["mask"] for r in rows], color=color, width=.82,
                 label=label, hatch=hatch, edgecolor=border)
    left.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    left.set_ylabel("Jumlah mask acuan")
    left.set_xlabel("Nomor foto dalam lampiran lokal")
    left.legend(frameon=False, loc="upper left", fontsize=9)
    left.set_title("Sebaran pada 40 foto HP", loc="left", weight="bold")
    polished(left)

    top = sorted(classes, key=lambda name: -int(classes[name]["private_instances"]))[:8]
    values = [int(classes[name]["private_instances"]) for name in top]
    bars = right.barh(np.arange(len(top)), values, color=PHONE_COLOR)
    right.bar_label(bars, padding=3, fontsize=8.5)
    right.set_yticks(np.arange(len(top)), top, fontsize=9)
    right.invert_yaxis()
    right.set_xlim(0, max(values) * 1.22)
    right.set_xlabel("Jumlah mask acuan")
    right.set_title("Delapan kelas terbanyak", loc="left", weight="bold")
    polished(right, grid="x")
    save(fig, "cakupan-foto-hp")


def difficult_and_sizes():
    comparison = load_json(EVALUATION)
    public = load_json("output/improvement/public-test.json")
    phone = load_json("output/improvement/phone-test.json")
    fig, axis = plt.subplots(1, 1, figsize=(8.3, 3.0), layout="constrained")
    groups = [comparison["comparisons"][key]["difficulty"] for key in RUNS[:3]]
    if len(groups[0]["classes"]) != 23 or any(row["classes"] != groups[0]["classes"] for row in groups):
        raise ValueError("Difficulty class support is different between models.")
    y = np.arange(3)
    for key, offset, color, border, pattern in (("difficult", -.18, THREE_COLOR, "#6f391e", "//"),
                                                 ("other", .18, FULL_COLOR, FULL_COLOR, "")):
        scores = [100 * group[key]["ap"] for group in groups]
        label = "Sulit · 12 foto" if key == "difficult" else "Reguler · 28 foto"
        bars = axis.barh(y + offset, scores, .31, label=label, color=color, hatch=pattern, edgecolor=border)
        axis.bar_label(bars, labels=[fmt(value) for value in scores], padding=3, fontsize=9)
    axis.set_yticks(y, MODELS[:3])
    axis.invert_yaxis()
    axis.set_xlim(0, 6.5)
    axis.set_xlabel("Mask AP50-95 (%) pada 23 kelas bersama")
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    polished(axis, grid="x")
    save(fig, "ap-kondisi-sulit")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True, layout="constrained")
    settings = (("Test resmi · 69 kelas", [public[key]["scores"] for key in ("control", "improved_configuration")]),
                ("Foto HP · 39 kelas", [phone[key]["private_native"] for key in ("control", "selected")]))
    sizes = ("small", "medium", "large")
    x = np.arange(3)
    for axis, (title, scores) in zip(axes, settings, strict=True):
        for row, offset, color, border, hatch in zip(scores, (-.18, .18), (FULL_COLOR, PHONE_COLOR),
                                                    (FULL_COLOR, "#08534b"), ("", "//"), strict=True):
            values = [row[f"ap_{key}"] * 100 for key in sizes]
            bars = axis.bar(x + offset, values, .32, color=color, hatch=hatch, edgecolor=border)
            axis.bar_label(bars, labels=[fmt(value) for value in values], padding=3, fontsize=8)
        axis.set_xticks(x, ["Kecil", "Sedang", "Besar"])
        axis.set_title(title, weight="bold", loc="left")
        polished(axis)
    upper = max(row[f"ap_{key}"] for _, rows in settings for row in rows for key in sizes)
    axes[0].set_ylim(0, 121 * upper)
    axes[0].set_ylabel("Mask AP50-95 (%)")
    fig.legend(handles=[Patch(color=FULL_COLOR, label="Mask R-CNN v2 kontrol 640"),
                        Patch(facecolor=PHONE_COLOR, edgecolor="#08534b", hatch="//", label="Mask R-CNN v2 COCO 800")],
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "ap-ukuran-mask")


def training():
    logs = Path("bukti/log-training/inti")
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.4), layout="constrained")
    max_yolo = 0
    for index, (axis, name) in enumerate(zip(axes, MODELS[:3], strict=True)):
        metric = "metrics/mAP50-95(M)" if index < 2 else "train_loss"
        for run, color, symbol, style in ((RUNS[index], FULL_COLOR, "o", "-"),
                                           (THREE[index], THREE_COLOR, "s", "--")):
            rows = read_csv(logs / run / "results.csv")
            epochs = [int(row["epoch"]) for row in rows]
            if epochs != list(range(1, 17)):
                raise ValueError(f"Unexpected epoch coverage for {run}.")
            values = [float(row[metric]) * (100 if index < 2 else 1) for row in rows]
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError(f"Unexpected training metric for {run}.")
            if index < 2:
                max_yolo = max(max_yolo, *values)
            axis.plot(epochs, values, color=color, marker=symbol, ms=3, linewidth=1.55, ls=style)
        axis.set_title(name, loc="left", weight="bold")
        axis.set_xlabel("Epoch")
        axis.set_xticks([1, 4, 8, 12, 16])
        polished(axis)
    for axis in axes[:2]:
        axis.set_ylim(0, max_yolo * 1.12)
    axes[0].set_ylabel("Mask AP50-95 validasi (%)")
    axes[2].set_ylabel("Training loss")
    axes[2].set_yscale("log")
    fig.legend(handles=[Line2D([], [], color=FULL_COLOR, marker="o", label="Semua kelas"),
                        Line2D([], [], color=THREE_COLOR, marker="s", ls="--", label="Tiga kelas")],
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "kurva-training-utama")


def threshold_curves(overlay):
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.9), sharex=True, sharey=True, layout="constrained")
    for axis, (key, title) in zip(axes, (("test_v5", "Test resmi"), ("test_hp", "Foto HP")), strict=True):
        part = overlay[key]
        for model, color, style in zip(MODELS, PALETTE, LINESTYLES, strict=True):
            for row in part["runs"][model]:
                if not 0 <= row["matched"] <= min(row["predictions"], part["reference_masks"]):
                    raise ValueError(f"Invalid population for {model} {key}.")
            points = [
                (100 * row["matched"] / part["reference_masks"], 100 * row["matched"] / row["predictions"],
                 row["threshold"])
                for row in part["runs"][model] if row["predictions"]
            ]
            recall, precision, _ = zip(*points)
            axis.plot(recall, precision, color=color, linewidth=1.6, ls=style)
            for threshold, marker in ((.15, "o"), (.30, "s")):
                selected = next((point for point in points if point[2] == threshold), None)
                if selected:
                    axis.scatter(*selected[:2], color=color, marker=marker, s=29, zorder=3)
        axis.set_title(f"{title} · n={part['reference_masks']} acuan", loc="left", weight="bold")
        axis.set_xlabel("Recall mikro (%)")
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        polished(axis)
    axes[0].set_ylabel("Precision mikro (%)")
    handles = [Line2D([], [], color=color, ls=style, label=name)
               for color, style, name in zip(PALETTE, LINESTYLES, MODELS, strict=True)]
    handles += [Line2D([], [], color=INK, marker=marker, ls="", label=label)
                for label, marker in (("0,15", "o"), ("0,30", "s"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8.4)
    save(fig, "precision-recall-confidence")


def gamma_iou():
    examples = sorted(load_json(SOURCE / "ringkasan.json")["gamma_examples"], key=lambda row: row["case"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.6), sharey=True, layout="constrained")
    y = np.arange(len(examples))
    modes = (("control", "Kontrol", FULL_COLOR, "o"), ("brightened", "Gamma 0,65", PHONE_COLOR, "s"))
    for axis, (metric, title) in zip(axes,
                                     (("same_class_iou", "Kandidat sekelas"),
                                      ("nearest_iou", "Kandidat terbaik lintas kelas")), strict=True):
        for index, (field, label, color, marker) in enumerate(modes):
            values = [row[field][metric] for row in examples]
            positions = y + (index - .5) * .30
            axis.scatter(values, positions, color=color, marker=marker, s=39, label=label)
            for row, value in zip(positions, values, strict=True):
                axis.annotate(fmt(value, 3), (value, row), xytext=(5, 0),
                              textcoords="offset points", va="center", fontsize=8.4)
        axis.set_xlim(-.035, 1.08)
        axis.set_xticks([0, .25, .50, .75, 1])
        axis.set_xlabel("IoU mask")
        axis.set_title(title, loc="left", weight="bold")
        polished(axis, grid="x")
    axes[0].set_yticks(y, [f"Kasus {row['case']} · {row['class']}" for row in examples])
    axes[0].set_ylim(len(examples) - .42, -.58)
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls="", label=label)
                        for _, label, color, marker in modes],
               loc="outside lower center", frameon=False, ncol=2)
    save(fig, "gamma-iou-kasus")


def main():
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    overlay = load_json(DEST / "ringkasan-overlay.json")
    classes, shared, mapping = class_scores()
    coverage(classes, overlay)
    target_heatmap(classes, mapping)
    aggregate_scores(shared, mapping)
    difficult_and_sizes()
    training()
    threshold_curves(overlay)
    class_heatmaps(classes, mapping)
    gamma_iou()
    print("Grafik laporan dan CSV AP per kelas selesai.")


if __name__ == "__main__":
    main()
