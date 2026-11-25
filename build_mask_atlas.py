"""Render landscape mask plates from the frozen public and phone test archives."""
from collections import Counter, defaultdict
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as coco_mask
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from build_visual_report import EVAL, FULL, LABELS, MODELS, PUBLIC, THREE, grid, photo, read
from audit_private_test import PHOTO
from metrics import load_json, write_json
from prepare_dataset import hash_file


DISPLAY_SCORE = 0.15
EVAL_SCORE = 0.005
MATCH_IOU = 0.50
THRESHOLDS = (0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
TARGETS = ("knife", "light", "spoon")
PDF_LOCAL = Path("output/report/atlas-foto-hp.pdf")
PDF_PUBLIC = PUBLIC / "atlas-v5.pdf"
V5 = Path("data/panoptic-indoor-v5-640.zip")
V5_TEST = Path("output/tuning/test-v5.zip")
V5_RAW = Path("data/panoptic-indoor-v5-original.coco.zip")
SELECTED = Path("output/improvement/coco-800-97-retry1")
PHONE = EVAL / "photo-test-640.zip"
SOURCE_URL = "https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
INK = colors.HexColor("#1e2126")
MUTED = colors.HexColor("#535c66")
RULE = colors.HexColor("#d5dbe0")
WIDTH, HEIGHT = landscape(A4)
MARGIN = 32
IMAGE_SIZE = 126
COL_WIDTH = (WIDTH - 2 * MARGIN) / 4


@dataclass(frozen=True)
class ImageResult:
    detections: list
    matched: list

    def counts(self, threshold=DISPLAY_SCORE, classes=None):
        eligible = (
            matched for detection, matched in zip(self.detections, self.matched, strict=True)
            if detection["score"] >= threshold and (classes is None or detection["category_id"] in classes)
        )
        tally = Counter(eligible)
        return tally[True] + tally[False], tally[True]

    def visible(self, threshold=DISPLAY_SCORE):
        return [detection for detection in self.detections if detection["score"] >= threshold]


def match_image(annotations, predictions):
    """Greedy one-to-one matches at mask IoU 0.50 within the same category."""
    predictions = sorted(predictions, key=lambda row: row["score"], reverse=True)
    matched = [False] * len(predictions)
    if not annotations or not predictions:
        return ImageResult(predictions, matched)
    similarities = coco_mask.iou(
        [row["segmentation"] for row in predictions],
        [row["segmentation"] for row in annotations],
        [0] * len(annotations),
    )
    reference_classes = np.array([row["category_id"] for row in annotations])
    detection_classes = np.array([row["category_id"] for row in predictions])
    similarities[detection_classes[:, None] != reference_classes] = -1
    for index, iou in enumerate(similarities):
        target = int(np.argmax(iou))
        if iou[target] >= MATCH_IOU:
            matched[index] = True
            similarities[index + 1:, target] = -1
    return ImageResult(predictions, matched)


def collect_runs(split, ground_truth, annotations, dataset_hash, test_hash=None):
    models = (*MODELS, *MODELS, "maskrcnn-v2", "maskrcnn-v2")
    if split == "test":
        paths = [Path("output/runs") / run / "result.zip" for run in (*FULL, *THREE)]
        paths += [Path("output/research/maskrcnn-v2-97/test-public-colab.zip"),
                  SELECTED / "public-test-colab.zip"]
    else:
        paths = [EVAL / f"{run}.zip" for run in (*FULL, *THREE, "maskrcnn-v2-97")]
        paths.append(SELECTED / "private-test.zip")
        test_hash = dataset_hash
    names = {row["id"]: row["name"] for row in ground_truth["categories"]}
    known_images = {row["id"] for row in ground_truth["images"]}
    selected_classes = {uid for uid, name in names.items() if name in TARGETS}
    results, hashes = [], []
    for index, (path, model) in enumerate(zip(paths, models, strict=True)):
        expected_hash = test_hash if index >= 6 else dataset_hash
        hashes.append(hash_file(path))
        with ZipFile(path) as archive:
            summary = read(archive, "summary.json")
            rows = read(archive, f"predictions-{split}.json")
        classes = selected_classes if 3 <= index < 6 else names.keys()
        if (
            summary["model"] != model or summary["dataset_sha256"] != expected_hash
            or summary["input_size"] != 640 or summary["confidence"] != EVAL_SCORE
            or summary["max_detections"] != 100
            or set(summary["class_names"]) != {names[uid] for uid in classes}
            or summary["seed"] != (97 if index >= 6 else 42)
            or (summary.get("rcnn_size") or 640) != (800 if index == 7 else 640)
        ):
            raise ValueError(f"Archive settings disagree with the report: {path}")
        by_image = defaultdict(list)
        for row in rows:
            if (row["image_id"] not in known_images or row["category_id"] not in classes
                    or row["segmentation"]["size"] != [640, 640]
                    or not EVAL_SCORE - 1e-9 <= row["score"] <= 1):
                raise ValueError(f"Unexpected detection in {path}")
            by_image[row["image_id"]].append(row)
        results.append({
            image_id: match_image(annotations[image_id], by_image[image_id])
            for image_id in sorted(known_images)
        })
    return results, hashes


def class_color(category):
    hue = (category * 0.61803398875) % 1
    return tuple(round(value * 255) for value in hsv_to_rgb(hue, 0.78, 0.94))


def contour_mask(binary):
    inner = np.pad(binary, 1, constant_values=255)
    core = binary.copy()
    height, width = binary.shape
    for dy in range(3):
        for dx in range(3):
            np.minimum(core, inner[dy:dy + height, dx:dx + width], out=core)
    return Image.fromarray(binary - core)


def mask_overlay(pixels, rows, reference=False):
    """Use the same color for a category in every panel and draw instance contours."""
    canvas = pixels.copy()
    masks = []
    sorted_rows = sorted(rows, key=lambda row: row["area"], reverse=True) if reference else reversed(rows)
    for row in sorted_rows:
        binary = coco_mask.decode(row["segmentation"]) * 255
        color = class_color(row["category_id"])
        alpha = Image.fromarray(binary).point([0] * 255 + [100])
        canvas.paste(color, (0, 0), alpha)
        # An inner edge stays in the mask, including masks that touch the image border.
        masks.append((contour_mask(binary), color))
    for contour, color in masks:
        canvas.paste(color, (0, 0), contour)
    return canvas


def text(canvas, x, y, value, size=9, bold=False, paint=INK):
    canvas.setFillColor(paint)
    canvas.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    canvas.drawString(x, y, value)


def rule(canvas, x, start, end):
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.55)
    canvas.line(x, start, end, start)


def heading(canvas, section, title, meta):
    text(canvas, MARGIN, HEIGHT - 27, section.upper(), 8, True, MUTED)
    text(canvas, MARGIN, HEIGHT - 52, title, 20, True)
    text(canvas, MARGIN, HEIGHT - 69, meta, 9, paint=MUTED)
    rule(canvas, MARGIN, HEIGHT - 77, WIDTH - MARGIN)


def footer(canvas, page, public):
    rule(canvas, MARGIN, 34, WIDTH - MARGIN)
    if public:
        attribution = "Panoptic Indoor Segmentation v5 | panopticindoor, Roboflow Universe | CC BY 4.0 | Panel dan overlay"
        text(canvas, MARGIN, 22, attribution, 7, paint=MUTED)
        canvas.linkURL(SOURCE_URL, (MARGIN, 19, MARGIN + 241, 30), relative=0)
        canvas.linkURL(LICENSE_URL, (MARGIN + 241, 19, MARGIN + 289, 30), relative=0)
    else:
        text(canvas, MARGIN, 22, "Lampiran foto HP | Evaluasi v3 | Paket lokal", 7, paint=MUTED)
    text(canvas, WIDTH - MARGIN - 26, 22, page, 8, True, MUTED)


def square(canvas, pixels, x, y):
    preview = pixels.resize((420, 420), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=87, optimize=True)
    canvas.drawImage(ImageReader(buffer), x, y, width=IMAGE_SIZE, height=IMAGE_SIZE)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.45)
    canvas.rect(x, y, IMAGE_SIZE, IMAGE_SIZE)


def chips(canvas, x, y, rows, names, reference):
    if reference:
        details = [(uid, str(count)) for uid, count in Counter(r["category_id"] for r in rows).most_common(3)]
    else:
        details = [(row["category_id"], f"{row['score']:.2f}") for row in rows[:3]]
    for index, (uid, value) in enumerate(details):
        top = y - 16 * index
        canvas.setFillColor(colors.Color(*(channel / 255 for channel in class_color(uid))))
        canvas.rect(x, top - 4, 4, 4, fill=1, stroke=0)
        label = names[uid]
        text(canvas, x + 7, top, label if len(label) <= 17 else label[:14] + "...", 6.4)
        text(canvas, x + 7, top - 7, value.replace(".", ","), 6.4, paint=MUTED)


def cell(canvas, column, top, title, pixels, rows, names, info, *, reference=False):
    x = MARGIN + column * COL_WIDTH
    text(canvas, x, top, title, 10, True)
    square(canvas, pixels, x, top - 10 - IMAGE_SIZE)
    side = x + IMAGE_SIZE + 7
    for index, line in enumerate(info):
        text(canvas, side, top - 19 - 12 * index, line, 8, index == 0)
    chips(canvas, side, top - 79, rows, names, reference)


def model_info(result, count, threshold=DISPLAY_SCORE):
    masks, true = result.counts(threshold)
    return [f"{masks} mask", f"{true} cocok", f"{masks - true} FP", f"{count - true} FN"]


def instructions(canvas, top, title, lines):
    text(canvas, MARGIN, top, title, 11, True)
    for index, line in enumerate(lines):
        text(canvas, MARGIN, top - 17 - 12 * index, line, 8.5, paint=MUTED)


def draw_plate(canvas, title, meta, serial, pixels, annotations, runs, names, public):
    canvas.bookmarkPage(serial)
    canvas.addOutlineEntry(title, serial)
    heading(canvas, "Test resmi v5" if public else "Test foto HP · v3", title, meta)
    first, second, third = 503, 347, 191
    targets = [row for row in annotations if names[row["category_id"]] in TARGETS]
    cell(canvas, 0, first, "Foto masukan · 640 × 640", pixels, [], names, [])
    cell(canvas, 1, first, "Mask acuan COCO", mask_overlay(pixels, annotations, True), annotations, names,
         [f"{len(annotations)} mask", f"{len({row['category_id'] for row in annotations})} kelas"],
         reference=True)
    for column, index, label in [(2, 6, "Mask R-CNN v2 · 640"), (3, 7, "Mask R-CNN v2 · COCO 800")]:
        rows = runs[index].visible()
        cell(canvas, column, first, label, mask_overlay(pixels, rows), rows, names,
             model_info(runs[index], len(annotations)))

    instructions(canvas, second, "Dilatih dengan semua kelas", [
        "Semua mask acuan dihitung.",
        "Overlay memakai skor ≥ 0,15.",
        "Cocok jika kelas sama, IoU ≥ 0,50.",
        "Satu prediksi per mask acuan.",
        "FP = prediksi yang tak cocok.",
        "FN = mask acuan yang tak cocok.",
    ])
    instructions(canvas, third, "Dilatih dengan tiga kelas", [
        f"Acuan knife, light, spoon: {len(targets)}.",
        "FN hanya untuk tiga kelas ini.",
        "Warna sama untuk kelas yang sama.",
        "Label kecil: tiga skor tertinggi atau",
        "tiga kelas acuan terbanyak.",
        "Prediksi 0 berarti overlay kosong.",
    ])
    for index, title in enumerate(LABELS):
        for offset, top, count in [(0, second, len(annotations)), (3, third, len(targets))]:
            result = runs[index + offset]
            rows = result.visible()
            cell(canvas, index + 1, top, title, mask_overlay(pixels, rows), rows, names,
                 model_info(result, count))
    footer(canvas, serial, public)
    canvas.showPage()


def phone_index(canvas, images, annotations, difficult):
    canvas.bookmarkPage("HP-index")
    canvas.addOutlineEntry("Indeks 40 foto uji HP", "HP-index")
    heading(canvas, "Lampiran visual · foto HP", "Indeks 40 foto uji HP",
            "Klik baris untuk membuka lembar. Semua foto dan mask acuan berasal dari evaluasi v3.")
    positions = [(MARGIN, WIDTH / 2 - 14), (WIDTH / 2 + 14, WIDTH - MARGIN)]
    for (left, right), start in zip(positions, (0, 20), strict=True):
        text(canvas, left + 8, 500, "Foto", 8, True)
        text(canvas, left + 112, 500, "Kondisi", 8, True)
        text(canvas, right - 75, 500, "Mask acuan", 8, True)
        for offset, image in enumerate(images[start:start + 20]):
            serial = f"HP-{start + offset + 1:02d}"
            y = 481 - 20 * offset
            if offset % 2 == 0:
                canvas.setFillColor(colors.HexColor("#f3f5f7"))
                canvas.rect(left, y - 5, right - left, 18, fill=1, stroke=0)
            text(canvas, left + 8, y, serial, 9, True)
            text(canvas, left + 112, y, "Sulit" if image["id"] in difficult else "Reguler", 9)
            text(canvas, right - 54, y, str(len(annotations[image["id"]])), 9)
            canvas.linkRect("", serial, (left, y - 5, right, y + 13), relative=0, thickness=0)
    text(canvas, MARGIN, 62, "Mask ditampilkan pada confidence ≥ 0,15. Setiap lembar menunjukkan acuan dan delapan keluaran model.", 9)
    footer(canvas, "Indeks", public=False)
    canvas.showPage()


def source_rows(dataset, split, expected):
    with ZipFile(dataset) as archive:
        content = read(archive, f"{split}/gt.json")
    images = sorted(content["images"], key=lambda row: row["id"])
    if (len(images), len(content["annotations"])) != expected or any(row["iscrowd"] for row in content["annotations"]):
        raise ValueError(f"Unexpected test population in {dataset}")
    annotations = defaultdict(list)
    for row in content["annotations"]:
        annotations[row["image_id"]].append(row)
    if len(annotations) != len(images):
        raise ValueError(f"Missing annotated images in {dataset}")
    names = {row["id"]: row["name"] for row in content["categories"]}
    return content, images, annotations, names


def choose_public_cases(images, annotations, runs):
    ids = [image["id"] for image in images]
    strength = {image_id: sum(run[image_id].counts()[1] for run in runs[:3]) for image_id in ids}
    delta = {image_id: runs[7][image_id].counts()[1] - runs[6][image_id].counts()[1] for image_id in ids}
    rankings = [
        ("Kecocokan gabungan tertinggi", strength),
        ("Mask acuan paling sering terlewat", {i: len(annotations[i]) * 3 - strength[i] for i in ids}),
        ("Kecocokan bertambah pada COCO 800", delta),
        ("Kecocokan berkurang pada COCO 800", {i: -delta[i] for i in ids}),
    ]
    cases = []
    for title, scores in rankings:
        chosen = max((i for i in ids if i not in {c[0] for c in cases}), key=lambda i: (scores[i], -i))
        cases.append((chosen, title, scores[chosen]))
    if any(value <= 0 for _, _, value in cases):
        raise ValueError("Public examples no longer isolate their intended result.")
    return cases


def demo_visual(archive, images, names, annotations, control):
    with ZipFile(V5_RAW) as original:
        source = read(original, "test/_annotations.coco.json")
    dimensions = {row["file_name"]: (row["width"], row["height"]) for row in source["images"]}
    candidates = []
    for image in images:
        width, height = dimensions[image["source_name"]]
        result = control[image["id"]]
        loose, matched = result.counts()
        strict, strict_matched = result.counts(.30)
        if width != height and 3 <= loose - strict <= 7 and 6 <= loose <= 14 and strict >= 3:
            candidates.append((matched - strict_matched, strict_matched, -image["id"], image))
    image = max(candidates, key=lambda row: row[:3])[-1]
    uid = image["id"]
    control = control[uid]
    pixels = photo(archive, "test", image)
    with ZipFile(V5_RAW) as original, Image.open(BytesIO(original.read(f"test/{image['source_name']}"))) as raw:
        original_size = f"{raw.width} x {raw.height} piksel"
        original_view = ImageOps.pad(ImageOps.exif_transpose(raw).convert("RGB"), (640, 640), color="#f0f0f0")
    numbers = {}
    tiles = [
        ("Foto dari ekspor v5", original_size, original_view),
        ("Kanvas masukan", "640 x 640 dengan padding", pixels),
    ]
    for threshold in (.15, .30):
        masks, true = control.counts(threshold)
        numbers[f"{threshold:.2f}"] = {"predictions": masks, "matched": true}
        tiles.append((f"Skor >= {threshold:.2f}".replace(".", ","), f"{masks} mask / {true} cocok",
                      mask_overlay(pixels, control.visible(threshold))))
    grid(PUBLIC / "contoh-inferensi.jpg", [tiles], size=260)
    return {"image_id": uid, "reference_masks": len(annotations[uid]), **numbers}


def gamma_inputs(archive, images, phone_hash):
    experiment = load_json("output/test-pribadi/gamma-065-kasus-v3.json")
    if experiment["dataset_sha256"] != phone_hash:
        raise ValueError("The lighting experiment uses a different phone test snapshot.")
    photos = {PHOTO.search(row["source_name"]).group(): row for row in images}
    original_sizes = {
        Path(row["file"]).stem: (row["width"], row["height"])
        for row in load_json("output/test-pribadi/metadata-exif.json")
    }
    lut = [round(255 * (value / 255) ** experiment["gamma"]) for value in range(256)]
    control, brightened = [], []
    for ordinal, source in enumerate(experiment["photos"], 1):
        pixels = photo(archive, "private", photos[source])
        transformed = pixels.copy()
        width, height = original_sizes[source]
        scale = 640 / max(width, height)
        fit_w, fit_h = round(scale * width), round(scale * height)
        left, top = (640 - fit_w) // 2, (640 - fit_h) // 2
        interior = pixels.crop((left, top, left + fit_w, top + fit_h))
        transformed.paste(interior.point(lut * 3), (left, top))
        control.append((f"Foto redup {ordinal}", "Piksel kontrol", pixels))
        brightened.append((f"Foto redup {ordinal}", "Piksel gamma 0,65", transformed))
    grid(PDF_LOCAL.parent / "visual/gamma-input.jpg", [control, brightened], size=255)


def sweep(annotations, runs, common_ids):
    reference = sum(row["category_id"] in common_ids for rows in annotations.values() for row in rows)
    by_run = {}
    labels = (*LABELS, "Mask R-CNN v2 640", "Mask R-CNN v2 COCO 800")
    for label, results in zip(labels, (*runs[:3], *runs[6:]), strict=True):
        counts = []
        for threshold in THRESHOLDS:
            values = [result.counts(threshold, common_ids) for result in results.values()]
            counts.append({
                "threshold": threshold, "predictions": sum(mask for mask, _ in values),
                "matched": sum(true for _, true in values),
            })
        by_run[label] = counts
    return {
        "reference_masks": reference,
        "runs": by_run,
    }


def main():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PDF_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    public_hash, test_hash, phone_hash = map(hash_file, (V5, V5_TEST, PHONE))
    freeze = load_json("bukti/hasil/ringkasan.json")["source_sha256"]
    if (public_hash, test_hash, phone_hash) != (
        freeze["dataset_public_640"], freeze["dataset_test_public"], freeze["dataset_phones_v3"]
    ):
        raise ValueError("Dataset hashes disagree with the published evaluation.")
    with ZipFile(V5_TEST) as packed, ZipFile(V5) as v5:
        manifest = read(packed, "manifest.json")
        if (
            manifest["parent_archive_sha256"] != public_hash
            or packed.read("test/gt.json") != v5.read("test/gt.json")
            or read(v5, "manifest.json")["raw_sha256"] != hash_file(V5_RAW)
        ):
            raise ValueError("Test and original exports disagree with the frozen archive.")

    public_gt, public_images, public_gt_rows, public_names = source_rows(V5, "test", (359, 5344))
    phone_gt, phone_images, phone_gt_rows, phone_names = source_rows(PHONE, "private", (40, 619))
    if public_names != phone_names:
        raise ValueError("COCO category tables differ between domains.")
    public_runs, public_artifacts = collect_runs("test", public_gt, public_gt_rows, public_hash, test_hash)
    phone_runs, phone_artifacts = collect_runs("private", phone_gt, phone_gt_rows, phone_hash)
    phone_compare = load_json("output/improvement/phone-test.json")
    test_compare = load_json("output/improvement/public-test.json")
    if (public_artifacts[6:] != [test_compare[field]["artifact_sha256"]
                                for field in ("control", "improved_configuration")]
            or phone_artifacts[6:] != [phone_compare[field]
                                       for field in ("before_artifact_sha256", "after_artifact_sha256")]):
        raise ValueError("The additional v2 model archives differ from the report.")
    both = {row["category_id"] for row in public_gt["annotations"]} & {
        row["category_id"] for row in phone_gt["annotations"]
    }
    hard_sources = set(load_json("difficult-photos.json"))
    difficult = {image["id"] for image in phone_images if PHOTO.search(image["source_name"]).group() in hard_sources}
    if len(both) != 37 or len(difficult) != 12:
        raise ValueError("Unexpected class intersection or difficult photo count.")

    public_examples = choose_public_cases(public_images, public_gt_rows, public_runs)
    image_lookup = {row["id"]: row for row in public_images}
    canvas = Canvas(str(PDF_PUBLIC), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    canvas.setTitle("Lampiran mask test resmi v5")
    with ZipFile(V5) as archive:
        demo = demo_visual(archive, public_images, public_names, public_gt_rows, public_runs[6])
        for order, (uid, title, value) in enumerate(public_examples, 1):
            details = f"ID test {uid}   |   {len(public_gt_rows[uid])} mask acuan   |   Skor tampilan ≥ 0,15"
            draw_plate(canvas, title, details, f"V5-{order:02d}", photo(archive, "test", image_lookup[uid]),
                       public_gt_rows[uid], [run[uid] for run in public_runs], public_names, public=True)
    canvas.save()

    canvas = Canvas(str(PDF_LOCAL), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    canvas.setTitle("Lampiran mask 40 foto uji HP")
    phone_index(canvas, phone_images, phone_gt_rows, difficult)
    with ZipFile(PHONE) as archive:
        (PDF_LOCAL.parent / "visual").mkdir(exist_ok=True)
        gamma_inputs(archive, phone_images, phone_hash)
        for order, image in enumerate(phone_images, 1):
            uid = image["id"]
            condition = "Kondisi sulit" if uid in difficult else "Kondisi reguler"
            details = f"Foto {order:02d} dari 40   |   {condition}   |   {len(phone_gt_rows[uid])} mask acuan"
            serial = f"HP-{order:02d}"
            draw_plate(canvas, serial, details, serial, photo(archive, "private", image),
                       phone_gt_rows[uid], [run[uid] for run in phone_runs], phone_names, public=False)
    canvas.save()

    write_json(PUBLIC / "ringkasan-overlay.json", {
        "iou": MATCH_IOU,
        "display_confidence": DISPLAY_SCORE,
        "shared_classes": len(both),
        "test_v5": sweep(public_gt_rows, public_runs, both),
        "test_hp": sweep(phone_gt_rows, phone_runs, both),
        "cakupan_hp": [{"n": i, "sulit": image["id"] in difficult, "mask": len(phone_gt_rows[image["id"]])}
                       for i, image in enumerate(phone_images, 1)],
        "adegan_v5": [{"image_id": uid, "seleksi": title, "nilai_seleksi": score}
                      for uid, title, score in public_examples],
        "contoh_threshold": demo,
        "sha256": {
            "test_v5": public_hash, "test_v5_subset": test_hash, "test_hp_v3": phone_hash,
            "prediksi_v5": dict(zip((*FULL, *THREE, "rcnn-v2-640", "rcnn-v2-coco800"), public_artifacts)),
            "prediksi_hp": dict(zip((*FULL, *THREE, "rcnn-v2-640", "rcnn-v2-coco800"), phone_artifacts)),
        },
    })
    print("Lampiran selesai: 4 adegan v5, indeks dan 40 lembar foto HP.")


if __name__ == "__main__":
    main()
