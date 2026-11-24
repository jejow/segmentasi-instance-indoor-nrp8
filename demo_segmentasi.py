"""Segment one newly captured photo with the selected Mask R-CNN checkpoint."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageOps
from pycocotools import mask as coco_mask
import torch

from metrics import load_json
from prepare_dataset import letterbox
from train import category_table, predict_maskrcnn


COLORS = ("#2ca48b", "#e79153", "#528ad5", "#bd6caa", "#b6a12a", "#9a7be3")


def infer(photo, checkpoint, taxonomy, variant, threshold, output, device, rcnn_size=None):
    manifest = load_json(taxonomy)
    classes = category_table(manifest, None)
    size = manifest["size"]
    with Image.open(photo) as raw:
        pixels, _ = letterbox(ImageOps.exif_transpose(raw).convert("RGB"), size)
    with TemporaryDirectory(prefix="demo-indoor-") as tmp:
        directory = Path(tmp)
        name = "test.jpg"
        pixels.save(directory / name, quality=89, optimize=True)
        records = {"images": [{"id": 1, "file_name": name}]}
        args = argparse.Namespace(model=variant, device=device, rcnn_size=rcnn_size)
        predictions = [
            row for row in predict_maskrcnn(args, directory, records, checkpoint, classes, size)
            if row["score"] >= threshold
        ]
    canvas = pixels.copy()
    labeled = []
    for index, row in enumerate(sorted(predictions, key=lambda r: r["score"])):
        color = COLORS[index % len(COLORS)]
        mask = Image.fromarray(coco_mask.decode(row["segmentation"])).convert("L")
        canvas.paste(color, (0, 0), mask.point(lambda value: 105 if value else 0))
        label = manifest["categories"][row["category_id"] - 1]
        x, y, _, _ = coco_mask.toBbox(row["segmentation"])
        labeled.append((int(x), int(y), f"{label} {row['score']:.2f}", color))
    draw = ImageDraw.Draw(canvas)
    for x, y, caption, color in labeled:
        width = round(draw.textlength(caption)) + 10
        x, y = min(x, size - width - 1), min(y, size - 17)
        draw.rectangle((x, y, x + width, y + 16), fill="#182B3D", outline=color)
        draw.text((x + 4, y + 2), caption, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"{len(predictions)} mask pada confidence >= {threshold:.3f}. {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("output/research/maskrcnn-v2-97/checkpoint.pt"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/panoptic-indoor-v5-640.json"))
    parser.add_argument("--variant", choices=["maskrcnn", "maskrcnn-v2"], default="maskrcnn-v2")
    parser.add_argument("--rcnn-size", type=int, help="Internal resize used to evaluate the checkpoint.")
    parser.add_argument("--threshold", type=float, default=.15)
    parser.add_argument("--device", choices=["cpu", "0"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("output/demo/mask.png"))
    options = parser.parse_args()
    if not 0.005 <= options.threshold <= 1:
        parser.error("Use a threshold between 0.005 and 1.")
    if options.rcnn_size is not None and (options.rcnn_size < 64 or options.rcnn_size % 32):
        parser.error("--rcnn-size must be a multiple of 32 and at least 64.")
    torch.set_num_threads(3)
    infer(options.photo, options.checkpoint, options.taxonomy, options.variant,
          options.threshold, options.output, options.device, options.rcnn_size)
