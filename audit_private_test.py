"""Inventory a job export and render local mask review sheets."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps
from pycocotools import mask as coco_mask

PHOTO = re.compile(r"100086(?:579[1-9]|58[0-2][0-9]|583[01])")
COLORS = [(0, 165, 165), (237, 113, 48), (149, 106, 223), (233, 186, 38), (88, 134, 236), (75, 170, 80)]


def inspect(source, taxonomy, output):
    names = set(json.loads(taxonomy.read_text(encoding="utf-8"))["categories"])
    output.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    records, tiles = [], []
    with ZipFile(source) as archive:
        files = {name for name in archive.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))}
        seen = set()
        for item in archive.namelist():
            if not item.endswith("_annotations.coco.json"):
                continue
            data = json.loads(archive.read(item))
            cats = {c["id"]: c["name"] for c in data["categories"]}
            by_image = defaultdict(list)
            for ann in data["annotations"]:
                by_image[ann["image_id"]].append(ann)
            records_by_file = {row["file_name"]: row for row in data["images"]}
            split_files = {f for f in files if str(Path(f).parent).replace("\\","/") == str(Path(item).parent).replace("\\","/")}
            for member in sorted(split_files):
                name = Path(member).name
                camera_id = PHOTO.search(name)
                photo_id = camera_id.group(0) if camera_id else name
                seen.add(member)
                record = records_by_file.get(name, {})
                width, height = record.get("width"), record.get("height")
                counts, issues = Counter(), Counter()
                with Image.open(BytesIO(archive.read(member))) as image:
                    expected = (width, height)
                    if record and image.size != expected:
                        orientation = ImageOps.exif_transpose(image)
                        if orientation.size == expected:
                            image = orientation
                        else:
                            issues["dimension_mismatch"] += 1
                    tile = image.convert("RGB")
                tile.thumbnail((315, 300))
                overlay = Image.new("RGBA", tile.size)
                draw = ImageDraw.Draw(overlay)
                signatures = set()
                for ann in by_image.get(record.get("id"), []):
                    label = cats.get(ann["category_id"], "_unknown")
                    counts[label] += 1
                    if label not in names:
                        issues["outside_taxonomy"] += 1
                    seg = ann.get("segmentation")
                    if not isinstance(seg, (list, dict)) or not seg:
                        issues["missing_segmentation"] += 1
                        continue
                    signature = (label, json.dumps(seg, separators=(",",":")))
                    if signature in signatures:
                        issues["exact_duplicate"] += 1
                    signatures.add(signature)
                    color = COLORS[ann["category_id"] % len(COLORS)]
                    if isinstance(seg, dict):
                        if list(seg.get("size", ())) != [height, width] or "counts" not in seg:
                            issues["invalid_rle_size"] += 1
                            continue
                        try:
                            rle = coco_mask.frPyObjects(seg, height, width) if isinstance(seg["counts"], list) else seg
                            mask = Image.fromarray(coco_mask.decode(rle)).resize(tile.size, Image.Resampling.NEAREST)
                        except (TypeError, ValueError, IndexError):
                            issues["invalid_rle"] += 1
                            continue
                        if not mask.getbbox():
                            issues["empty_preview_mask"] += 1
                        overlay.paste((*color, 75), (0, 0), mask.convert("L").point(lambda value: 255 if value else 0))
                        continue
                    for polygon in seg:
                        if len(polygon) < 6 or len(polygon) % 2 or not all(math.isfinite(x) for x in polygon):
                            issues["invalid_polygon"] += 1
                            continue
                        points = list(zip(polygon[::2], polygon[1::2]))
                        if any(x < 0 or x > width or y < 0 or y > height for x,y in points):
                            issues["out_of_bounds"] += 1
                        scaled = [(x*tile.width/width, y*tile.height/height) for x,y in points]
                        draw.polygon(scaled, fill=(*color,48), outline=(*color,235))
                if not counts:
                    issues["no_masks"] += 1
                if not record:
                    issues["absent_from_coco_images"] += 1
                records.append({"id":photo_id,"file":name,"masks":counts.total(),
                                "classes":dict(counts),"flags":dict(issues)})
                review_tile = Image.new("RGB",(640,330),"white")
                review_tile.paste(tile, (0,0))
                review_tile.paste(Image.alpha_composite(tile.convert("RGBA"),overlay).convert("RGB"),(320,0))
                ImageDraw.Draw(review_tile).text((4,306), f"{photo_id}    {counts.total()} masks", fill="#202830")
                tiles.append((photo_id,review_tile))
        if files - seen:
            raise ValueError(f"{len(files-seen)} images are outside directories with COCO JSON.")
    for start in range(0,len(tiles),8):
        sheet = Image.new("RGB",(1280,1320),"#f1f4f7")
        for index, (_, image) in enumerate(sorted(tiles)[start:start+8]):
            sheet.paste(image,((index%2)*640,(index//2)*330))
        sheet.save(output/f"audit-{start//8+1:02d}.jpg", quality=88)
    summary = {"zip_sha256":digest,"photos":len(records),"with_masks":sum(bool(r["masks"]) for r in records),
               "masks":sum(r["masks"] for r in records),"records":sorted(records,key=lambda r:r["id"])}
    (output/"audit.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="records"}))
    counts = Counter()
    for row in records:
        counts.update(row["classes"])
    print("NRP 8 support:", {name: {"masks": counts[name], "photos": sum(name in r["classes"] for r in records)}
                             for name in ("knife", "light", "spoon")})
    print("Flagged:", [(r["id"],r["flags"]) for r in summary["records"] if r["flags"]])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source",type=Path)
    cli.add_argument("--taxonomy",type=Path,default=Path("data/panoptic-indoor-v5-640.json"))
    cli.add_argument("--output",type=Path,default=Path("output/test-pribadi/audit"))
    options=cli.parse_args()
    inspect(options.source,options.taxonomy,options.output)
