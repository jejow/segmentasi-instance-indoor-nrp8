"""Shared COCO mask evaluation for every architecture."""
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score_masks(ground_truth, predictions, category_ids):
    truth = COCO()
    truth.dataset = ground_truth
    with redirect_stdout(StringIO()):
        truth.createIndex()
        if predictions:
            # COCO.loadRes enriches its input with NumPy area and bbox fields.
            detected = truth.loadRes([prediction.copy() for prediction in predictions])
        else:
            detected = COCO()
            detected.dataset = {"images": ground_truth["images"], "annotations": [],
                                "categories": ground_truth["categories"]}
            detected.createIndex()
        scorer = COCOeval(truth, detected, "segm")
        scorer.params.catIds = sorted(category_ids)
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()

    precision = scorer.eval["precision"]
    scores = precision[:, :, :, 0, -1]
    counts = Counter(a["category_id"] for a in ground_truth["annotations"] if not a["iscrowd"])
    names = {c["id"]: c["name"] for c in ground_truth["categories"]}

    def mean_score(data):
        present = data[data >= 0]
        return float(present.mean()) if present.size else None

    per_class = {}
    for index, category_id in enumerate(scorer.params.catIds):
        values = scores[:, :, index]
        per_class[names[category_id]] = {
            "ap": mean_score(values), "ap50": mean_score(values[0]),
            "ap75": mean_score(values[5]), "instances": counts[category_id],
        }
    return {
        "ap": mean_score(scores), "ap50": mean_score(scores[0]), "ap75": mean_score(scores[5]),
        **{
            f"ap_{label}": mean_score(precision[:, :, :, index, -1])
            for index, label in enumerate(scorer.params.areaRngLbl) if label != "all"
        },
        "images": len(ground_truth["images"]), "per_class": per_class,
    }


def encode_predictions(image_id, category_ids, scores, masks):
    if not (len(category_ids) == len(scores) == len(masks)):
        raise ValueError("A predicted mask must have a score and category.")
    results = []
    for category_id, score, mask in zip(category_ids, scores, masks, strict=True):
        if not np.any(mask):
            continue
        rle = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
        rle["counts"] = rle["counts"].decode("ascii")
        results.append({"image_id": image_id, "category_id": int(category_id),
                        "score": float(score), "segmentation": rle})
    return results
