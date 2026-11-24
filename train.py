"""Train or evaluate one instance-segmentation model on the prepared COCO data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import hashlib
import math
import os
import random
from time import monotonic
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import torch
from torch.utils.data import DataLoader, Dataset

from metrics import encode_predictions, load_json, score_masks, write_json

try:
    from colab_progress import report
except ModuleNotFoundError:
    def report(**kwargs):
        pass

YOLO_WEIGHTS = {"yolov8n": "yolov8n-seg.pt", "yolo11n": "yolo11n-seg.pt"}
MODELS = (*YOLO_WEIGHTS, "maskrcnn", "maskrcnn-v2")
CONFIDENCE = 0.005
MAX_DETECTIONS = 100


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_dataset(archive, destination):
    with ZipFile(archive) as source:
        for member in source.infolist():
            path = (destination / member.filename).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError("Archive contains a path outside the dataset directory.")
            source.extract(member, destination)


def category_table(manifest, classes):
    names = manifest["categories"]
    if classes is None:
        classes = names
    if len(set(classes)) != len(classes) or any(name not in names for name in classes):
        raise ValueError("Class selection contains an unknown or duplicate class.")
    return {names.index(name) + 1: label for label, name in enumerate(classes)}


def sampled_yolo_images(root, categories, policy, seed, draws):
    """Keep every training image and add seed-fixed repeats at equal epoch size."""
    base = root / "train"
    gt = load_json(base / "gt.json")
    images = gt["images"]
    if draws < len(images):
        raise ValueError("Epoch size must preserve every original training image.")
    classes_by_image = defaultdict(set)
    for ann in gt["annotations"]:
        if ann["category_id"] in categories and not ann["iscrowd"]:
            classes_by_image[ann["image_id"]].add(ann["category_id"])
    frequency = Counter(category for labels in classes_by_image.values() for category in labels)
    weights = []
    for image in images:
        labels = classes_by_image[image["id"]]
        if policy == "repeat-factor":
            weight = max((math.sqrt(.05 * len(images) / frequency[c]) for c in labels), default=1)
            weights.append(max(1, weight))
        elif policy == "foreground":
            weights.append(int(bool(labels)))
        else:
            weights.append(1)
    if not sum(weights):
        raise ValueError("Sampling found no foreground images.")
    extra = random.Random(seed).choices(images, weights=weights, k=draws - len(images))
    combined = images + extra
    sampled = root / "sampled"
    for part in ("images", "labels"):
        (sampled / part).mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(combined):
        original = Path(image["file_name"])
        new_name = f"{index:05d}{original.suffix}"
        os.link(base / "images" / original, sampled / "images" / new_name)
        os.link(base / "labels" / original.with_suffix(".txt"),
                sampled / "labels" / Path(new_name).with_suffix(".txt"))
    positives = sum(bool(classes_by_image[image["id"]]) for image in combined)
    print(f"Sampling {policy}: {len(images)} unique images, {len(combined)} records, "
          f"{positives} records with foreground.", flush=True)
    return "sampled/images"


def write_yolo_labels(root, categories, manifest, args):
    from ultralytics.data.converter import merge_multi_segment
    import yaml

    for split in ("train", "valid"):
        base = root / split
        labels = base / "labels"
        labels.mkdir()
        dataset = load_json(base / "gt.json")
        by_image = defaultdict(list)
        size = manifest["size"]
        for ann in dataset["annotations"]:
            if ann["category_id"] not in categories or ann["iscrowd"]:
                continue
            polygons = ann["polygons"]
            if not polygons:
                raise ValueError("One training annotation cannot be mapped to a YOLO polygon.")
            if len(polygons) > 1:
                points = np.concatenate(merge_multi_segment(polygons)).reshape(-1) / size
            else:
                points = np.asarray(polygons[0], dtype=np.float32) / size
            line = " ".join(f"{value:.7f}" for value in np.clip(points, 0, 1))
            by_image[ann["image_id"]].append(f'{categories[ann["category_id"]]} {line}')
        for image in dataset["images"]:
            rows = by_image[image["id"]]
            (labels / Path(image["file_name"]).with_suffix(".txt")).write_text(
                "".join(row + "\n" for row in rows), encoding="utf-8"
            )
    policy = getattr(args, "sampling", "native")
    train_images = "train/images"
    if policy != "native":
        train_images = sampled_yolo_images(root, categories, policy, args.seed, args.epoch_images)
    yaml_file = root / "segmentation.yaml"
    yaml_file.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": train_images, "val": "valid/images",
        "names": {label: manifest["categories"][id_ - 1] for id_, label in categories.items()},
    }), encoding="utf-8")
    return yaml_file


def train_yolo(args, root, classes, manifest, work):
    from ultralytics import YOLO, settings

    settings.update({"wandb": False, "mlflow": False, "tensorboard": False})
    model = YOLO(YOLO_WEIGHTS[args.model])
    data = write_yolo_labels(root, classes, manifest, args)
    deadline = monotonic() + 60 * args.minutes

    def on_batch(trainer):
        steps = len(trainer.train_loader)
        index = batch_count["value"] % steps + 1
        batch_count["value"] += 1
        if index == 1 or index % 25 == 0:
            report(phase="training", epoch=trainer.epoch+1, batch=index, batches=steps)

    def on_epoch(trainer):
        report(phase="validating", epoch=trainer.epoch + 1, batch=0)
        epoch_seen["value"] = trainer.epoch + 1

    def stop_at_budget(trainer):
        if monotonic() >= deadline:
            trainer.stop = True
            print("Training time budget reached.", flush=True)

    batch_count = {"value": 0}
    epoch_seen = {"value": 0}
    model.add_callback("on_train_batch_end", on_batch)
    model.add_callback("on_train_epoch_end", on_epoch)
    model.add_callback("on_fit_epoch_end", stop_at_budget)
    lighting = getattr(args, "lighting_augmentation", False)
    model.train(
        data=str(data), task="segment", imgsz=manifest["size"], epochs=args.epochs,
        batch=args.yolo_batch, workers=args.workers, seed=args.seed, device=args.device,
        project=str(work), name="ultralytics", save=True, exist_ok=False, plots=False,
        patience=0, optimizer="AdamW", lr0=.001, lrf=.1,
        cos_lr=True, nbs=args.yolo_batch, weight_decay=.0005,
        hsv_h=0, hsv_s=.15 if lighting else 0, hsv_v=.25 if lighting else 0, degrees=0, translate=0, scale=0,
        shear=0, perspective=0, fliplr=.5, flipud=0, mosaic=0, mixup=0,
        copy_paste=0, close_mosaic=0, overlap_mask=False, deterministic=False,
        mask_ratio=getattr(args, "mask_ratio", 4),
    )
    run = work / "ultralytics"
    checkpoint = run / "weights" / "last.pt"
    if not checkpoint.is_file() or epoch_seen["value"] < 1:
        raise RuntimeError("YOLO did not produce a trained checkpoint.")
    return checkpoint, run / "results.csv", epoch_seen["value"]


def open_image(path):
    with Image.open(path) as image:
        data = np.array(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(data).permute(2, 0, 1).contiguous()


class COCOMasks(Dataset):
    def __init__(self, split_dir, data, category_table):
        self.directory = split_dir / "images"
        self.images = data["images"]
        self.classes = category_table
        self.anns = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] in self.classes and not ann["iscrowd"]:
                self.anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        pixels = open_image(self.directory / image["file_name"])
        anns = self.anns[image["id"]]
        boxes = torch.tensor(
            [[x, y, x+w, y+h] for x, y, w, h in (a["bbox"] for a in anns)], dtype=torch.float32
        ).reshape(-1, 4)
        labels = torch.tensor([self.classes[a["category_id"]]+1 for a in anns], dtype=torch.int64)
        if anns:
            array = coco_mask.decode([a["segmentation"] for a in anns])
            masks = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        else:
            masks = torch.zeros((0, image["height"], image["width"]), dtype=torch.uint8)
        if torch.rand(()) < .5:
            pixels = pixels.flip(-1)
            masks = masks.flip(-1)
            boxes[:, [0, 2]] = image["width"] - boxes[:, [2, 0]]
        return pixels, {"boxes": boxes, "labels": labels, "masks": masks}


def collate_samples(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def transfer_coco_predictors(source_box, source_mask, target_box, target_mask,
                             source_names, target_names):
    source_ids = {name: label for label, name in enumerate(source_names) if label and name != "N/A"}
    matches = [
        (label, source_ids[name]) for label, name in enumerate(target_names, start=1)
        if name in source_ids
    ]
    target_mask.conv5_mask.load_state_dict(source_mask.conv5_mask.state_dict())
    for target, source in [(0, 0), *matches]:
        target_box.cls_score.weight[target].copy_(source_box.cls_score.weight[source])
        target_box.cls_score.bias[target].copy_(source_box.cls_score.bias[source])
        target_slice = slice(4 * target, 4 * (target + 1))
        source_slice = slice(4 * source, 4 * (source + 1))
        target_box.bbox_pred.weight[target_slice].copy_(source_box.bbox_pred.weight[source_slice])
        target_box.bbox_pred.bias[target_slice].copy_(source_box.bbox_pred.bias[source_slice])
        target_mask.mask_fcn_logits.weight[target].copy_(source_mask.mask_fcn_logits.weight[source])
        target_mask.mask_fcn_logits.bias[target].copy_(source_mask.mask_fcn_logits.bias[source])
    return [target_names[label - 1] for label, _ in matches]


def build_maskrcnn(size, number_classes, pretrained, variant="maskrcnn", transfer_names=None):
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    builder, weights = {
        "maskrcnn": (maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights.DEFAULT),
        "maskrcnn-v2": (maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    }[variant]
    model = builder(
        weights=weights if pretrained else None,
        weights_backbone=None, min_size=size, max_size=size,
        box_score_thresh=CONFIDENCE, box_detections_per_img=MAX_DETECTIONS,
    )
    source_box = model.roi_heads.box_predictor
    source_mask = model.roi_heads.mask_predictor
    box = FastRCNNPredictor(source_box.cls_score.in_features, number_classes + 1)
    mask = MaskRCNNPredictor(source_mask.conv5_mask.in_channels, 256, number_classes + 1)
    if transfer_names is not None:
        if not pretrained or len(transfer_names) != number_classes:
            raise ValueError("Transferring category weights requires COCO weights and ordered class names.")
        matched = transfer_coco_predictors(
            source_box, source_mask, box, mask, weights.meta["categories"], transfer_names,
        )
        print(f"Transferred COCO predictor weights for {len(matched)} classes: {', '.join(matched)}",
              flush=True)
    model.roi_heads.box_predictor = box
    model.roi_heads.mask_predictor = mask
    return model


def amp_context(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def train_maskrcnn(args, root, classes, manifest, work):
    size = getattr(args, "rcnn_size", None) or manifest["size"]
    device = torch.device("cuda:0" if args.device != "cpu" else "cpu")
    train = root / "train"
    loader = DataLoader(
        COCOMasks(train, load_json(train / "gt.json"), classes), batch_size=args.rcnn_batch,
        num_workers=args.workers, shuffle=True, collate_fn=collate_samples,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    transfer_names = None
    if getattr(args, "rcnn_head_init", "random") == "coco-overlap":
        transfer_names = [manifest["categories"][category_id - 1] for category_id in classes]
    model = build_maskrcnn(size, len(classes), pretrained=True, variant=args.model,
                           transfer_names=transfer_names).to(device)
    roi_count = 0

    def track_roi_count(_module, _inputs, outputs):
        nonlocal roi_count
        roi_count = outputs[0].shape[0]

    model.roi_heads.box_predictor.register_forward_hook(track_roi_count)

    def forward_losses(images, targets):
        losses = model(images, targets)
        if roi_count == 0:
            # TorchVision's box regression normalizes by the count of sampled RoIs.
            # An empty proposal set can still provide valid RPN supervision.
            losses.pop("loss_classifier")
            losses.pop("loss_box_reg")
        return losses

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=3e-4, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = work / "last.pt"
    csv_path = work / "results.csv"
    deadline = monotonic() + 60 * args.minutes
    completed = 0
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        columns = ["epoch", "batches", "train_loss", "lr", "epoch_seconds", "fp32_retry_batches", "empty_roi_batches"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            model.train()
            start = monotonic()
            losses = []
            fp32_retries = 0
            empty_roi_batches = 0
            for step, (images, targets) in enumerate(loader, start=1):
                if step == 1 or step % 25 == 0:
                    report(phase="training", epoch=epoch, batch=step, batches=len(loader))
                images = [image.to(device, non_blocking=True).float().div_(255) for image in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device):
                    outputs = forward_losses(images, targets)
                    loss = sum(outputs.values())
                if not torch.isfinite(loss) and device.type == "cuda":
                    del loss, outputs
                    outputs = forward_losses(images, targets)
                    loss = sum(outputs.values())
                    fp32_retries += 1
                    print(f"Retried Mask R-CNN forward in FP32: epoch {epoch}, step {step}", flush=True)
                if not torch.isfinite(loss):
                    invalid = [name for name, value in outputs.items() if not torch.isfinite(value)]
                    raise FloatingPointError(f"Non-finite losses {invalid}, epoch {epoch}, step {step}, {roi_count} sampled RoIs")
                empty_roi_batches += int(roi_count == 0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.detach())
                if monotonic() >= deadline:
                    break
            loss_value = torch.stack(losses).mean().item()
            row = {"epoch": epoch, "batches": step, "train_loss": loss_value,
                   "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": round(monotonic()-start, 3),
                   "fp32_retry_batches": fp32_retries, "empty_roi_batches": empty_roi_batches}
            writer.writerow(row)
            stream.flush()
            print(row, flush=True)
            if step == len(loader):
                completed = epoch
                scheduler.step()
                state = {
                    k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                    for k, v in model.state_dict().items()
                }
                temporary = checkpoint.with_suffix(".part")
                torch.save(state, temporary)
                temporary.replace(checkpoint)
            if monotonic() >= deadline:
                print("Training time budget reached.", flush=True)
                break
    if completed == 0:
        raise RuntimeError("Mask R-CNN did not finish any epoch within the time budget.")
    return checkpoint, csv_path, completed


def predict_yolo(args, directory, gt, checkpoint, classes, size):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    reverse = {label: id_ for id_, label in classes.items()}
    id_by_file = {image["file_name"]: image["id"] for image in gt["images"]}
    source = str(directory)
    results = model.predict(
        source, stream=True, imgsz=size, batch=4, retina_masks=True, verbose=False,
        max_det=MAX_DETECTIONS, conf=CONFIDENCE, device=args.device,
    )
    for result in results:
        if result.masks is None:
            continue
        category_ids = [reverse[int(label)] for label in result.boxes.cls.tolist()]
        masks = (result.masks.data > .5).cpu().numpy()
        yield from encode_predictions(id_by_file[Path(result.path).name], category_ids,
                                      result.boxes.conf.tolist(), masks)


@torch.inference_mode()
def predict_maskrcnn(args, directory, gt, checkpoint, classes, size):
    device = torch.device("cuda:0" if args.device != "cpu" else "cpu")
    model = build_maskrcnn(getattr(args, "rcnn_size", None) or size, len(classes),
                           pretrained=False, variant=args.model)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    reverse = {label + 1: id_ for id_, label in classes.items()}
    for index, image in enumerate(gt["images"], start=1):
        pixels = open_image(directory / image["file_name"]).to(device).float().div_(255)
        with amp_context(device):
            result = model([pixels])[0]
        labels = [reverse[int(label)] for label in result["labels"].tolist()]
        masks = (result["masks"][:, 0] >= .5).cpu().numpy()
        yield from encode_predictions(image["id"], labels, result["scores"].tolist(), masks)
        if index % 25 == 0:
            print(f"Inference: {index}/{len(gt['images'])} images", flush=True)


def evaluate_split(args, root, split, checkpoint, classes, manifest, output):
    gt = load_json(root / split / "gt.json")
    predict = predict_yolo if args.model in YOLO_WEIGHTS else predict_maskrcnn
    detections = list(predict(args, root / split / "images", gt, checkpoint, classes, manifest["size"]))
    result = score_masks(gt, detections, classes.keys())
    if output is not None:
        write_json(output / f"predictions-{split}.json", detections)
    print(f"{split}: AP={result['ap']}, AP50={result['ap50']}, AP75={result['ap75']}, "
          f"AP small={result['ap_small']}", flush=True)
    return result


def run(args):
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU is unavailable from PyTorch.")
    started_at = now()
    seed_everything(args.seed)
    report(phase="dataset")
    with TemporaryDirectory(prefix="indoor-nrp8-") as temp:
        work = Path(temp)
        root = work / "data"
        extract_dataset(args.dataset, root)
        manifest = load_json(root / "manifest.json")
        class_list = load_json(args.classes)["classes"] if args.classes else None
        classes = category_table(manifest, class_list)
        if "private" in manifest["splits"] and args.checkpoint is None:
            raise ValueError("The external test archive is for evaluation only.")
        model_gpu = torch.cuda.get_device_name(0) if args.device != "cpu" else "CPU"
        print(f"{started_at} | {args.model}, {len(classes)} classes, {model_gpu}", flush=True)
        training_start = monotonic()
        if args.checkpoint is None:
            train = train_yolo if args.model in YOLO_WEIGHTS else train_maskrcnn
            checkpoint, results, actual_epochs = train(args, root, classes, manifest, work)
        else:
            checkpoint, results, actual_epochs = args.checkpoint, None, None
        training_seconds = monotonic() - training_start if results else None

        evals = {}
        predictions_directory = None if args.pilot else work
        report(phase="validating", epoch=actual_epochs)
        for split in args.eval_splits:
            evals[split] = evaluate_split(args, root, split, checkpoint, classes, manifest, predictions_directory)
        if actual_epochs and "valid" in evals and evals["valid"]["ap"] is not None:
            report(epoch=actual_epochs, value=100 * evals["valid"]["ap"], seconds=training_seconds,
                   metrics={"AP50": 100 * evals["valid"]["ap50"], "AP75": 100 * evals["valid"]["ap75"]})

        with args.dataset.open("rb") as stream:
            dataset_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        summary = {
            "model": args.model, "categories": classes,
            "class_names": [manifest["categories"][key - 1] for key in classes],
            "pilot": args.pilot, "seed": args.seed, "input_size": manifest["size"],
            "gpu": model_gpu, "torch": torch.__version__, "started_at_utc": started_at,
            "finished_at_utc": now(), "target_epochs": args.epochs if results else None,
            "completed_epochs": actual_epochs, "training_seconds": training_seconds,
            "total_seconds": monotonic() - training_start, "dataset_sha256": dataset_sha,
            "parent_dataset_sha256": manifest.get("parent_archive_sha256", dataset_sha),
            "confidence": CONFIDENCE, "max_detections": MAX_DETECTIONS, "scores": evals,
            "rcnn_size": (
                getattr(args, "rcnn_size", None) or manifest["size"]
                if args.model not in YOLO_WEIGHTS else None
            ),
            "rcnn_head_init": (
                getattr(args, "rcnn_head_init", "random")
                if args.model not in YOLO_WEIGHTS and results else None
            ),
            "sampling": getattr(args, "sampling", "native"),
            "epoch_images": getattr(args, "epoch_images", None) if getattr(args, "sampling", "native") != "native" else None,
            "mask_ratio": getattr(args, "mask_ratio", 4) if args.model in YOLO_WEIGHTS else None,
            "lighting_augmentation": getattr(args, "lighting_augmentation", False),
        }
        summary_path = work / "summary.json"
        write_json(summary_path, summary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_suffix(".part")
        with ZipFile(temporary_output, "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
            archive.write(summary_path, "summary.json")
            if results:
                archive.write(results, "results.csv")
                if not args.pilot:
                    archive.write(checkpoint, "checkpoint.pt", compress_type=ZIP_STORED)
            for file in sorted(work.glob("predictions-*.json")):
                archive.write(file, file.name)
        temporary_output.replace(args.output)
        print(f"Saved evaluation artifact: {args.output}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, choices=MODELS)
    cli.add_argument("--dataset", type=Path, default=Path("data/panoptic-indoor-v5-640.zip"))
    cli.add_argument("--classes", type=Path)
    cli.add_argument("--output", type=Path, default=Path("output/result.zip"))
    cli.add_argument("--epochs", type=int, default=16)
    cli.add_argument("--minutes", type=float, default=45)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--workers", type=int, default=2)
    cli.add_argument("--yolo-batch", type=int, default=8)
    cli.add_argument("--rcnn-batch", type=int, default=2)
    cli.add_argument("--rcnn-size", type=int, help="Internal Mask R-CNN resize, keeping the evaluation grid.")
    cli.add_argument("--rcnn-head-init", choices=["random", "coco-overlap"], default="random")
    cli.add_argument("--device", choices=["0", "cpu"], default="0")
    cli.add_argument("--pilot", action="store_true")
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--eval-splits", nargs="+", default=["valid", "test"])
    cli.add_argument("--sampling", choices=["native", "uniform", "repeat-factor", "foreground"], default="native")
    cli.add_argument("--epoch-images", type=int, default=640)
    cli.add_argument("--mask-ratio", type=int, choices=[2, 4], default=4)
    cli.add_argument("--lighting-augmentation", action="store_true")
    options = cli.parse_args()
    if options.sampling != "native" and options.model not in YOLO_WEIGHTS:
        cli.error("--sampling is supported by the YOLO training path.")
    if options.model in YOLO_WEIGHTS and (options.rcnn_size or options.rcnn_head_init != "random"):
        cli.error("--rcnn-size and --rcnn-head-init apply to Mask R-CNN.")
    if options.rcnn_size is not None and (options.rcnn_size < 64 or options.rcnn_size % 32):
        cli.error("--rcnn-size must be a multiple of 32 and at least 64.")
    if options.checkpoint and options.rcnn_head_init != "random":
        cli.error("--rcnn-head-init applies to training, not a saved checkpoint.")
    run(options)
