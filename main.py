"""Train, evaluate, and run the supplied blood-cell detector."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "blood_cell_dataset/data.yaml"
DEFAULT_MODEL = ROOT / "blood_cell_experiments/balanced_loss/weights/best.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_dataset(path):
    """Resolve dataset paths relative to the YAML file, on any machine."""
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = Path(config.get("path", "."))
    if not base.is_absolute():
        base = path.parent / base
    config["path"] = str(base.resolve())
    for split in ("train", "val", "test"):
        if split in config:
            config[split] = str((base / config[split]).resolve())
    return config


def check_dataset(path):
    """Verify image decoding, matching annotations, and YOLO box values."""
    import cv2

    config = read_dataset(path)
    names = config["names"]
    if config.get("nc", len(names)) != len(names):
        raise ValueError("nc does not match the class names")
    summary = {}
    for split in ("train", "val", "test"):
        image_dir = Path(config[split])
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No images in {image_dir}")
        counts = Counter()
        label_dir = image_dir.parent / "labels"
        image_stems = {p.stem for p in images}
        orphaned = [p.name for p in label_dir.glob("*.txt") if p.stem not in image_stems]
        if orphaned:
            raise ValueError(f"Labels without images in {label_dir}: {orphaned[:3]}")
        for image in images:
            if cv2.imread(str(image)) is None:
                raise ValueError(f"Cannot decode image: {image}")
            label = label_dir / f"{image.stem}.txt"
            for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                values = [float(v) for v in line.split()]
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Invalid annotation: {label}:{number}")
                class_id, x, y, width, height = values
                if (not class_id.is_integer() or not 0 <= class_id < len(names)
                        or not 0 <= x <= 1 or not 0 <= y <= 1
                        or not 0 < width <= 1 or not 0 < height <= 1):
                    raise ValueError(f"Invalid class or box: {label}:{number}")
                counts[names[int(class_id)]] += 1
        summary[split] = {"images": len(images), "annotations": dict(counts)}
    return summary


def count_cells(result):
    counts = {name: 0 for name in result.names.values()}
    for class_id in result.boxes.cls.tolist():
        counts[result.names[int(class_id)]] += 1
    return counts


def resolved_yaml(data, output):
    """Write a local runtime config for Ultralytics without changing the source."""
    import yaml

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "data.resolved.yaml"
    target.write_text(yaml.safe_dump(read_dataset(data), sort_keys=False), encoding="utf-8")
    return str(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Verify the complete dataset")
    check.add_argument("--data", type=Path, default=DEFAULT_DATA)
    for command in ("predict", "val", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        sub.add_argument("--device", default="cpu", help="cpu, 0, or another PyTorch device")
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--output", type=Path, default=ROOT / "outputs" / command)
        if command == "predict":
            sub.add_argument("--source", type=Path, required=True, help="Image or image directory")
            sub.add_argument("--conf", type=float, default=0.25)
        else:
            sub.add_argument("--data", type=Path, default=DEFAULT_DATA)
            sub.add_argument("--batch", type=int, default=8)
            if command == "val":
                sub.add_argument("--split", choices=("val", "test"), default="test")
            else:
                sub.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps(check_dataset(args.data), indent=2))
        return
    if not args.model.is_file():
        parser.error(f"Model does not exist: {args.model}")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.command == "predict":
        if not args.source.exists():
            parser.error(f"Source does not exist: {args.source}")
        if not 0 <= args.conf <= 1:
            parser.error("--conf must be between 0 and 1")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    output = args.output.resolve()
    options = dict(device=args.device, imgsz=args.imgsz,
                   project=str(output.parent), name=output.name, exist_ok=True)
    if args.command == "predict":
        records = []
        for result in model.predict(source=str(args.source), conf=args.conf,
                                    save=True, stream=True, **options):
            records.append({"image": str(result.path), "counts": count_cells(result)})
        output.mkdir(parents=True, exist_ok=True)
        (output / "counts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(records, indent=2))
    else:
        data = resolved_yaml(args.data, output)
        if args.command == "val":
            metrics = model.val(data=data, split=args.split, batch=args.batch, workers=0, **options)
            (output / "metrics.json").write_text(json.dumps(metrics.results_dict, indent=2), encoding="utf-8")
        else:
            model.train(data=data, epochs=args.epochs, batch=args.batch, workers=0, **options)


if __name__ == "__main__":
    main()
