"""
Performs two-stage inference on an image folder and saves YOLO-format labels.

Default behavior:
- Reads images from `temp/source_images`
- Uses `weights/best_localizer.pt` for stage 1 cropping
- Uses `weights/best_finetune.pt` for stage 2 stave detection
- Saves cropped images to `temp/images`
- Saves YOLO label files to `temp/labels`
- Builds a CVAT-importable `YOLO 1.1` package in `temp/yolo11_package`
- Writes `temp/yolo11_package.zip`

Example:
    uv run python scripts/inference_and_save.py --source temp --output-dir temp
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile

from PIL import Image, ImageOps
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-stage YOLO PT inference and save CVAT-compatible YOLO labels."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("temp"),
        help="Base directory containing a source_images subfolder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("temp"),
        help="Base directory where images and labels subfolders will be created.",
    )
    parser.add_argument(
        "--localizer-weights",
        type=Path,
        default=Path("weights/best_localizer.pt"),
        help="Stage 1 localizer .pt weights.",
    )
    parser.add_argument(
        "--counter-weights",
        type=Path,
        default=Path("weights/best_finetune.pt"),
        help="Stage 2 counter .pt weights.",
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.60, help="IoU threshold.")
    parser.add_argument("--device", default=None, help="Torch device, for example 0 or cpu.")
    parser.add_argument("--max-det", type=int, default=3000, help="Max detections per image.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Inference image size passed to Ultralytics.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference when supported by the selected device.",
    )
    parser.add_argument(
        "--class-name",
        default="stave",
        help="Single class name written into the YOLO 1.1 package metadata.",
    )
    return parser.parse_args()


def iter_image_paths(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Source images directory does not exist: {source}")
    return sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    image.load()
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def clamp_xyxy(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = min(max(float(x1), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    x2 = min(max(float(x2), 0.0), float(width))
    y2 = min(max(float(y2), 0.0), float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def select_localizer_crop(localizer_result, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if localizer_result.boxes is None or len(localizer_result.boxes) == 0:
        return 0.0, 0.0, float(image_width), float(image_height)

    conf = localizer_result.boxes.conf.cpu().numpy()
    xyxy = localizer_result.boxes.xyxy.cpu().numpy()
    best_idx = int(conf.argmax())
    return clamp_xyxy(xyxy[best_idx].tolist(), image_width, image_height)


def write_yolo_labels(
    output_path: Path,
    counter_result,
) -> int:
    if counter_result.boxes is None or len(counter_result.boxes) == 0:
        output_path.write_text("", encoding="utf-8")
        return 0

    xywhn = counter_result.boxes.xywhn.cpu().numpy()
    cls = counter_result.boxes.cls.cpu().numpy()
    lines: list[str] = []

    for cls_id, box in zip(cls, xywhn):
        center_x, center_y, width_n, height_n = [float(v) for v in box.tolist()]
        if width_n <= 0 or height_n <= 0:
            continue

        lines.append(
            f"{int(cls_id)} {center_x:.6f} {center_y:.6f} {width_n:.6f} {height_n:.6f}"
        )

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def build_yolo11_package(
    output_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    class_name: str,
) -> tuple[Path, Path]:
    package_dir = output_dir / "yolo11_package"
    package_data_dir = package_dir / "obj_train_data"
    zip_path = output_dir / "yolo11_package.zip"

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_data_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
    train_entries: list[str] = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, package_data_dir / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, package_data_dir / label_path.name)
        else:
            (package_data_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
        train_entries.append(f"data/obj_train_data/{image_path.name}")

    (package_dir / "obj.names").write_text(f"{class_name}\n", encoding="utf-8")
    (package_dir / "obj.data").write_text(
        "classes = 1\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n",
        encoding="utf-8",
    )
    (package_dir / "train.txt").write_text(
        "\n".join(train_entries) + ("\n" if train_entries else ""),
        encoding="utf-8",
    )

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package_dir))

    return package_dir, zip_path


def run_inference(args: argparse.Namespace) -> None:
    if not args.localizer_weights.exists():
        raise FileNotFoundError(f"Localizer weights not found: {args.localizer_weights}")
    if not args.counter_weights.exists():
        raise FileNotFoundError(f"Counter weights not found: {args.counter_weights}")

    source_images_dir = args.source / "source_images"
    output_images_dir = args.output_dir / "images"
    output_labels_dir = args.output_dir / "labels"

    image_paths = iter_image_paths(source_images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {source_images_dir}")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    localizer_model = YOLO(str(args.localizer_weights))
    counter_model = YOLO(str(args.counter_weights))

    for image_path in image_paths:
        image = load_image(image_path)
        image_width, image_height = image.size

        localizer_result = localizer_model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=max(args.max_det, 1),
            device=args.device,
            half=args.half,
            verbose=False,
        )[0]

        crop_box = select_localizer_crop(localizer_result, image_width, image_height)
        cropped_image = image.crop(crop_box)

        counter_result = counter_model.predict(
            source=cropped_image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device,
            half=args.half,
            verbose=False,
        )[0]

        cropped_output_path = output_images_dir / image_path.name
        cropped_image.save(cropped_output_path)

        label_output_path = output_labels_dir / f"{image_path.stem}.txt"
        box_count = write_yolo_labels(label_output_path, counter_result)
        print(
            f"{image_path.name}: saved crop -> {cropped_output_path}, "
            f"{box_count} boxes -> {label_output_path}"
        )

    package_dir, zip_path = build_yolo11_package(
        args.output_dir, output_images_dir, output_labels_dir, args.class_name
    )
    print(f"YOLO 1.1 package directory: {package_dir}")
    print(f"YOLO 1.1 zip: {zip_path}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
