"""
Runs the two-stage ONNX pipeline on sample images twice:
1. regular post-processing
2. density-filtered post-processing

The script writes side-by-side comparison images into `temp_outs/`.

Example:
    uv run python scripts/test_density_filter_side_by_side.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from yoloPostprocessUtils import crop_pillow_img_from_bbox, extract_highest_conf_bbox, convert_cornerWidthHeight_to_cornerCords
from yolo_onnx import YOLO_OnnxRuntime


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stave counts with and without density-based post-processing."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("sample_data"),
        help="Directory containing sample images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("temp_outs"),
        help="Directory where side-by-side comparison images will be written.",
    )
    parser.add_argument(
        "--localizer-weights",
        type=Path,
        default=None,
        help="ONNX localizer weights. Defaults to the latest matching file under weights/.",
    )
    parser.add_argument(
        "--counter-weights",
        type=Path,
        default=None,
        help="ONNX counter weights. Defaults to the latest matching file under weights/.",
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.60, help="IoU threshold.")
    return parser.parse_args()


def iter_images(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    return sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    image.load()
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def find_latest_onnx(directory: str, mode: str) -> str:
    pattern = Path(directory).glob(f"staves_detector_{mode}_onnx_v*.onnx")
    matches = list(pattern)
    if not matches:
        raise FileNotFoundError(f"No ONNX model found for mode '{mode}' in '{directory}'")

    def version_num(path: Path) -> int:
        stem = path.name
        parts = stem.split("_v")
        if len(parts) < 2:
            return -1
        version_part = parts[-1].removesuffix(".onnx")
        return int(version_part) if version_part.isdigit() else -1

    return str(max(matches, key=version_num))


def select_crop(image: Image.Image, localizer_path: str, conf: float, iou: float) -> Image.Image:
    localizer_results = YOLO_OnnxRuntime(
        localizer_path,
        image,
        confidence_thres=conf,
        iou_thres=iou,
    )
    _, boxes, scores, _ = localizer_results.main()
    top_box, _ = extract_highest_conf_bbox(boxes, scores)
    if top_box is None:
        return image
    crop_box = convert_cornerWidthHeight_to_cornerCords(top_box.tolist())
    return crop_pillow_img_from_bbox(crop_box, image)


def run_counter(
    cropped_image: Image.Image,
    counter_path: str,
    conf: float,
    iou: float,
    density_filter: bool,
) -> tuple[Image.Image, int]:
    counter_results = YOLO_OnnxRuntime(
        counter_path,
        cropped_image,
        confidence_thres=conf,
        iou_thres=iou,
        enable_density_filter=density_filter,
    )
    annotated_arr, boxes, _, _ = counter_results.main()
    return Image.fromarray(annotated_arr[:, :, ::-1].copy()), len(boxes)


def make_comparison(
    source_name: str,
    left_img: Image.Image,
    right_img: Image.Image,
    left_title: str,
    right_title: str,
) -> Image.Image:
    left_img = left_img.convert("RGB")
    right_img = right_img.convert("RGB")

    panel_width = max(left_img.width, right_img.width)
    panel_height = max(left_img.height, right_img.height)
    header_height = 56
    padding = 18
    gap = 18

    canvas_width = padding * 2 + panel_width * 2 + gap
    canvas_height = padding * 2 + header_height + panel_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((padding, 10), source_name, fill=(230, 237, 243), font=font)
    draw.rounded_rectangle(
        (padding, 28, canvas_width - padding, 28 + header_height - 10),
        radius=12,
        outline=(48, 54, 61),
        width=1,
        fill=(22, 27, 34),
    )
    draw.text((padding + 12, 40), left_title, fill=(230, 237, 243), font=font)
    draw.text((padding + panel_width + gap + 12, 40), right_title, fill=(230, 237, 243), font=font)

    left_x = padding
    right_x = padding + panel_width + gap
    image_y = padding + header_height

    canvas.paste(left_img, (left_x, image_y))
    canvas.paste(right_img, (right_x, image_y))

    draw.rounded_rectangle(
        (left_x, image_y, left_x + panel_width, image_y + panel_height),
        radius=12,
        outline=(88, 166, 255),
        width=1,
    )
    draw.rounded_rectangle(
        (right_x, image_y, right_x + panel_width, image_y + panel_height),
        radius=12,
        outline=(63, 185, 80),
        width=1,
    )

    return canvas


def main() -> None:
    args = parse_args()

    localizer_path = str(args.localizer_weights or find_latest_onnx("weights", "localizer"))
    counter_path = str(args.counter_weights or find_latest_onnx("weights", "finetune"))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(args.source)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.source}")

    for image_path in image_paths:
        image = load_image(image_path)
        cropped_image = select_crop(image, localizer_path, args.conf, args.iou)

        plain_img, plain_count = run_counter(
            cropped_image,
            counter_path,
            args.conf,
            args.iou,
            density_filter=False,
        )
        filtered_img, filtered_count = run_counter(
            cropped_image,
            counter_path,
            args.conf,
            args.iou,
            density_filter=True,
        )

        comparison = make_comparison(
            image_path.name,
            plain_img,
            filtered_img,
            f"Without density filter: {plain_count} stave(s)",
            f"With density filter: {filtered_count} stave(s)",
        )

        output_path = output_dir / f"{image_path.stem}_comparison.jpg"
        comparison.save(output_path, quality=95)
        print(f"{image_path.name}: saved {output_path}")


if __name__ == "__main__":
    main()
