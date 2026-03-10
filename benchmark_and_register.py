"""
Benchmarks a trained Ultralytics YOLO model in both PyTorch and ONNX runtimes,
logs all metrics and artifacts to MLflow on Dagshub, and registers the ONNX
model to the MLflow Model Registry for manual promotion review.

Usage:
    uv run python benchmark_and_register.py --model yolov8n --mode finetune

Args:
    --model : YOLO model stem name, e.g. "yolov8n", "yolo11m"
    --mode  : one of "pretrain", "finetune", "localizer"

Expected weights file: weights/best_{mode}.pt
ONNX output:           {model}_{mode}.onnx
MLflow run name:       {model}_{mode}_benchmark

Required .env variables:
    DAGSHUB_TOKEN
    DAGSHUB_REPO_OWNER
    DAGSHUB_REPO_NAME
"""

import argparse
import glob
import os
import shutil
import time

# Prevent ultralytics from auto-installing/upgrading packages (e.g. onnxruntime-gpu)
# which bypasses uv and overwrites pinned DLLs, causing load failures on Windows.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2
import mlflow
import mlflow.onnx
import numpy as np
import onnxruntime as ort
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

ONNX_OPSET     = 19
INPUT_IMGSZ    = 512
BENCHMARK_IMGS = 5

DATA_PRETRAIN           = "pretrain_dataset/data.yaml"
DATA_FINETUNE_LOCALIZER = "finetune_dataset/staves_localizer/stavesImg50.yaml"
DATA_FINETUNE_COUNTER   = "finetune_dataset/stave_count/ft2imgs.yaml"

MODE_CONFIG = {
    "pretrain":  DATA_PRETRAIN,
    "finetune":  DATA_FINETUNE_COUNTER,
    "localizer": DATA_FINETUNE_LOCALIZER,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and register a YOLO model to MLflow.")
    parser.add_argument("--model", required=True, help="YOLO model stem name, e.g. yolov8n")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pretrain", "finetune", "localizer"],
        help="Training mode used to locate the weights file",
    )
    return parser.parse_args()


def load_val_images(data_yaml: str, n: int, imgsz: int):
    """
    Returns two parallel lists:
      img_paths  – absolute paths to the first n val images (for PyTorch)
      img_arrays – corresponding preprocessed float32 numpy arrays (for ONNX)
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    val_dir  = os.path.join(yaml_dir, cfg["val"])

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(val_dir, ext)))
    paths = sorted(paths)[:n]

    if not paths:
        raise FileNotFoundError(f"No val images found in: {val_dir}")

    arrays = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.resize(img, (imgsz, imgsz))
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalise
        img = np.transpose(img, (2, 0, 1))[np.newaxis]    # HWC→CHW, add batch dim
        arrays.append(np.ascontiguousarray(img))

    return paths, arrays


def benchmark_pytorch(pt_path: str, img_paths: list) -> list:
    """Time inference on real val images (CPU). Returns latencies in ms."""
    model = YOLO(pt_path)
    latencies = []
    for p in img_paths:
        t0 = time.perf_counter()
        model(p, verbose=False, device="cpu")
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def benchmark_onnx(onnx_path: str, img_arrays: list) -> list:
    """Time inference on real val images via onnxruntime (CPU). Returns latencies in ms."""
    session    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    latencies  = []
    for arr in img_arrays:
        t0 = time.perf_counter()
        session.run(None, {input_name: arr})
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def main():
    args       = parse_args()
    model_name = args.model
    mode       = args.mode
    run_name   = f"{model_name}_{mode}_benchmark"
    pt_path    = os.path.join("weights", f"best_{mode}.pt")
    onnx_name  = f"{model_name}_{mode}.onnx"
    data_yaml  = MODE_CONFIG[mode]

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Weights file not found: {pt_path}")

    # MLflow / Dagshub setup
    load_dotenv()

    mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("staves-detector")

    mlflow.start_run(run_name=run_name)
    try:
        # --- Params ---
        mlflow.log_params({
            "model":            model_name,
            "mode":             mode,
            "onnx_opset":       ONNX_OPSET,
            "input_imgsz":      INPUT_IMGSZ,
            "benchmark_n_imgs": BENCHMARK_IMGS,
        })

        # --- Step 1: Validate ---
        print(f"[1/8] Running PyTorch validation on {pt_path}...")
        val_model    = YOLO(pt_path)
        val_results  = val_model.val(data=data_yaml, device="cpu")
        mlflow.log_metrics({
            "val/pt_mAP50":     float(val_results.box.map50),
            "val/pt_mAP50-95":  float(val_results.box.map),
            "val/pt_precision": float(val_results.box.mp),
            "val/pt_recall":    float(val_results.box.mr),
        })

        # --- Load val images once (shared by both benchmarks) ---
        print(f"[*] Loading {BENCHMARK_IMGS} val images for benchmarking...")
        img_paths, img_arrays = load_val_images(data_yaml, BENCHMARK_IMGS, INPUT_IMGSZ)
        print(f"    Using: {[os.path.basename(p) for p in img_paths]}")

        # --- Step 2: PyTorch benchmark ---
        print("[2/8] Benchmarking PyTorch (CPU)...")
        pt_latencies = benchmark_pytorch(pt_path, img_paths)
        pt_mean_ms   = float(np.mean(pt_latencies))
        pt_p50_ms    = float(np.percentile(pt_latencies, 50))
        pt_size_mb   = os.path.getsize(pt_path) / (1024 ** 2)
        mlflow.log_metrics({
            "inference/pt_mean_ms": pt_mean_ms,
            "inference/pt_p50_ms":  pt_p50_ms,
            "model/pt_size_mb":     pt_size_mb,
        })

        # --- Step 3: ONNX export ---
        print("[3/8] Exporting to ONNX...")
        export_model  = YOLO(pt_path)
        exported_path = export_model.export(format="onnx", imgsz=INPUT_IMGSZ, opset=ONNX_OPSET, simplify=False, device="cpu")

        if exported_path and os.path.exists(str(exported_path)):
            shutil.move(str(exported_path), onnx_name)
        else:
            candidate = pt_path.replace(".pt", ".onnx")
            if os.path.exists(candidate):
                shutil.move(candidate, onnx_name)
            else:
                raise FileNotFoundError("ONNX export did not produce a file.")

        onnx_size_mb = os.path.getsize(onnx_name) / (1024 ** 2)
        mlflow.log_metric("model/onnx_size_mb", onnx_size_mb)

        # --- Step 4: ONNX validation ---
        print(f"[4/8] Running ONNX validation on {onnx_name}...")
        onnx_val_model   = YOLO(onnx_name)
        onnx_val_results = onnx_val_model.val(data=data_yaml, device="cpu", imgsz=INPUT_IMGSZ)
        mlflow.log_metrics({
            "val/onnx_mAP50":     float(onnx_val_results.box.map50),
            "val/onnx_mAP50-95":  float(onnx_val_results.box.map),
            "val/onnx_precision": float(onnx_val_results.box.mp),
            "val/onnx_recall":    float(onnx_val_results.box.mr),
        })

        # --- Step 5: ONNX benchmark ---
        print("[5/8] Benchmarking ONNX (CPU)...")
        onnx_latencies = benchmark_onnx(onnx_name, img_arrays)
        onnx_mean_ms   = float(np.mean(onnx_latencies))
        onnx_p50_ms    = float(np.percentile(onnx_latencies, 50))
        mlflow.log_metrics({
            "inference/onnx_mean_ms": onnx_mean_ms,
            "inference/onnx_p50_ms":  onnx_p50_ms,
        })

        # --- Step 6: Comparison ---
        print("[6/8] Computing comparison metrics...")
        speedup_ratio         = pt_mean_ms / onnx_mean_ms
        latency_reduction_pct = (1.0 - onnx_mean_ms / pt_mean_ms) * 100
        size_reduction_mb     = pt_size_mb - onnx_size_mb
        mlflow.log_metrics({
            "optimization/speedup_ratio":         speedup_ratio,
            "optimization/latency_reduction_pct": latency_reduction_pct,
            "optimization/size_reduction_mb":     size_reduction_mb,
        })

        # --- Step 7: Log artifacts ---
        # Use mlflow.onnx.log_model so the artifact has an MLmodel manifest,
        # which is required for register_model to find it via runs:/ URI.
        print("[7/8] Logging ONNX artifact...")
        import onnx as onnx_lib
        onnx_model_proto = onnx_lib.load(onnx_name)
        model_info = mlflow.onnx.log_model(onnx_model_proto, artifact_path="onnx_model")

        # --- Step 8: Register to Model Registry ---
        print("[8/8] Registering model to MLflow Model Registry...")
        registry_name = f"staves_detector_{mode}_onnx"
        mv = mlflow.register_model(model_uri=model_info.model_uri, name=registry_name)

        description = (
            f"ONNX export of {model_name} trained in '{mode}' mode. "
            f"Speedup vs PyTorch: {speedup_ratio:.2f}x. "
            f"val/onnx_mAP50: {float(onnx_val_results.box.map50):.4f}. "
            f"Do NOT promote to Production without reviewing benchmarks in Dagshub UI."
        )
        client = MlflowClient()
        client.update_model_version(
            name=registry_name,
            version=mv.version,
            description=description,
        )

        print(f"\nDone. Registered as '{registry_name}' version {mv.version}")
        print(f"  PT mean/p50 latency:   {pt_mean_ms:.1f} ms / {pt_p50_ms:.1f} ms")
        print(f"  ONNX mean/p50 latency: {onnx_mean_ms:.1f} ms / {onnx_p50_ms:.1f} ms")
        print(f"  Speedup:               {speedup_ratio:.2f}x")
        print(f"  Latency reduction: {latency_reduction_pct:.1f}%")
        print(f"  Size reduction:    {size_reduction_mb:.1f} MB")

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
