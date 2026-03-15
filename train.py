"""
Usage:
uv run python train.py --model yolo11s --mode pretrain

Modes: 
pretrain: pretrains with domain relevant dataset
finetune: finetunes model with target dataset
localizer: train first stage localzier
"""

import argparse
import os
import shutil

import mlflow
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

EPOCHS    = 50
IMGSZ     = 512
BATCH     = 16
LR0       = 0.01
OPTIMIZER = "AdamW"
MOSAIC    = 1.0
FLIPUD    = 0.0
FLIPLR    = 0.5

# finetune: finetuning counter model
# localizer: finetuning 1st stage localizer model
# pretrain: pre-training staves counter model with domain relevant dataset from roboflow dataset
DATA_PRETRAIN  = "pretrain_dataset/data.yaml"
DATA_FINETUNE_LOCALIZER  = "finetune_dataset/staves_localizer/stavesImg50.yaml"
DATA_FINETUNE_COUNTER   = "finetune_dataset/stave_count/ft2imgs.yaml"

MODE_CONFIG = {
    "pretrain":  DATA_PRETRAIN,
    "finetune":  DATA_FINETUNE_COUNTER,
    "localizer": DATA_FINETUNE_LOCALIZER,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model with MLflow tracking.")
    parser.add_argument("--model", required=True, help="YOLO model name, ex yolov8n")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pretrain", "finetune", "localizer"],
        help="pretrain - pretrains with domain relevant dataset | finetune - finetunes model with target dataset| localizer - train first stage localzier",
    )
    return parser.parse_args()

# need callback fn to log metrics to mlflow, per epoch
def on_fit_epoch_end(trainer):
    """Ultralytics callback – logs per-epoch metrics to the active MLflow run."""
    metrics = trainer.metrics
    step = trainer.epoch

    to_log = {k: float(v) for k, v in {
        "metrics/mAP50": metrics.get("metrics/mAP50"),
        "metrics/mAP50-95": metrics.get("metrics/mAP50-95"),
        "metrics/precision": metrics.get("metrics/precision"),
    }.items() if v is not None}
    if to_log:
        mlflow.log_metrics(to_log, step=step)


def main():
    args = parse_args()
    model_name = args.model
    mode       = args.mode
    run_name   = f"{model_name}_{mode}"
    data_yaml  = MODE_CONFIG[mode]

    # Mlflow setup: connect to dagshub hosted mlflow. needs the MLFLOW_TRACKING_PASSWORD and MLFLOW_TRACKING_USERNAME env vars
    load_dotenv()
    mlflow_uri      = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("staves-detector")

    # For finetune mode, prefer a pretrain checkpoint over the base model weights
    finetune_base = None
    if mode == "finetune":
        for candidate in ["weights/best_pretrain.pt", "weights/last_pretrain.pt"]:
            if os.path.exists(candidate):
                finetune_base = candidate
                print(f"[finetune] Loading pretrain checkpoint: {finetune_base}")
                break
        if finetune_base is None:
            print("[finetune] No pretrain checkpoint found in weights/, starting from base model.")

    pretrained = finetune_base is not None or model_name.endswith(".pt")
    weights_to_load = finetune_base if finetune_base else model_name

    run = mlflow.start_run(run_name=run_name)
    try:
        mlflow.log_params({
            "model":            model_name,
            "mode":             mode,
            "epochs":           EPOCHS,
            "imgsz":            IMGSZ,
            "batch":            BATCH,
            "lr0":              LR0,
            "optimizer":        OPTIMIZER,
            "mosaic":           MOSAIC,
            "flipud":           FLIPUD,
            "fliplr":           FLIPLR,
            "num_classes":      1,
            "dataset":          data_yaml,
            "pretrained":       pretrained,
            "pretrain_weights": finetune_base or "none",
        })

        # yolo model init and train

        model = YOLO(weights_to_load)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            lr0=LR0,
            optimizer=OPTIMIZER,
            mosaic=MOSAIC,
            flipud=FLIPUD,
            fliplr=FLIPLR,
            project="results",
            name=run_name,
        )

        # saving artifacts post training to mlflow
        save_dir = str(results.save_dir)

        for artifact_name in [
            "confusion_matrix.png",
            "PR_curve.png",
            "F1_curve.png",
            "results.png",
        ]:
            artifact_path = os.path.join(save_dir, artifact_name)
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

        weights_dir = os.path.join(save_dir, "weights")
        os.makedirs("weights", exist_ok=True)

        best_src  = os.path.join(weights_dir, "best.pt")
        last_src  = os.path.join(weights_dir, "last.pt")
        best_dest = os.path.join("weights", f"best_{mode}.pt")
        last_dest = os.path.join("weights", f"last_{mode}.pt")

        if os.path.exists(best_src):
            shutil.copy2(best_src, best_dest)
            mlflow.log_artifact(best_dest, artifact_path="weights")

        if os.path.exists(last_src):
            shutil.copy2(last_src, last_dest)
            mlflow.log_artifact(last_dest, artifact_path="weights")

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
