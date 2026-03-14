"""
Downloads a registered ONNX model from the MLflow Model Registry (hosted on Dagshub).

Usage:
    uv run python download_registered_model.py --mode finetune
    uv run python download_registered_model.py --mode localizer --version 2
    uv run python download_registered_model.py --mode finetune --out models/counterModel.onnx

Args:
    --mode     : one of "pretrain", "finetune", "localizer"
    --version  : model version number (int). If omitted, downloads the latest version.
    --out      : destination file path. Default: ./weights/{model_name}_{mode}.onnx in cwd.

Required .env variables:
    MLFLOW_TRACKING_URI
    MLFLOW_TRACKING_USERNAME
    MLFLOW_TRACKING_PASSWORD
"""

import argparse
import os
import shutil

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import mlflow


def parse_args():
    parser = argparse.ArgumentParser(description="Download a registered ONNX model from MLflow.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pretrain", "finetune", "localizer"],
        help="Training mode — selects the registry name staves_detector_{mode}_onnx",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Model version number. Omit to use the latest registered version.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Destination file path for the .onnx file. Default: ./weights/{registry_name}_v{version}.onnx",
    )
    return parser.parse_args()


def get_latest_version(client: MlflowClient, registry_name: str) -> str:
    """Return the version string of the most recently created model version."""
    versions = client.search_model_versions(f"name='{registry_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{registry_name}' in the registry.")
    # search_model_versions returns newest-first by default; take the first
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    return latest.version


def main():
    args = parse_args()

    load_dotenv()
    mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(mlflow_uri)

    client        = MlflowClient()
    registry_name = f"staves_detector_{args.mode}_onnx"

    # Resolve version
    if args.version is not None:
        version = str(args.version)
    else:
        version = get_latest_version(client, registry_name)
        print(f"No --version specified. Using latest: v{version}")

    # Fetch model version metadata
    mv = client.get_model_version(name=registry_name, version=version)
    print(f"Downloading '{registry_name}' v{version}")
    print(f"  Source run  : {mv.run_id}")
    print(f"  Description : {mv.description or '(none)'}")

    # Determine output path
    out_path = args.out or f"weights/{registry_name}_v{version}.onnx"

    # Download the MLflow model artifact directory to a temp location
    model_uri = f"models:/{registry_name}/{version}"
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    # The mlflow.onnx.log_model artifact contains model.onnx inside the directory
    onnx_candidate = os.path.join(local_dir, "model.onnx")
    if not os.path.exists(onnx_candidate):
        # Fallback: search recursively for any .onnx file
        found = []
        for root, _, files in os.walk(local_dir):
            for f in files:
                if f.endswith(".onnx"):
                    found.append(os.path.join(root, f))
        if not found:
            raise FileNotFoundError(
                f"No .onnx file found in downloaded artifact directory: {local_dir}"
            )
        onnx_candidate = found[0]
        print(f"  Found ONNX at: {onnx_candidate}")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(onnx_candidate, out_path)

    # Copy any companion .data files (ONNX external data) from the same artifact directory
    onnx_dir = os.path.dirname(onnx_candidate)
    for f in os.listdir(onnx_dir):
        if f.endswith(".data"):
            src = os.path.join(onnx_dir, f)
            dst = os.path.join(out_dir, f)
            shutil.copy2(src, dst)
            print(f"  Copied external data: {f}")

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\nSaved to: {os.path.abspath(out_path)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
