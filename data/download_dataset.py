#!/usr/bin/env python3
"""
Dataset Downloader — Configurable for S3, HuggingFace Hub, and local sources.

This script is designed to run in a Kubernetes init container to pre-download
datasets before the training containers start.

Supports:
  - HuggingFace Hub datasets (public or gated)
  - AWS S3 (via boto3 or aws CLI)
  - Any S3-compatible backend (MinIO, GCS with S3 compat, etc.)
  - Local file system (copy/symlink)

Usage:
  # Download from HuggingFace Hub
  python download_dataset.py --source huggingface --name tatsu-lab/alpaca --output /mnt/data/dataset

  # Download from S3
  python download_dataset.py --source s3 --s3-bucket my-bucket --s3-prefix datasets/my-data --output /mnt/data/dataset

  # Download from S3-compatible (e.g., MinIO)
  python download_dataset.py --source s3 --s3-bucket my-bucket --s3-prefix data/ \
      --s3-endpoint http://minio:9000 --output /mnt/data/dataset

Environment Variables:
  HF_TOKEN          — HuggingFace API token (for gated models/datasets)
  AWS_ACCESS_KEY_ID — S3 access key
  AWS_SECRET_ACCESS_KEY — S3 secret key
  S3_ENDPOINT_URL   — Custom S3 endpoint (MinIO, etc.)
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dataset-downloader")


def download_from_huggingface(name: str, output: str, split: str = "train",
                               revision: str = None):
    """Download dataset from HuggingFace Hub and save to disk."""
    from datasets import load_dataset

    logger.info(f"Downloading from HuggingFace Hub: {name} (split={split})")

    token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(
        name,
        split=split,
        token=token,
        revision=revision,
    )

    # Save to disk in Arrow format (efficient for training)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    logger.info(f"Saved {len(dataset)} samples to: {output_path}")
    logger.info(f"Dataset columns: {dataset.column_names}")
    try:
        logger.info(f"Dataset size: {dataset.data.nbytes / 1e6:.1f} MB")
    except (AttributeError, TypeError):
        logger.info("Dataset size: (unable to determine)")

    return dataset


def download_from_s3(bucket: str, prefix: str, output: str,
                     endpoint_url: str = None, region: str = None):
    """Download dataset from S3 or S3-compatible storage."""
    import boto3
    from botocore.config import Config

    logger.info(f"Downloading from S3: s3://{bucket}/{prefix}")

    # Configure S3 client
    client_kwargs = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
        logger.info(f"Using custom S3 endpoint: {endpoint_url}")
    if region:
        client_kwargs["region_name"] = region

    s3 = boto3.client("s3", **client_kwargs,
                      config=Config(max_pool_connections=20))

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # List and download all objects under prefix
    paginator = s3.get_paginator("list_objects_v2")
    total_files = 0
    total_bytes = 0
    max_retries = 3

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Compute relative path from prefix
            rel_path = key[len(prefix):].lstrip("/")
            if not rel_path:
                continue

            local_path = output_path / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Retry with exponential backoff for transient failures
            for attempt in range(max_retries):
                try:
                    logger.info(f"  Downloading: {key} → {local_path}")
                    s3.download_file(bucket, key, str(local_path))
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"  Retry {attempt + 1}/{max_retries} for {key}: {e}. Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"  Failed to download {key} after {max_retries} attempts: {e}")
                        raise
            total_files += 1
            total_bytes += obj.get("Size", 0)

    if total_files == 0:
        logger.warning(f"No files found under s3://{bucket}/{prefix} — is the prefix correct?")

    logger.info(f"Downloaded {total_files} files ({total_bytes / 1e6:.1f} MB) to: {output_path}")


def download_from_local(source_path: str, output: str):
    """Copy or symlink local dataset."""
    source = Path(source_path)
    output_path = Path(output)

    if not source.exists():
        raise FileNotFoundError(f"Local dataset not found: {source_path}")

    if source.is_dir():
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(str(source), str(output_path))
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(output_path / source.name))

    logger.info(f"Copied local dataset: {source_path} → {output_path}")


def verify_dataset(output: str):
    """Verify the downloaded dataset is valid."""
    output_path = Path(output)

    if not output_path.exists():
        raise RuntimeError(f"Output directory does not exist: {output}")

    files = list(output_path.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())

    if file_count == 0:
        raise RuntimeError(f"No files found in: {output}")

    total_size = sum(f.stat().st_size for f in files if f.is_file())
    logger.info(f"Dataset verification passed: {file_count} files, {total_size / 1e6:.1f} MB")

    # Try loading as HF dataset
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(output_path))
        logger.info(f"HF Dataset loaded: {len(ds)} samples, columns={ds.column_names}")
    except Exception:
        logger.info("Not an Arrow dataset — will be loaded as raw files during training")


def main():
    parser = argparse.ArgumentParser(description="Download training dataset")
    parser.add_argument("--source", required=True,
                        choices=["huggingface", "s3", "local"],
                        help="Dataset source type")
    parser.add_argument("--output", required=True,
                        help="Output directory for downloaded data")

    # HuggingFace options
    parser.add_argument("--name", type=str,
                        help="HuggingFace dataset name (e.g., tatsu-lab/alpaca)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to download")
    parser.add_argument("--revision", type=str, default=None,
                        help="Dataset revision/version for versioning")

    # S3 options
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default="",
                        help="S3 key prefix")
    parser.add_argument("--s3-endpoint", type=str, default=None,
                        help="Custom S3 endpoint URL (for MinIO, etc.)")
    parser.add_argument("--s3-region", type=str, default=None,
                        help="AWS region")

    # Local options
    parser.add_argument("--local-path", type=str,
                        help="Local path to dataset")

    # Misc
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip download if output already has data")
    parser.add_argument("--no-verify", action="store_true", default=False,
                        help="Skip dataset verification after download")

    args = parser.parse_args()
    args.verify = not args.no_verify  # Default is to verify

    try:
        _run_download(args)
    except Exception as e:
        logger.error(f"Dataset download failed: {e}", exc_info=True)
        sys.exit(1)


def _run_download(args):

    # Skip if already downloaded
    if args.skip_if_exists and Path(args.output).exists():
        files = list(Path(args.output).rglob("*"))
        if any(f.is_file() for f in files):
            logger.info(f"Dataset already exists at {args.output}, skipping download")
            if args.verify:
                verify_dataset(args.output)
            return

    # Download
    if args.source == "huggingface":
        if not args.name:
            raise ValueError("--name is required for HuggingFace datasets")
        download_from_huggingface(args.name, args.output, args.split, args.revision)

    elif args.source == "s3":
        if not args.s3_bucket:
            raise ValueError("--s3-bucket is required for S3 datasets")
        endpoint = args.s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
        download_from_s3(args.s3_bucket, args.s3_prefix, args.output,
                         endpoint_url=endpoint, region=args.s3_region)

    elif args.source == "local":
        if not args.local_path:
            raise ValueError("--local-path is required for local datasets")
        download_from_local(args.local_path, args.output)

    # Verify
    if args.verify:
        verify_dataset(args.output)

    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()
