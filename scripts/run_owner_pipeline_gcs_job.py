from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected a GCS object URI like gs://bucket/path/file.csv, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def parse_gcs_prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected a GCS prefix URI like gs://bucket/reports/run-id, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_object(client: storage.Client, source_uri: str, destination: Path) -> None:
    bucket_name, blob_name = parse_gcs_uri(source_uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(destination)
    print(f"Downloaded {source_uri} to {destination}")


def upload_directory(client: storage.Client, source_dir: Path, destination_uri: str) -> None:
    bucket_name, prefix = parse_gcs_prefix(destination_uri)
    bucket = client.bucket(bucket_name)
    uploaded = 0

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_dir).as_posix()
        blob_name = f"{prefix.rstrip('/')}/{relative_path}" if prefix else relative_path
        bucket.blob(blob_name).upload_from_filename(path)
        uploaded += 1
        print(f"Uploaded {path} to gs://{bucket_name}/{blob_name}")

    if uploaded == 0:
        raise RuntimeError(f"No files found to upload from {source_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the owner report pipeline as a Cloud Run Job.")
    parser.add_argument(
        "--input-gcs-uri",
        default=os.getenv("OWNER_INPUT_GCS_URI"),
        help="Input owner extract URI, for example gs://bucket/raw/Owner_20260602.csv.",
    )
    parser.add_argument(
        "--output-gcs-prefix",
        default=os.getenv("REPORT_OUTPUT_GCS_PREFIX"),
        help="Output report prefix, for example gs://bucket/reports/latest.",
    )
    parser.add_argument("--clusters", default=os.getenv("OWNER_SEGMENT_CLUSTERS", "8"))
    parser.add_argument("--sample-segments", default=os.getenv("OWNER_SAMPLE_SEGMENTS"))
    parser.add_argument(
        "--primary-only-multi-property",
        action="store_true",
        default=os.getenv("OWNER_PRIMARY_ONLY_MULTI_PROPERTY", "").lower() in {"1", "true", "yes"},
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_gcs_uri:
        raise SystemExit("Missing --input-gcs-uri or OWNER_INPUT_GCS_URI.")
    if not args.output_gcs_prefix:
        raise SystemExit("Missing --output-gcs-prefix or REPORT_OUTPUT_GCS_PREFIX.")

    work_dir = Path(os.getenv("OWNER_JOB_WORK_DIR", "/tmp/wilcoanalysis-job"))
    data_path = work_dir / "input" / Path(parse_gcs_uri(args.input_gcs_uri)[1]).name
    output_dir = work_dir / "reports"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    download_object(client, args.input_gcs_uri, data_path)

    command = [
        sys.executable,
        "scripts/run_owner_pipeline.py",
        "--data",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--clusters",
        str(args.clusters),
    ]
    if args.sample_segments:
        command.extend(["--sample-segments", str(args.sample_segments)])
    if args.primary_only_multi_property:
        command.append("--primary-only-multi-property")

    print("Running:", " ".join(command))
    subprocess.check_call(command)
    upload_directory(client, output_dir, args.output_gcs_prefix)


if __name__ == "__main__":
    main()
