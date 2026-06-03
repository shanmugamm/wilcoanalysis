from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owner_ml.data import find_latest_owner_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all owner-data reports for the latest extract.")
    parser.add_argument("--data", help="Specific Owner CSV to process. Defaults to newest Owner_*.csv.")
    parser.add_argument("--data-dir", default=".", help="Directory to search for Owner_*.csv.")
    parser.add_argument("--output-dir", default="reports", help="Directory for generated outputs.")
    parser.add_argument("--clusters", type=int, default=8, help="Clusters for the segmentation model.")
    parser.add_argument(
        "--sample-segments",
        type=int,
        help="Use a row sample for segmentation instead of the full file.",
    )
    parser.add_argument(
        "--primary-only-multi-property",
        action="store_true",
        help="Only count primary owners in the multiple-property report.",
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print()
    print("Running:", " ".join(command))
    subprocess.check_call(command)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else find_latest_owner_csv(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "pipeline_manifest.md"
    started_at = datetime.now().astimezone()

    commands = [
        [
            sys.executable,
            "scripts/01_profile_owner_data.py",
            "--data",
            str(data_path),
            "--output",
            str(output_dir / "owner_profile.md"),
        ],
        [
            sys.executable,
            "scripts/03_owner_type_geo_analysis.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ],
        [
            sys.executable,
            "scripts/04_multi_property_owners.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ],
        [
            sys.executable,
            "scripts/02_owner_segments.py",
            "--data",
            str(data_path),
            "--clusters",
            str(args.clusters),
            "--output-dir",
            str(output_dir),
        ],
    ]

    if args.primary_only_multi_property:
        commands[2].append("--primary-only")

    if args.sample_segments:
        commands[3].extend(["--sample-rows", str(args.sample_segments)])
    else:
        commands[3].append("--full")

    print(f"Using data file: {data_path}")
    for command in commands:
        run_step(command)

    finished_at = datetime.now().astimezone()
    manifest_path.write_text(
        "\n".join(
            [
                "# Owner Pipeline Manifest",
                "",
                f"- Data file: `{data_path}`",
                f"- Started: `{started_at.isoformat(timespec='seconds')}`",
                f"- Finished: `{finished_at.isoformat(timespec='seconds')}`",
                f"- Output directory: `{output_dir}`",
                "",
                "## Generated Report Entrypoints",
                "",
                "- `reports/owner_profile.md`",
                "- `reports/owner_type_geo_analysis.md`",
                "- `reports/multi_property_owners_report.md`",
                "- `reports/owner_segmentation_report.md`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
