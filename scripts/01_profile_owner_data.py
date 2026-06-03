from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owner_ml.data import DEFAULT_DATA_PATH, read_owner_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the owner CSV in chunks.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to owner CSV.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per profiling chunk.")
    parser.add_argument("--top-n", type=int, default=15, help="Top values to keep for selected columns.")
    parser.add_argument("--output", default="reports/owner_profile.md", help="Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    row_count = 0
    columns: list[str] | None = None
    missing = defaultdict(int)
    top_columns = [
        "State",
        "City",
        "AddressChgReasonDesc",
        "AddressTypeKey",
        "ExemptionList",
        "IsUndeliverable",
        "PrimaryOwner",
    ]
    counters = {column: Counter() for column in top_columns}
    for chunk in read_owner_chunks(data_path, chunksize=args.chunksize):
        if columns is None:
            columns = list(chunk.columns)
        row_count += len(chunk)
        for column, count in chunk.isna().sum().to_dict().items():
            missing[column] += int(count)

        for column in top_columns:
            if column in chunk:
                counters[column].update(chunk[column].fillna("").astype(str).replace("", "(blank)"))

    lines = [
        "# Owner Data Profile",
        "",
        f"- Source: `{data_path}`",
        f"- Rows: `{row_count:,}`",
        f"- Columns: `{0 if columns is None else len(columns):,}`",
        "",
        "## Columns",
        "",
    ]
    lines.extend(f"- `{column}`" for column in (columns or []))

    lines.extend(["", "## Missing Values", ""])
    for column in columns or []:
        lines.append(f"- `{column}`: `{missing[column]:,}`")

    for column, counter in counters.items():
        if not counter:
            continue
        lines.extend(["", f"## Top `{column}` Values", ""])
        for value, count in counter.most_common(args.top_n):
            lines.append(f"- `{value}`: `{count:,}`")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
