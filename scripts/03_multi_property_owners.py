from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owner_ml.data import DEFAULT_DATA_PATH, read_owner_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find owners with 2+ properties.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to owner CSV.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk.")
    parser.add_argument("--min-properties", type=int, default=2, help="Minimum properties to report.")
    parser.add_argument("--output", default="reports/multi_property_owners.md", help="Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Track property counts by owner
    owner_property_count = defaultdict(int)
    owner_properties = defaultdict(list)
    
    # Track columns
    columns: list[str] | None = None
    
    for chunk in read_owner_chunks(data_path, chunksize=args.chunksize):
        if columns is None:
            columns = list(chunk.columns)
        
        # Try to identify owner columns - common patterns
        owner_col = None
        for col in ["OwnerName", "Owner", "Name", "PrimaryOwner"]:
            if col in chunk.columns:
                owner_col = col
                break
        
        if owner_col is None:
            # Use first string column as fallback
            for col in chunk.columns:
                if chunk[col].dtype == "object":
                    owner_col = col
                    break
        
        if owner_col:
            for idx, row in chunk.iterrows():
                owner = str(row[owner_col]).strip()
                if owner and owner.lower() not in ["", "(blank)", "nan"]:
                    owner_property_count[owner] += 1
                    owner_properties[owner].append(idx)
    
    # Filter owners with 2+ properties
    multi_property_owners = {
        owner: count for owner, count in owner_property_count.items()
        if count >= args.min_properties
    }
    
    # Sort by property count (descending)
    sorted_owners = sorted(multi_property_owners.items(), key=lambda x: x[1], reverse=True)
    
    # Generate report
    lines = [
        "# Multi-Property Owners Report",
        "",
        f"- Source: `{data_path}`",
        f"- Minimum Properties: `{args.min_properties}`",
        f"- Total Multi-Property Owners: `{len(sorted_owners):,}`",
        "",
        "## Top Owners by Property Count",
        "",
    ]
    
    for owner, count in sorted_owners[:50]:  # Top 50
        lines.append(f"- `{owner}`: `{count}` properties")
    
    lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"- Total unique owners: `{len(owner_property_count):,}`",
        f"- Owners with 2+ properties: `{len(sorted_owners):,}`",
        f"- Percentage of multi-property owners: `{len(sorted_owners) / len(owner_property_count) * 100:.2f}%`",
    ])
    
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Found {len(sorted_owners):,} owners with {args.min_properties}+ properties")


if __name__ == "__main__":
    main()
