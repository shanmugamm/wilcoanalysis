from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owner_ml.data import DEFAULT_DATA_PATH, read_owner_chunks
from owner_ml.features import add_owner_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze owner types and mailing geography.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to owner CSV.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk.")
    parser.add_argument("--output-dir", default="reports", help="Directory for report outputs.")
    parser.add_argument("--top-n", type=int, default=20, help="Top places to include in tables.")
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{value:.1%}"


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    columns = [
        "OwnerTypeGuess",
        "MailingGeo",
        "IsOutOfAreaMailing",
        "State",
        "City",
        "Exemption_HS",
        "Exemption_AG",
        "Exemption_DV",
        "Exemption_OV",
        "ExemptionCount",
        "IsUndeliverable",
        "PercentOwnership",
        "PrimaryOwner",
    ]

    for chunk in read_owner_chunks(args.data, chunksize=args.chunksize):
        featured = add_owner_features(chunk)
        chunks.append(featured[[column for column in columns if column in featured.columns]])

    df = pd.concat(chunks, ignore_index=True)
    total_rows = len(df)

    owner_type_summary = (
        df.groupby("OwnerTypeGuess", dropna=False)
        .agg(
            rows=("OwnerTypeGuess", "size"),
            out_of_area_rate=("IsOutOfAreaMailing", "mean"),
            homestead_rate=("Exemption_HS", "mean"),
            ag_rate=("Exemption_AG", "mean"),
            disabled_veteran_rate=("Exemption_DV", "mean"),
            over_65_rate=("Exemption_OV", "mean"),
            undeliverable_rate=("IsUndeliverable", "mean"),
            avg_percent_ownership=("PercentOwnership", "mean"),
            avg_exemption_count=("ExemptionCount", "mean"),
        )
        .sort_values("rows", ascending=False)
        .reset_index()
    )
    owner_type_summary["share"] = owner_type_summary["rows"] / total_rows

    geo_summary = (
        df.groupby("MailingGeo", dropna=False)
        .agg(
            rows=("MailingGeo", "size"),
            business_rate=("OwnerTypeGuess", lambda value: (value == "business").mean()),
            trust_estate_rate=("OwnerTypeGuess", lambda value: (value == "trust_estate").mean()),
            homestead_rate=("Exemption_HS", "mean"),
            undeliverable_rate=("IsUndeliverable", "mean"),
        )
        .sort_values("rows", ascending=False)
        .reset_index()
    )
    geo_summary["share"] = geo_summary["rows"] / total_rows

    owner_type_by_geo = (
        df.pivot_table(
            index="OwnerTypeGuess",
            columns="MailingGeo",
            values="PrimaryOwner",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    count_columns = [column for column in owner_type_by_geo.columns if column != "OwnerTypeGuess"]
    owner_type_by_geo["total"] = owner_type_by_geo[count_columns].sum(axis=1)
    owner_type_by_geo = owner_type_by_geo.sort_values("total", ascending=False)

    top_out_of_state = (
        df[df["MailingGeo"].eq("out_of_state")]
        .groupby(["State", "City", "OwnerTypeGuess"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .head(args.top_n)
    )

    top_texas_other = (
        df[df["MailingGeo"].eq("texas_other")]
        .groupby(["City", "OwnerTypeGuess"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .head(args.top_n)
    )

    write_table(owner_type_summary, output_dir / "owner_type_summary.csv")
    write_table(geo_summary, output_dir / "mailing_geo_summary.csv")
    write_table(owner_type_by_geo, output_dir / "owner_type_by_geo.csv")
    write_table(top_out_of_state, output_dir / "top_out_of_state_owner_places.csv")
    write_table(top_texas_other, output_dir / "top_texas_other_owner_places.csv")

    report_lines = [
        "# Owner Type And Mailing Geography Analysis",
        "",
        f"- Source: `{args.data}`",
        f"- Rows analyzed: `{total_rows:,}`",
        "",
        "## Key Counts",
        "",
    ]

    for _, row in owner_type_summary.iterrows():
        report_lines.append(
            f"- `{row['OwnerTypeGuess']}`: `{int(row['rows']):,}` records "
            f"({percent(row['share'])}), out-of-area rate `{percent(row['out_of_area_rate'])}`"
        )

    report_lines.extend(["", "## Mailing Geography", ""])
    for _, row in geo_summary.iterrows():
        report_lines.append(
            f"- `{row['MailingGeo']}`: `{int(row['rows']):,}` records "
            f"({percent(row['share'])}), business rate `{percent(row['business_rate'])}`, "
            f"homestead rate `{percent(row['homestead_rate'])}`"
        )

    report_lines.extend(["", "## Output Tables", ""])
    for filename in [
        "owner_type_summary.csv",
        "mailing_geo_summary.csv",
        "owner_type_by_geo.csv",
        "top_out_of_state_owner_places.csv",
        "top_texas_other_owner_places.csv",
    ]:
        report_lines.append(f"- `{output_dir / filename}`")

    report_path = output_dir / "owner_type_geo_analysis.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
