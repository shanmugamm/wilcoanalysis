from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owner_ml.data import DEFAULT_DATA_PATH, read_owner_sample
from owner_ml.features import add_owner_features


BASE_COLUMNS = [
    "OwnerID",
    "FullName",
    "MailingAddress",
    "State",
    "City",
    "ZIP",
    "OwnerTypeGuess",
    "OwnerProfileSegment",
    "MailingGeo",
    "PropertyID",
    "QuickRefID",
    "OwnerPropertyNumber",
    "PrimaryOwner",
    "PercentOwnership",
    "ExemptionList",
    "Address1",
    "Address2",
    "Address3",
]

ORGANIZATION_OWNER_TYPES = {"business", "government"}
ORGANIZATION_NAME_PATTERN = (
    r"\b(?:LLC|L L C|INC|CORP|CO\b|LTD|LP|LLP|BANK|HOLDINGS|PROPERTIES|"
    r"INVEST|VENTURES|PARTNERS|ASSOC|ASSOCIATION|COMPANY|CITY OF|COUNTY|STATE OF|ISD)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find owners connected to multiple properties.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to owner CSV.")
    parser.add_argument(
        "--min-properties",
        type=int,
        default=2,
        help="Minimum distinct properties required to include an owner.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Only count records where PrimaryOwner is 1.",
    )
    parser.add_argument(
        "--exclude-organizations",
        action="store_true",
        help="Exclude organization-style owners such as LLCs, corporations, banks, and governments.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Rows to include in the top-owner CSV.",
    )
    parser.add_argument("--output-dir", default="reports", help="Directory for outputs.")
    return parser.parse_args()


def mode_or_blank(values: pd.Series) -> str:
    non_blank = values.dropna()
    if non_blank.dtype == "object":
        non_blank = non_blank[non_blank.astype(str).str.len().gt(0)]
    mode = non_blank.mode()
    return "" if mode.empty else str(mode.iat[0])


def percent(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "OwnerID",
        "FullName",
        "MailingAddress",
        "State",
        "City",
        "ZIP",
        "PropertyID",
        "QuickRefID",
        "OwnerPropertyNumber",
        "PrimaryOwner",
        "PercentOwnership",
        "ExemptionList",
        "Address1",
        "Address2",
        "Address3",
        "DateAddrChanged",
        "DataDate",
        "TaxingUnitGroupDesc",
    ]
    df = add_owner_features(read_owner_sample(args.data, nrows=None, usecols=usecols))
    if args.primary_only:
        df = df[df["PrimaryOwner"].eq(1)].copy()
    excluded_organization_rows = 0
    if args.exclude_organizations:
        organization_mask = df["OwnerTypeGuess"].isin(ORGANIZATION_OWNER_TYPES) | df[
            "FullName"
        ].str.contains(ORGANIZATION_NAME_PATTERN, case=False, regex=True, na=False)
        excluded_organization_rows = int(organization_mask.sum())
        df = df[~organization_mask].copy()

    df = df[[column for column in BASE_COLUMNS if column in df.columns]].copy()
    df["OwnerKey"] = df["OwnerID"].where(df["OwnerID"].ne(""), df["FullName"] + "|" + df["MailingAddress"])

    grouped = df.groupby("OwnerKey", sort=False, dropna=False)
    summary = (
        grouped
        .agg(
            owner_key=("OwnerKey", "first"),
            owner_id=("OwnerID", "first"),
            full_name=("FullName", "first"),
            mailing_address=("MailingAddress", "first"),
            city=("City", "first"),
            state=("State", "first"),
            zip=("ZIP", "first"),
            owner_type=("OwnerTypeGuess", "first"),
            profile_segment=("OwnerProfileSegment", "first"),
            mailing_geo=("MailingGeo", "first"),
            owner_rows=("OwnerKey", "size"),
            distinct_properties=("PropertyID", "nunique"),
            distinct_quick_refs=("QuickRefID", "nunique"),
            avg_percent_ownership=("PercentOwnership", "mean"),
            primary_owner_rate=("PrimaryOwner", "mean"),
        )
        .reset_index(drop=True)
    )
    multi_owner_summary = (
        summary[summary["distinct_properties"].ge(args.min_properties)]
        .sort_values(["distinct_properties", "owner_rows"], ascending=False)
        .reset_index(drop=True)
    )

    multi_owner_keys = set(summary.loc[summary["distinct_properties"].ge(args.min_properties), "owner_key"])
    detail = (
        df[df["OwnerKey"].isin(multi_owner_keys)]
        .sort_values(["OwnerID", "FullName", "PropertyID", "QuickRefID"])
        .reset_index(drop=True)
    )

    top_owners = multi_owner_summary.head(args.top_n)
    total_owners = summary.shape[0]
    multi_owner_count = multi_owner_summary.shape[0]
    multi_property_records = detail["PropertyID"].nunique()

    summary_path = output_dir / "multi_property_owner_summary.csv"
    top_path = output_dir / "top_multi_property_owners.csv"
    detail_path = output_dir / "multi_property_owner_detail.csv"
    report_path = output_dir / "multi_property_owners_report.md"

    multi_owner_summary.to_csv(summary_path, index=False)
    top_owners.to_csv(top_path, index=False)
    detail.to_csv(detail_path, index=False)
    report_path.write_text(
        build_report(
            total_owners=total_owners,
            multi_owner_count=multi_owner_count,
            multi_property_records=multi_property_records,
            min_properties=args.min_properties,
            primary_only=args.primary_only,
            exclude_organizations=args.exclude_organizations,
            excluded_organization_rows=excluded_organization_rows,
            top_owners=top_owners,
        ),
        encoding="utf-8",
    )

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(top_owners.head(25))
    print(f"Wrote {summary_path}")
    print(f"Wrote {top_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {report_path}")


def build_report(
    total_owners: int,
    multi_owner_count: int,
    multi_property_records: int,
    min_properties: int,
    primary_only: bool,
    exclude_organizations: bool,
    excluded_organization_rows: int,
    top_owners: pd.DataFrame,
) -> str:
    lines = [
        "# Multiple-Property Owners",
        "",
        f"- Minimum distinct properties: `{min_properties}`",
        f"- Primary owners only: `{'yes' if primary_only else 'no'}`",
        f"- Exclude organizations: `{'yes' if exclude_organizations else 'no'}`",
        f"- Organization rows excluded: `{excluded_organization_rows:,}`",
        f"- Distinct owner keys analyzed: `{total_owners:,}`",
        f"- Owners meeting threshold: `{multi_owner_count:,}` ({percent(multi_owner_count / total_owners)})",
        f"- Distinct properties tied to these owners: `{multi_property_records:,}`",
        "",
        "## Top Owners",
        "",
    ]

    for _, row in top_owners.head(20).iterrows():
        lines.append(
            f"- `{row['full_name']}`: `{int(row['distinct_properties']):,}` properties, "
            f"`{row['owner_type']}`, `{row['mailing_geo']}`, mailing city `{row['city']}`"
        )

    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `reports/multi_property_owner_summary.csv`",
            "- `reports/top_multi_property_owners.csv`",
            "- `reports/multi_property_owner_detail.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
