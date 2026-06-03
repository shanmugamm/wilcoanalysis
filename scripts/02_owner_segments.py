from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owner_ml.data import DEFAULT_DATA_PATH, read_owner_sample
from owner_ml.features import add_owner_features


FEATURE_COLUMNS = [
    "State",
    "City",
    "ZIP5",
    "AddressChgReasonDesc",
    "AddressTypeKey",
    "OwnerTypeGuess",
    "OwnerProfileSegment",
    "MailingGeo",
    "IsOutOfAreaMailing",
    "PrimaryOwner",
    "PercentOwnership",
    "IsUndeliverable",
    "ExemptionCount",
    "Exemption_HS",
    "Exemption_OV",
    "Exemption_DP",
    "Exemption_DV",
    "Exemption_AG",
    "Exemption_CBL",
    "AddressAgeDays",
    "TaxingUnitCount",
]


def percent(value: float) -> str:
    return f"{value:.1%}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a starter owner segmentation model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to owner CSV.")
    parser.add_argument("--sample-rows", type=int, default=50_000, help="Rows to sample from the CSV.")
    parser.add_argument("--full", action="store_true", help="Use the full CSV instead of a row sample.")
    parser.add_argument("--clusters", type=int, default=8, help="Number of owner segments.")
    parser.add_argument("--output-dir", default="reports", help="Directory for model outputs.")
    return parser.parse_args()


def main() -> None:
    try:
        import pandas as pd
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Run: python -m pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    sample_rows = None if args.full else args.sample_rows
    df = add_owner_features(read_owner_sample(args.data, nrows=sample_rows))

    model_df = df[[column for column in FEATURE_COLUMNS if column in df.columns]].copy()
    categorical_columns = model_df.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in model_df.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "cluster",
                MiniBatchKMeans(
                    n_clusters=args.clusters,
                    random_state=42,
                    batch_size=2048,
                    n_init="auto",
                ),
            ),
        ]
    )
    df["OwnerSegment"] = pipeline.fit_predict(model_df)

    summary = (
        df.groupby("OwnerSegment")
        .agg(
            rows=("OwnerSegment", "size"),
            owner_type=("OwnerTypeGuess", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            profile_segment=(
                "OwnerProfileSegment",
                lambda value: value.mode().iat[0] if not value.mode().empty else "",
            ),
            mailing_geo=("MailingGeo", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            top_state=("State", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            top_city=("City", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            out_of_area_rate=("IsOutOfAreaMailing", "mean"),
            homestead_rate=("Exemption_HS", "mean"),
            ag_rate=("Exemption_AG", "mean"),
            undeliverable_rate=("IsUndeliverable", "mean"),
            avg_percent_ownership=("PercentOwnership", "mean"),
            avg_exemption_count=("ExemptionCount", "mean"),
            avg_address_age_days=("AddressAgeDays", "mean"),
        )
        .sort_values("rows", ascending=False)
        .reset_index()
    )

    profile_summary = (
        df.groupby("OwnerProfileSegment")
        .agg(
            rows=("OwnerProfileSegment", "size"),
            top_owner_type=("OwnerTypeGuess", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            top_mailing_geo=("MailingGeo", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            top_state=("State", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            top_city=("City", lambda value: value.mode().iat[0] if not value.mode().empty else ""),
            out_of_area_rate=("IsOutOfAreaMailing", "mean"),
            homestead_rate=("Exemption_HS", "mean"),
            ag_rate=("Exemption_AG", "mean"),
            undeliverable_rate=("IsUndeliverable", "mean"),
            avg_percent_ownership=("PercentOwnership", "mean"),
            avg_exemption_count=("ExemptionCount", "mean"),
        )
        .sort_values("rows", ascending=False)
        .reset_index()
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "owner_segments.csv", index=False)
    summary.to_csv(output_dir / "owner_segment_summary.csv", index=False)
    profile_summary.to_csv(output_dir / "owner_profile_segment_summary.csv", index=False)
    report_path = output_dir / "owner_segmentation_report.md"
    report_path.write_text(
        build_report(len(df), args.clusters, summary, profile_summary),
        encoding="utf-8",
    )

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print("ML cluster summary")
        print(summary)
        print()
        print("Interpretable profile summary")
        print(profile_summary)
    print(f"Wrote {output_dir / 'owner_segments.csv'}")
    print(f"Wrote {output_dir / 'owner_segment_summary.csv'}")
    print(f"Wrote {output_dir / 'owner_profile_segment_summary.csv'}")
    print(f"Wrote {report_path}")


def build_report(
    total_rows: int,
    clusters: int,
    cluster_summary: "pd.DataFrame",
    profile_summary: "pd.DataFrame",
) -> str:
    lines = [
        "# Owner Segmentation Report",
        "",
        f"- Rows analyzed: `{total_rows:,}`",
        f"- ML clusters: `{clusters}`",
        "",
        "## Interpretable Profile Segments",
        "",
    ]

    for _, row in profile_summary.iterrows():
        lines.append(
            f"- `{row['OwnerProfileSegment']}`: `{int(row['rows']):,}` records, "
            f"top geography `{row['top_mailing_geo']}`, "
            f"out-of-area `{percent(row['out_of_area_rate'])}`, "
            f"homestead `{percent(row['homestead_rate'])}`"
        )

    lines.extend(["", "## ML Cluster Themes", ""])
    for _, row in cluster_summary.iterrows():
        lines.append(
            f"- Cluster `{int(row['OwnerSegment'])}`: `{int(row['rows']):,}` records, "
            f"mostly `{row['profile_segment']}`, geography `{row['mailing_geo']}`, "
            f"out-of-area `{percent(row['out_of_area_rate'])}`, "
            f"homestead `{percent(row['homestead_rate'])}`, ag `{percent(row['ag_rate'])}`"
        )

    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `reports/owner_segments.csv`",
            "- `reports/owner_segment_summary.csv`",
            "- `reports/owner_profile_segment_summary.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
