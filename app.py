from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from google.cloud import storage

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from owner_ml.features import organization_owner_mask


DEFAULT_GCS_PREFIX = "gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest"
DEFAULT_PAGE_SIZE = 500

REPORT_LABELS = {
    "multi_property_owner_summary.csv": "Multi-Property Owner Summary",
    "multi_property_owner_detail.csv": "Multi-Property Owner Detail",
    "top_multi_property_owners.csv": "Top Multi-Property Owners",
    "owner_type_summary.csv": "Owner Type Summary",
    "mailing_geo_summary.csv": "Mailing Geography Summary",
    "owner_type_by_geo.csv": "Owner Type By Geography",
    "owner_segments.csv": "Owner Segments",
    "owner_segment_summary.csv": "ML Segment Summary",
    "owner_profile_segment_summary.csv": "Profile Segment Summary",
    "owner_profile.md": "Owner Profile",
    "owner_type_geo_analysis.md": "Owner Type And Geography",
    "multi_property_owners_report.md": "Multi-Property Owners Report",
    "owner_segmentation_report.md": "Owner Segmentation Report",
    "pipeline_manifest.md": "Pipeline Manifest",
}

PREFERRED_TABLES = [
    "multi_property_owner_summary.csv",
    "top_multi_property_owners.csv",
    "multi_property_owner_detail.csv",
    "owner_type_summary.csv",
    "mailing_geo_summary.csv",
    "owner_type_by_geo.csv",
    "owner_segment_summary.csv",
    "owner_profile_segment_summary.csv",
    "owner_segments.csv",
]


@dataclass(frozen=True)
class ReportFile:
    name: str
    suffix: str
    source: str


def parse_gcs_prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected a GCS prefix like gs://bucket/path, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


@st.cache_data(ttl=300)
def list_gcs_reports(gcs_prefix: str) -> list[ReportFile]:
    bucket_name, prefix = parse_gcs_prefix(gcs_prefix)
    client = storage.Client()
    reports = []
    for blob in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
        if blob.name.endswith("/"):
            continue
        name = Path(blob.name).name
        reports.append(ReportFile(name=name, suffix=Path(name).suffix.lower(), source=blob.name))
    return sorted(reports, key=lambda report: report.name)


@st.cache_data(ttl=300)
def read_gcs_report(gcs_prefix: str, blob_name: str) -> bytes:
    bucket_name, _ = parse_gcs_prefix(gcs_prefix)
    client = storage.Client()
    return client.bucket(bucket_name).blob(blob_name).download_as_bytes()


def list_local_reports(report_dir: Path) -> list[ReportFile]:
    return [
        ReportFile(name=path.name, suffix=path.suffix.lower(), source=str(path))
        for path in sorted(report_dir.iterdir())
        if path.is_file()
    ]


def read_report_bytes(report: ReportFile, gcs_prefix: str | None) -> bytes:
    if gcs_prefix:
        return read_gcs_report(gcs_prefix, report.source)
    return Path(report.source).read_bytes()


@st.cache_data(ttl=300)
def load_csv_report(report: ReportFile, gcs_prefix: str | None) -> pd.DataFrame:
    data = read_report_bytes(report, gcs_prefix)
    return pd.read_csv(BytesIO(data))


def get_report_gcs_prefix() -> str:
    env_value = os.getenv("REPORT_SOURCE_GCS_PREFIX")
    if env_value:
        return env_value
    try:
        return st.secrets.get("REPORT_SOURCE_GCS_PREFIX", DEFAULT_GCS_PREFIX)
    except Exception:
        return DEFAULT_GCS_PREFIX


def display_name(report: ReportFile) -> str:
    return REPORT_LABELS.get(report.name, report.name.replace("_", " ").replace(".csv", "").title())


def report_category(report: ReportFile) -> str:
    name = report.name.lower()
    if "multi_property" in name:
        return "Multi-Property"
    if "segment" in name:
        return "Segmentation"
    if "geo" in name or "state" in name or "city" in name:
        return "Geography"
    if "owner_type" in name or "profile" in name:
        return "Owner Profile"
    return "Other"


def is_multi_property_report(report: ReportFile) -> bool:
    return "multi_property" in report.name.lower()


def find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def apply_text_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return df
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    if text_columns.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for column in text_columns:
        mask = mask | df[column].fillna("").astype(str).str.contains(query, case=False, regex=False)
    return df[mask]


def apply_select_filter(df: pd.DataFrame, column: str | None, values: list[str]) -> pd.DataFrame:
    if not column or not values:
        return df
    return df[df[column].fillna("").astype(str).isin(values)]


def apply_numeric_min(df: pd.DataFrame, column: str | None, minimum: float | None) -> pd.DataFrame:
    if not column or minimum is None:
        return df
    numeric = pd.to_numeric(df[column], errors="coerce")
    return df[numeric.ge(minimum)]


def sorted_reports(reports: list[ReportFile], preferred: list[str]) -> list[ReportFile]:
    rank = {name: index for index, name in enumerate(preferred)}
    return sorted(reports, key=lambda report: (rank.get(report.name, 999), display_name(report)))


def report_by_name(reports: list[ReportFile], name: str) -> ReportFile | None:
    return next((report for report in reports if report.name == name), None)


def metric_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_header(source_label: str) -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem;}
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 14px;
        }
        div[data-testid="stMetricLabel"] {font-size: 0.82rem;}
        .report-caption {color: #64748b; font-size: 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Owner Reports Dashboard")
    st.caption(f"Report source: {source_label}")


def render_overview(csv_reports: list[ReportFile], md_reports: list[ReportFile], gcs_prefix: str | None) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tables", len(csv_reports))
    c2.metric("Narrative Reports", len(md_reports))

    multi_summary = next((report for report in csv_reports if report.name == "multi_property_owner_summary.csv"), None)
    if multi_summary:
        df = load_csv_report(multi_summary, gcs_prefix)
        c3.metric("Multi-Property Owners", f"{len(df):,}")
        property_col = find_column(df, ["distinct_properties"])
        if property_col:
            c4.metric("Top Property Count", metric_value(pd.to_numeric(df[property_col], errors="coerce").max()))
        else:
            c4.metric("Rows Loaded", f"{len(df):,}")
    else:
        c3.metric("Multi-Property Owners", "-")
        c4.metric("Rows Loaded", "-")

    st.subheader("Available Report Groups")
    category_rows = []
    for report in csv_reports + md_reports:
        category_rows.append(
            {
                "category": report_category(report),
                "report": display_name(report),
                "file": report.name,
                "type": "table" if report.suffix == ".csv" else "markdown",
            }
        )
    st.dataframe(pd.DataFrame(category_rows), hide_index=True, use_container_width=True)


def render_markdown_report(report: ReportFile, gcs_prefix: str | None) -> None:
    text = read_report_bytes(report, gcs_prefix).decode("utf-8")
    st.subheader(display_name(report))
    st.download_button("Download Markdown", text, file_name=report.name)
    st.markdown(text, unsafe_allow_html=True)


def build_owner_label(row: pd.Series) -> str:
    owner_name = row.get("full_name", row.get("FullName", "Unknown owner"))
    city = row.get("city", row.get("City", ""))
    state = row.get("state", row.get("State", ""))
    count = row.get("distinct_properties", "")
    location = ", ".join([value for value in [str(city), str(state)] if value and value != "nan"])
    suffix = f" - {location}" if location else ""
    return f"{owner_name}{suffix} ({count} properties)"


def selected_dataframe_rows(event: object) -> list[int]:
    if hasattr(event, "selection"):
        selection = event.selection
        if isinstance(selection, dict):
            return selection.get("rows", [])
        return getattr(selection, "rows", [])
    if isinstance(event, dict):
        selection = event.get("selection", {})
        return selection.get("rows", []) if isinstance(selection, dict) else []
    return []


def render_owner_drilldown(csv_reports: list[ReportFile], gcs_prefix: str | None) -> None:
    summary_report = report_by_name(csv_reports, "multi_property_owner_summary.csv")
    detail_report = report_by_name(csv_reports, "multi_property_owner_detail.csv")

    if not summary_report or not detail_report:
        st.info("Owner drilldown needs `multi_property_owner_summary.csv` and `multi_property_owner_detail.csv`.")
        return

    summary = load_csv_report(summary_report, gcs_prefix)
    detail = load_csv_report(detail_report, gcs_prefix)

    st.subheader("Unique Property Owners")
    st.caption("Select an owner row to see every property tied to that owner.")

    filtered = summary.copy()
    with st.expander("Owner Filters", expanded=True):
        query = st.text_input("Search owner, address, city, or state", placeholder="Try MORROW, AUSTIN, TX...")
        filtered = apply_text_search(filtered, query)

        exclude_organizations = st.checkbox("Exclude organizations / LLCs", value=True)
        if exclude_organizations:
            before_rows = len(filtered)
            filtered = filtered[~organization_owner_mask(filtered)].copy()
            st.caption(f"Filtered out {before_rows - len(filtered):,} organization-style owners.")

        filter_cols = st.columns(4)
        owner_type_col = find_column(filtered, ["owner_type", "OwnerTypeGuess"])
        geo_col = find_column(filtered, ["mailing_geo", "MailingGeo"])
        city_col = find_column(filtered, ["city", "City"])
        property_count_col = find_column(filtered, ["distinct_properties"])

        if owner_type_col:
            options = sorted(filtered[owner_type_col].fillna("").astype(str).unique())
            selected = filter_cols[0].multiselect("Owner type", options)
            filtered = apply_select_filter(filtered, owner_type_col, selected)

        if geo_col:
            options = sorted(filtered[geo_col].fillna("").astype(str).unique())
            selected = filter_cols[1].multiselect("Mailing geography", options)
            filtered = apply_select_filter(filtered, geo_col, selected)

        if city_col:
            city_counts = filtered[city_col].fillna("").astype(str).value_counts()
            selected = filter_cols[2].multiselect("City", city_counts.head(50).index.tolist())
            filtered = apply_select_filter(filtered, city_col, selected)

        if property_count_col:
            property_counts = pd.to_numeric(filtered[property_count_col], errors="coerce")
            min_count = int(property_counts.min()) if property_counts.notna().any() else 2
            max_count = int(property_counts.max()) if property_counts.notna().any() else min_count
            if min_count < max_count:
                selected_min = filter_cols[3].slider(
                    "Min properties",
                    min_value=min_count,
                    max_value=max_count,
                    value=min_count,
                )
                filtered = filtered[property_counts.ge(selected_min)]

    property_count_col = find_column(filtered, ["distinct_properties"])
    if property_count_col:
        filtered = filtered.sort_values(property_count_col, ascending=False, kind="stable")

    m1, m2, m3 = st.columns(3)
    m1.metric("Owners", f"{len(filtered):,}")
    m2.metric("Total Property Links", metric_value(pd.to_numeric(filtered.get("distinct_properties", 0), errors="coerce").sum()))
    m3.metric("Top Owner Count", metric_value(pd.to_numeric(filtered.get("distinct_properties", 0), errors="coerce").max()))

    owner_columns = [
        column
        for column in [
            "full_name",
            "distinct_properties",
            "owner_rows",
            "mailing_address",
            "city",
            "state",
            "owner_type",
            "profile_segment",
            "mailing_geo",
            "owner_key",
        ]
        if column in filtered
    ]
    owner_view = filtered[owner_columns].head(1000).reset_index(drop=True)
    if owner_view.empty:
        st.info("No owners match the current filters.")
        return

    event = st.dataframe(
        owner_view,
        hide_index=True,
        use_container_width=True,
        height=420,
        selection_mode="single-row",
        on_select="rerun",
        key="owner_drilldown_table",
    )

    selected_owner_key = None
    selected_rows = selected_dataframe_rows(event)
    if selected_rows:
        selected_owner_key = owner_view.iloc[selected_rows[0]].get("owner_key")

    if not selected_owner_key:
        labels = {build_owner_label(row): row.get("owner_key") for _, row in owner_view.head(200).iterrows()}
        selected_label = st.selectbox("Or choose an owner", list(labels))
        selected_owner_key = labels[selected_label]

    if not selected_owner_key:
        return

    detail_owner_col = find_column(detail, ["OwnerKey", "owner_key"])
    if not detail_owner_col:
        st.warning("The detail report does not include an owner key column.")
        return

    owner_properties = detail[detail[detail_owner_col].astype(str).eq(str(selected_owner_key))].copy()
    st.divider()
    st.subheader("Properties Owned")

    selected_owner = filtered[filtered["owner_key"].astype(str).eq(str(selected_owner_key))]
    if not selected_owner.empty:
        row = selected_owner.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Owner", row.get("full_name", ""))
        c2.metric("Properties", metric_value(row.get("distinct_properties", len(owner_properties))))
        c3.metric("Mailing Geo", row.get("mailing_geo", ""))

    display_columns = [
        column
        for column in [
            "PropertyID",
            "QuickRefID",
            "OwnerPropertyNumber",
            "Address1",
            "Address2",
            "Address3",
            "PercentOwnership",
            "PrimaryOwner",
            "ExemptionList",
            "OwnerProfileSegment",
            "MailingGeo",
        ]
        if column in owner_properties
    ]
    if not display_columns:
        display_columns = owner_properties.columns.tolist()

    st.dataframe(owner_properties[display_columns], hide_index=True, use_container_width=True)
    st.download_button(
        "Download Owner Properties",
        owner_properties.to_csv(index=False).encode("utf-8"),
        file_name=f"properties_{selected_owner_key}.csv",
        mime="text/csv",
    )


def filter_dataframe(df: pd.DataFrame, report: ReportFile) -> pd.DataFrame:
    filtered = df.copy()

    with st.expander("Filters", expanded=True):
        query = st.text_input("Search all text columns", placeholder="Try owner name, city, state, segment...")
        filtered = apply_text_search(filtered, query)

        if is_multi_property_report(report):
            exclude_organizations = st.checkbox("Exclude organizations / LLCs", value=True)
            if exclude_organizations:
                before_rows = len(filtered)
                filtered = filtered[~organization_owner_mask(filtered)].copy()
                st.caption(f"Filtered out {before_rows - len(filtered):,} organization-style rows.")

        filter_cols = st.columns(3)
        owner_type_col = find_column(filtered, ["owner_type", "OwnerTypeGuess"])
        geo_col = find_column(filtered, ["mailing_geo", "MailingGeo"])
        city_col = find_column(filtered, ["city", "City"])

        if owner_type_col:
            options = sorted(filtered[owner_type_col].fillna("").astype(str).unique())
            selected = filter_cols[0].multiselect("Owner type", options)
            filtered = apply_select_filter(filtered, owner_type_col, selected)

        if geo_col:
            options = sorted(filtered[geo_col].fillna("").astype(str).unique())
            selected = filter_cols[1].multiselect("Mailing geography", options)
            filtered = apply_select_filter(filtered, geo_col, selected)

        if city_col:
            city_counts = filtered[city_col].fillna("").astype(str).value_counts()
            options = city_counts.head(50).index.tolist()
            selected = filter_cols[2].multiselect("City", options)
            filtered = apply_select_filter(filtered, city_col, selected)

        numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            selected_numeric = st.selectbox("Numeric threshold", ["None"] + numeric_cols, index=0)
            if selected_numeric != "None":
                min_value = float(pd.to_numeric(filtered[selected_numeric], errors="coerce").min())
                max_value = float(pd.to_numeric(filtered[selected_numeric], errors="coerce").max())
                if pd.notna(min_value) and pd.notna(max_value) and min_value < max_value:
                    minimum = st.slider(
                        f"Minimum {selected_numeric}",
                        min_value=min_value,
                        max_value=max_value,
                        value=min_value,
                    )
                    filtered = apply_numeric_min(filtered, selected_numeric, minimum)
                else:
                    st.caption(f"`{selected_numeric}` has no numeric range to filter.")

    return filtered


def render_table_report(report: ReportFile, gcs_prefix: str | None) -> None:
    df = load_csv_report(report, gcs_prefix)
    st.subheader(display_name(report))
    st.caption(report.name)

    filtered = filter_dataframe(df, report)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(filtered):,}", delta=f"{len(filtered) - len(df):,}")
    c2.metric("Columns", len(filtered.columns))
    numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
    c3.metric("Numeric Fields", len(numeric_cols))

    controls = st.columns([2, 1, 1])
    selected_columns = controls[0].multiselect(
        "Columns",
        filtered.columns.tolist(),
        default=filtered.columns.tolist()[: min(12, len(filtered.columns))],
    )
    page_size = controls[1].number_input("Rows to show", min_value=50, max_value=5000, value=DEFAULT_PAGE_SIZE, step=50)
    sort_column = controls[2].selectbox("Sort by", ["None"] + filtered.columns.tolist())

    view_df = filtered
    if sort_column != "None":
        ascending = st.toggle("Ascending sort", value=False)
        view_df = view_df.sort_values(sort_column, ascending=ascending, kind="stable")
    if selected_columns:
        view_df = view_df[selected_columns]

    st.dataframe(view_df.head(int(page_size)), hide_index=True, use_container_width=True)
    st.download_button(
        "Download Filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"filtered_{report.name}",
        mime="text/csv",
    )

    if numeric_cols:
        chart_col = st.selectbox("Chart numeric column", ["None"] + numeric_cols)
        if chart_col != "None":
            st.bar_chart(filtered[chart_col].head(100))


st.set_page_config(page_title="Owner Reports", layout="wide")

gcs_prefix = get_report_gcs_prefix()
report_dir = Path("reports")

try:
    files = list_gcs_reports(gcs_prefix) if gcs_prefix else list_local_reports(report_dir)
    active_gcs_prefix = gcs_prefix
    source_label = gcs_prefix if gcs_prefix else str(report_dir)
except Exception as exc:
    if report_dir.exists():
        files = list_local_reports(report_dir)
        active_gcs_prefix = None
        source_label = str(report_dir)
        st.warning(f"Could not load reports from `{gcs_prefix}`. Showing local reports instead.")
    else:
        st.error(f"Could not load reports from `{gcs_prefix}`.")
        st.exception(exc)
        st.stop()

csv_reports = sorted_reports([f for f in files if f.suffix == ".csv"], PREFERRED_TABLES)
md_reports = sorted_reports([f for f in files if f.suffix in (".md", ".markdown")], [])

render_header(source_label)

with st.sidebar:
    st.header("Reports")
    section = st.radio(
        "View",
        ["Overview", "Owner Drilldown", "Tables", "Narratives"],
        label_visibility="collapsed",
    )

if section == "Overview":
    render_overview(csv_reports, md_reports, active_gcs_prefix)
elif section == "Owner Drilldown":
    render_owner_drilldown(csv_reports, active_gcs_prefix)
elif section == "Tables":
    if not csv_reports:
        st.info("No CSV reports found.")
    else:
        categories = ["All"] + sorted({report_category(report) for report in csv_reports})
        category = st.sidebar.selectbox("Category", categories)
        visible_reports = [
            report for report in csv_reports if category == "All" or report_category(report) == category
        ]
        choices = {display_name(report): report for report in visible_reports}
        selected = st.sidebar.selectbox("Table", list(choices))
        render_table_report(choices[selected], active_gcs_prefix)
else:
    if not md_reports:
        st.info("No markdown reports found.")
    else:
        choices = {display_name(report): report for report in md_reports}
        selected = st.sidebar.selectbox("Report", list(choices))
        render_markdown_report(choices[selected], active_gcs_prefix)
