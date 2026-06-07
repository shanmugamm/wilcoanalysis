from pathlib import Path
from dataclasses import dataclass
from io import BytesIO
import os
import sys
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from google.cloud import storage

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from owner_ml.features import organization_owner_mask


DEFAULT_GCS_PREFIX = "gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest"


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


def get_report_gcs_prefix() -> str:
    env_value = os.getenv("REPORT_SOURCE_GCS_PREFIX")
    if env_value:
        return env_value
    try:
        return st.secrets.get("REPORT_SOURCE_GCS_PREFIX", DEFAULT_GCS_PREFIX)
    except Exception:
        return DEFAULT_GCS_PREFIX


def is_multi_property_report(report: ReportFile) -> bool:
    return "multi_property" in report.name.lower()


st.set_page_config(page_title="Owner Reports", layout="wide")
st.title("Owner Reports Dashboard")

gcs_prefix = get_report_gcs_prefix()
report_dir = Path("reports")

try:
    files = list_gcs_reports(gcs_prefix) if gcs_prefix else list_local_reports(report_dir)
    active_gcs_prefix = gcs_prefix
    source_label = gcs_prefix if gcs_prefix else str(report_dir)
except Exception as exc:
    if report_dir.exists():
        st.warning(f"Could not load reports from `{gcs_prefix}`. Showing local reports instead.")
        files = list_local_reports(report_dir)
        active_gcs_prefix = None
        source_label = str(report_dir)
    else:
        st.error(f"Could not load reports from `{gcs_prefix}`.")
        st.exception(exc)
        st.stop()

md_files = [f for f in files if f.suffix in (".md", ".markdown")]
csv_files = [f for f in files if f.suffix == ".csv"]

st.sidebar.header("Controls")
st.sidebar.caption(f"Source: {source_label}")
mode = st.sidebar.radio("Show", ["Markdown", "CSV", "All"], index=0)


def show_markdown(report: ReportFile):
    text = read_report_bytes(report, active_gcs_prefix).decode("utf-8")
    st.markdown(f"### {report.name}")
    st.markdown(text, unsafe_allow_html=True)
    st.download_button("Download", text, file_name=report.name)


def show_csv(report: ReportFile):
    data = read_report_bytes(report, active_gcs_prefix)
    df = pd.read_csv(BytesIO(data))
    st.markdown(f"### {report.name}")
    if is_multi_property_report(report):
        exclude_organizations = st.checkbox(
            "Exclude organizations / LLCs",
            value=True,
            key=f"exclude_orgs_{report.name}",
        )
        if exclude_organizations:
            before_rows = len(df)
            df = df[~organization_owner_mask(df)].copy()
            st.caption(f"Filtered out {before_rows - len(df):,} organization-style rows.")
    st.dataframe(df)
    st.download_button("Download CSV", data, file_name=report.name)
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        col = st.selectbox(f"Plot numeric column ({report.name})", numeric.columns.tolist())
        st.bar_chart(df[col])


if mode == "Markdown":
    if not md_files:
        st.info("No markdown reports found.")
    else:
        choices = {f.name: f for f in md_files}
        choice = st.sidebar.selectbox("Choose markdown", list(choices))
        show_markdown(choices[choice])

elif mode == "CSV":
    if not csv_files:
        st.info("No CSV reports found.")
    else:
        choices = {f.name: f for f in csv_files}
        choice = st.sidebar.selectbox("Choose CSV", list(choices))
        show_csv(choices[choice])

else:  # All
    for md in md_files:
        show_markdown(md)
    for csv in csv_files:
        show_csv(csv)
