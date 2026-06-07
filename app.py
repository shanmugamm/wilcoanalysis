from pathlib import Path
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Owner Reports", layout="wide")
st.title("Owner Reports Dashboard")

report_dir = Path("reports")
if not report_dir.exists():
    st.error("No `reports/` directory found in the workspace.")
    st.stop()

files = sorted(report_dir.iterdir())
md_files = [f for f in files if f.suffix.lower() in (".md", ".markdown")]
csv_files = [f for f in files if f.suffix.lower() == ".csv"]

st.sidebar.header("Controls")
mode = st.sidebar.radio("Show", ["Markdown", "CSV", "All"], index=0)

def show_markdown(path: Path):
    text = path.read_text(encoding="utf-8")
    st.markdown(f"### {path.name}")
    st.markdown(text, unsafe_allow_html=True)
    st.download_button("Download", text, file_name=path.name)

def show_csv(path: Path):
    df = pd.read_csv(path)
    st.markdown(f"### {path.name}")
    st.dataframe(df)
    st.download_button("Download CSV", path.read_bytes(), file_name=path.name)
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        col = st.selectbox(f"Plot numeric column ({path.name})", numeric.columns.tolist())
        st.bar_chart(df[col])

if mode == "Markdown":
    if not md_files:
        st.info("No markdown reports found in `reports/`.")
    else:
        choice = st.sidebar.selectbox("Choose markdown", [f.name for f in md_files])
        show_markdown(report_dir / choice)

elif mode == "CSV":
    if not csv_files:
        st.info("No CSV reports found in `reports/`.")
    else:
        choice = st.sidebar.selectbox("Choose CSV", [f.name for f in csv_files])
        show_csv(report_dir / choice)

else:  # All
    for md in md_files:
        show_markdown(md)
    for csv in csv_files:
        show_csv(csv)
