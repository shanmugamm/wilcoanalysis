from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from owner_ml.data import DEFAULT_DATA_PATH, read_owner_chunks, read_owner_sample


def is_corporate_entity(name: str) -> bool:
    """Check if a name appears to be a corporate entity."""
    corporate_suffixes = [
        "llc", "llc.", "ltd", "ltd.", "inc", "inc.", "corp", "corp.",
        "corporation", "company", "co.", "co", "llp", "llp.", "plc",
        "plc.", "gmbh", "ag", "sa", "sarl", "pty", "pty.", "limited",
        "partnership", "trust", "trustee", "association", "foundation",
        "group", "holdings", "enterprises", "industries", "international"
    ]
    name_lower = name.lower()
    return any(suffix in name_lower for suffix in corporate_suffixes)


def find_multi_property_owners(data_path: Path, min_properties: int = 2, owner_col: str | None = None, exclude_corporate: bool = True) -> dict[str, int]:
    """Find owners with multiple properties."""
    owner_property_count = defaultdict(int)
    
    for chunk in read_owner_chunks(data_path, chunksize=100000):
        # Use provided column or try to identify owner columns
        if owner_col is None or owner_col not in chunk.columns:
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
            for _, row in chunk.iterrows():
                owner = str(row[owner_col]).strip()
                if owner and owner.lower() not in ["", "(blank)", "nan"]:
                    # Filter out corporate entities if enabled
                    if exclude_corporate and is_corporate_entity(owner):
                        continue
                    owner_property_count[owner] += 1
    
    # Filter and sort
    multi_property_owners = {
        owner: count for owner, count in owner_property_count.items()
        if count >= min_properties
    }
    return dict(sorted(multi_property_owners.items(), key=lambda x: x[1], reverse=True))


def detect_suspicious_owners(data_path: Path, contamination: float = 0.05, owner_col: str | None = None) -> pd.DataFrame:
    """Detect suspicious owners using Isolation Forest."""
    all_owner_features = defaultdict(lambda: defaultdict(list))
    
    for chunk in read_owner_chunks(data_path, chunksize=100000):
        # Find owner column
        if owner_col is None or owner_col not in chunk.columns:
            current_owner_col = None
            for col in ["OwnerName", "Owner", "Name", "PrimaryOwner"]:
                if col in chunk.columns:
                    current_owner_col = col
                    break
            if current_owner_col is None:
                for col in chunk.columns:
                    if chunk[col].dtype == "object":
                        current_owner_col = col
                        break
        else:
            current_owner_col = owner_col
        
        if not current_owner_col:
            continue
        
        # Extract features
        for _, row in chunk.iterrows():
            owner = str(row[current_owner_col]).strip()
            if not owner or owner.lower() in ["", "(blank)", "nan"]:
                continue
            
            features = {}
            features["property_count"] = 1
            features["ownership_pct"] = float(row["PercentOwnership"]) if "PercentOwnership" in row and pd.notna(row["PercentOwnership"]) else 0
            features["is_primary"] = 1 if "PrimaryOwner" in row and pd.notna(row["PrimaryOwner"]) and row["PrimaryOwner"] == 1 else 0
            features["is_undeliverable"] = 1 if "IsUndeliverable" in row and pd.notna(row["IsUndeliverable"]) and row["IsUndeliverable"] == 1 else 0
            features["has_address_change"] = 1 if "AddressChgReasonDesc" in row and str(row["AddressChgReasonDesc"]).strip().lower() not in ["", "(blank)", "nan"] else 0
            
            for key, value in features.items():
                all_owner_features[owner][key].append(value)
    
    # Aggregate features
    feature_matrix = []
    owner_names = []
    
    for owner, feat_dict in all_owner_features.items():
        row = {}
        for key, values in feat_dict.items():
            if key == "property_count":
                row[key] = sum(values)
            else:
                row[key] = np.mean(values)
        
        row["avg_ownership_per_property"] = row.get("ownership_pct", 0) / max(row.get("property_count", 1), 1)
        row["undeliverable_ratio"] = row.get("is_undeliverable", 0) / max(row.get("property_count", 1), 1)
        row["address_change_ratio"] = row.get("has_address_change", 0) / max(row.get("property_count", 1), 1)
        
        feature_matrix.append(row)
        owner_names.append(owner)
    
    df = pd.DataFrame(feature_matrix)
    df["owner"] = owner_names
    
    feature_cols = [col for col in df.columns if col != "owner"]
    X = df[feature_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    model.fit(X_scaled)
    
    # Predict
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_suspicious"] = model.predict(X_scaled)
    df["is_suspicious"] = df["is_suspicious"].map({1: 0, -1: 1})
    
    return df


def main() -> None:
    st.set_page_config(page_title="Owner Data Viewer", layout="wide")

    st.title("Owner Data Viewer")

    # Sidebar for file selection
    st.sidebar.header("Settings")
    data_path = st.sidebar.text_input("Data Path", value=str(DEFAULT_DATA_PATH))
    nrows = st.sidebar.number_input("Sample Rows", min_value=1000, max_value=100000, value=50000)
    min_properties = st.sidebar.number_input("Min Properties", min_value=2, max_value=10, value=2)

    path = Path(data_path)
    if not path.exists():
        st.error(f"File not found: {data_path}")
        return

    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["Data Viewer", "Multi-Property Owners", "Suspicious Owner Detection"])

    with tab1:
        # Load data
        with st.spinner("Loading data..."):
            df = read_owner_sample(path, nrows=nrows)

        # Display basic info
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns):,}")
        col3.metric("File Size", f"{path.stat().st_mtime:,}")

        # Display data table
        st.header("Data Preview")
        st.dataframe(df, use_container_width=True)

        # Column statistics
        st.header("Column Statistics")
        col_to_view = st.selectbox("Select Column", df.columns)
        st.write(f"### {col_to_view}")

        col_data = df[col_to_view]
        st.metric("Non-null Count", f"{col_data.notna().sum():,}")
        st.metric("Null Count", f"{col_data.isna().sum():,}")

        if col_data.dtype in ["int64", "float64"]:
            st.write("**Statistics:**")
            st.write(col_data.describe())
        else:
            st.write("**Value Counts (Top 20):**")
            st.write(col_data.value_counts().head(20))

        # Data types
        st.header("Data Types")
        dtypes_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str),
            "Non-Null": df.notna().sum(),
            "Null": df.isna().sum()
        })
        st.dataframe(dtypes_df, use_container_width=True)

    with tab2:
        st.header("Multi-Property Owners Analysis")
        
        # Load sample to get column names
        with st.spinner("Loading column names..."):
            sample_df = read_owner_sample(path, nrows=1000)
        
        # Column selector
        owner_col = st.selectbox("Select Owner Column", sample_df.columns, index=0)
        
        # Filter toggle
        exclude_corporate = st.checkbox("Exclude Corporate Entities (LLC, Corp, Inc, etc.)", value=True)
        
        with st.spinner("Analyzing owners..."):
            multi_property_owners = find_multi_property_owners(path, min_properties, owner_col, exclude_corporate)
        
        st.metric("Multi-Property Owners", f"{len(multi_property_owners):,}")
        
        if multi_property_owners:
            st.write(f"### Owners with {min_properties}+ Properties")
            
            # Create DataFrame for display
            owners_df = pd.DataFrame({
                "Owner": list(multi_property_owners.keys()),
                "Properties": list(multi_property_owners.values())
            })
            st.dataframe(owners_df, use_container_width=True)
            
            # Summary statistics
            st.write("### Summary")
            col1, col2 = st.columns(2)
            col1.metric("Total Properties", f"{sum(multi_property_owners.values()):,}")
            col2.metric("Avg Properties per Owner", f"{sum(multi_property_owners.values()) / len(multi_property_owners):.1f}")
        else:
            st.info(f"No owners found with {min_properties}+ properties")

    with tab3:
        st.header("Suspicious Owner Detection")
        st.write("Uses Isolation Forest anomaly detection to identify potentially suspicious owners based on patterns.")
        
        # Load sample to get column names
        with st.spinner("Loading column names..."):
            sample_df = read_owner_sample(path, nrows=1000)
        
        # Column selector
        owner_col = st.selectbox("Select Owner Column", sample_df.columns, index=0, key="suspicious_owner_col")
        
        # Contamination slider
        contamination = st.slider("Expected Suspicious Rate (%)", min_value=1, max_value=20, value=5) / 100
        
        if st.button("Run Detection"):
            with st.spinner("Analyzing owners for suspicious patterns..."):
                results_df = detect_suspicious_owners(path, contamination, owner_col)
            
            suspicious = results_df[results_df["is_suspicious"] == 1].sort_values("anomaly_score", ascending=True)
            
            st.metric("Suspicious Owners Found", f"{len(suspicious):,}")
            st.metric("Total Owners Analyzed", f"{len(results_df):,}")
            
            if len(suspicious) > 0:
                st.write(f"### Top Suspicious Owners")
                
                display_cols = ["owner", "anomaly_score", "property_count", "avg_ownership_per_property", "undeliverable_ratio"]
                available_cols = [col for col in display_cols if col in suspicious.columns]
                st.dataframe(suspicious[available_cols].head(20), use_container_width=True)
                
                st.write("### Feature Importance")
                st.info("Lower anomaly scores indicate more suspicious behavior. Features considered: property count, ownership percentage, undeliverable flags, and address change patterns.")
            else:
                st.info("No suspicious owners detected with current settings. Try increasing the expected suspicious rate.")


if __name__ == "__main__":
    main()
