from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owner_ml.data import DEFAULT_DATA_PATH, read_owner_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect suspicious/illegal owners using anomaly detection.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to owner CSV.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected proportion of outliers.")
    parser.add_argument("--output", default="reports/suspicious_owners.csv", help="Output CSV path.")
    return parser.parse_args()


def extract_owner_features(chunk: pd.DataFrame, owner_col: str) -> dict[str, dict]:
    """Extract features for each owner from a data chunk."""
    owner_features = defaultdict(lambda: defaultdict(list))
    
    for _, row in chunk.iterrows():
        owner = str(row[owner_col]).strip()
        if not owner or owner.lower() in ["", "(blank)", "nan"]:
            continue
        
        # Extract features
        features = {}
        
        # Property count per owner (will be aggregated later)
        features["property_count"] = 1
        
        # Ownership percentage if available
        if "PercentOwnership" in row:
            features["ownership_pct"] = float(row["PercentOwnership"]) if pd.notna(row["PercentOwnership"]) else 0
        else:
            features["ownership_pct"] = 0
        
        # Primary owner flag
        if "PrimaryOwner" in row:
            features["is_primary"] = 1 if pd.notna(row["PrimaryOwner"]) and row["PrimaryOwner"] == 1 else 0
        else:
            features["is_primary"] = 0
        
        # Undeliverable flag
        if "IsUndeliverable" in row:
            features["is_undeliverable"] = 1 if pd.notna(row["IsUndeliverable"]) and row["IsUndeliverable"] == 1 else 0
        else:
            features["is_undeliverable"] = 0
        
        # Address change reason (encode as categorical)
        if "AddressChgReasonDesc" in row:
            reason = str(row["AddressChgReasonDesc"]).strip()
            features["has_address_change"] = 1 if reason and reason.lower() not in ["", "(blank)", "nan"] else 0
        else:
            features["has_address_change"] = 0
        
        # Store features per owner
        for key, value in features.items():
            owner_features[owner][key].append(value)
    
    # Aggregate features per owner
    aggregated = {}
    for owner, feat_dict in owner_features.items():
        agg = {}
        for key, values in feat_dict.items():
            if key == "property_count":
                agg[key] = sum(values)
            elif key in ["ownership_pct"]:
                agg[f"{key}_mean"] = np.mean(values)
                agg[f"{key}_std"] = np.std(values) if len(values) > 1 else 0
                agg[f"{key}_max"] = np.max(values)
            else:
                agg[key] = sum(values)
        aggregated[owner] = agg
    
    return aggregated


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("Loading data and extracting features...")
    
    # Collect all owner features
    all_owner_features = defaultdict(lambda: defaultdict(list))
    
    for chunk in read_owner_chunks(data_path, chunksize=args.chunksize):
        # Find owner column
        owner_col = None
        for col in ["OwnerName", "Owner", "Name", "PrimaryOwner"]:
            if col in chunk.columns:
                owner_col = col
                break
        
        if owner_col is None:
            for col in chunk.columns:
                if chunk[col].dtype == "object":
                    owner_col = col
                    break
        
        if not owner_col:
            continue
        
        # Extract features from this chunk
        chunk_features = extract_owner_features(chunk, owner_col)
        
        # Aggregate across chunks
        for owner, features in chunk_features.items():
            for key, value in features.items():
                all_owner_features[owner][key].append(value)
    
    # Final aggregation
    print("Aggregating features...")
    feature_matrix = []
    owner_names = []
    
    for owner, feat_dict in all_owner_features.items():
        row = {}
        for key, values in feat_dict.items():
            if key == "property_count":
                row[key] = sum(values)
            else:
                row[key] = np.mean(values)
        
        # Add derived features
        row["avg_ownership_per_property"] = row.get("ownership_pct_mean", 0) / max(row.get("property_count", 1), 1)
        row["undeliverable_ratio"] = row.get("is_undeliverable", 0) / max(row.get("property_count", 1), 1)
        row["address_change_ratio"] = row.get("has_address_change", 0) / max(row.get("property_count", 1), 1)
        
        feature_matrix.append(row)
        owner_names.append(owner)
    
    # Create DataFrame
    df = pd.DataFrame(feature_matrix)
    df["owner"] = owner_names
    
    print(f"Total owners: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    
    # Prepare for ML
    feature_cols = [col for col in df.columns if col != "owner"]
    X = df[feature_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    print("Training Isolation Forest model...")
    model = IsolationForest(
        contamination=args.contamination,
        random_state=42,
        n_estimators=100
    )
    model.fit(X_scaled)
    
    # Predict anomalies
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_suspicious"] = model.predict(X_scaled)
    df["is_suspicious"] = df["is_suspicious"].map({1: 0, -1: 1})  # 1 = suspicious
    
    # Get suspicious owners
    suspicious = df[df["is_suspicious"] == 1].sort_values("anomaly_score", ascending=True)
    
    print(f"Found {len(suspicious)} suspicious owners ({args.contamination * 100:.1f}% expected)")
    
    # Save results
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suspicious.to_csv(output, index=False)
    print(f"Saved suspicious owners to {output}")
    
    # Print top suspicious owners
    print("\nTop 10 Most Suspicious Owners:")
    print(suspicious[["owner", "anomaly_score"] + feature_cols].head(10).to_string())


if __name__ == "__main__":
    main()
