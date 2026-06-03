from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_PATH = Path("Owner_20260602.csv")


DATE_COLUMNS = ["DateAddrChanged", "DataDate"]
DATE_FORMAT = "%Y %b %d %I:%M:%S %p"
NUMERIC_COLUMNS = [
    "AdHocTaxYear",
    "PrimaryOwner",
    "PercentOwnership",
    "IsUndeliverable",
    "TxTu_HSCapAdj",
]


def read_owner_sample(
    path: str | Path = DEFAULT_DATA_PATH,
    nrows: int | None = 50_000,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read a manageable owner-data sample with basic type cleanup."""
    df = pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False)
    return clean_owner_frame(df)


def read_owner_chunks(
    path: str | Path = DEFAULT_DATA_PATH,
    chunksize: int = 100_000,
    usecols: Iterable[str] | None = None,
):
    """Yield cleaned chunks so large extracts can be profiled on modest machines."""
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False):
        yield clean_owner_frame(chunk)


def clean_owner_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in DATE_COLUMNS:
        if column in df:
            df[column] = pd.to_datetime(df[column], format=DATE_FORMAT, errors="coerce")

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    if "ZIP" in df:
        df["ZIP5"] = df["ZIP"].str.extract(r"(\d{5})", expand=False).fillna("")

    return df
