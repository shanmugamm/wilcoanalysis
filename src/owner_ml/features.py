from __future__ import annotations

import re

import numpy as np
import pandas as pd


WILCO_CITY_NAMES = {
    "AUSTIN",
    "BARTLETT",
    "CEDAR PARK",
    "COUPLAND",
    "FLORENCE",
    "GEORGETOWN",
    "GRANGER",
    "HUTTO",
    "JARRELL",
    "LEANDER",
    "LIBERTY HILL",
    "ROUND ROCK",
    "TAYLOR",
    "THRALL",
    "WEIR",
}

BUSINESS_TERMS = re.compile(
    r"\b(?:LLC|L L C|INC|CORP|CO\b|LTD|LP|LLP|BANK|HOLDINGS|PROPERTIES|"
    r"INVEST|VENTURES|PARTNERS|ASSOC|ASSOCIATION|COMPANY)\b",
    flags=re.IGNORECASE,
)
TRUST_TERMS = re.compile(r"\b(?:TRUST|TRUSTEE|REVOCABLE|IRREVOCABLE|ESTATE)\b", flags=re.IGNORECASE)
GOV_TERMS = re.compile(
    r"\b(?:CITY OF|COUNTY|STATE OF|ISD|SCHOOL|USA|UNITED STATES)\b", flags=re.IGNORECASE
)


def add_owner_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact features useful for first-pass owner segmentation."""
    featured = df.copy()

    name = featured.get("FullName", pd.Series("", index=featured.index)).fillna("")
    featured["OwnerTypeGuess"] = np.select(
        [
            name.str.contains(GOV_TERMS, regex=True),
            name.str.contains(TRUST_TERMS, regex=True),
            name.str.contains(BUSINESS_TERMS, regex=True),
        ],
        ["government", "trust_estate", "business"],
        default="individual_or_unknown",
    )

    if "ExemptionList" in featured:
        exemption = featured["ExemptionList"].fillna("")
        featured["ExemptionCount"] = exemption.apply(
            lambda value: 0 if not value else len([token for token in re.split(r"[|,; ]+", value) if token])
        )
        for code in ["HS", "OV", "DP", "DV", "AG", "CBL"]:
            featured[f"Exemption_{code}"] = exemption.str.contains(rf"\b{code}\b", regex=True).astype(int)
    else:
        featured["ExemptionCount"] = 0

    if {"DataDate", "DateAddrChanged"}.issubset(featured.columns):
        age_days = (featured["DataDate"] - featured["DateAddrChanged"]).dt.days
        featured["AddressAgeDays"] = age_days.clip(lower=0).fillna(age_days.median())
    else:
        featured["AddressAgeDays"] = 0

    if "TaxingUnitGroupDesc" in featured:
        featured["TaxingUnitCount"] = featured["TaxingUnitGroupDesc"].fillna("").str.count(r"\|") + 1
    else:
        featured["TaxingUnitCount"] = 0

    if {"City", "State"}.issubset(featured.columns):
        state = featured["State"].fillna("").str.upper().str.strip()
        city = featured["City"].fillna("").str.upper().str.strip()
        featured["MailingGeo"] = np.select(
            [
                state.eq("") | state.eq("UNAVAILABLE") | city.eq("") | city.eq("UNAVAILABLE"),
                state.ne("TX"),
                state.eq("TX") & city.isin(WILCO_CITY_NAMES),
                state.eq("TX"),
            ],
            ["unavailable", "out_of_state", "wilco_area", "texas_other"],
            default="unknown",
        )
        featured["IsOutOfAreaMailing"] = featured["MailingGeo"].isin(["out_of_state", "texas_other"]).astype(int)
    else:
        featured["MailingGeo"] = "unknown"
        featured["IsOutOfAreaMailing"] = 0

    featured["OwnerProfileSegment"] = build_owner_profile_segments(featured)

    return featured


def build_owner_profile_segments(df: pd.DataFrame) -> pd.Series:
    """Create interpretable owner profile labels for analysis and cluster summaries."""
    owner_type = df.get("OwnerTypeGuess", pd.Series("unknown", index=df.index)).fillna("unknown")
    mailing_geo = df.get("MailingGeo", pd.Series("unknown", index=df.index)).fillna("unknown")
    homestead = df.get("Exemption_HS", pd.Series(0, index=df.index)).fillna(0).astype(int)
    ag = df.get("Exemption_AG", pd.Series(0, index=df.index)).fillna(0).astype(int)
    percent_ownership = pd.to_numeric(
        df.get("PercentOwnership", pd.Series(100, index=df.index)), errors="coerce"
    ).fillna(100)

    is_out_of_area = mailing_geo.isin(["out_of_state", "texas_other"])
    low_ownership = percent_ownership.lt(50)

    labels = np.select(
        [
            owner_type.eq("government"),
            owner_type.eq("business") & is_out_of_area,
            owner_type.eq("business"),
            owner_type.eq("trust_estate") & is_out_of_area,
            owner_type.eq("trust_estate"),
            homestead.eq(1) & mailing_geo.eq("wilco_area"),
            ag.eq(1) & is_out_of_area,
            is_out_of_area & low_ownership,
            is_out_of_area,
            mailing_geo.eq("unavailable"),
        ],
        [
            "government_public",
            "business_out_of_area",
            "business_local_or_unknown",
            "trust_estate_out_of_area",
            "trust_estate_local_or_unknown",
            "local_homestead",
            "ag_out_of_area",
            "partial_owner_out_of_area",
            "individual_out_of_area",
            "address_unavailable",
        ],
        default="individual_local_or_unknown",
    )

    return pd.Series(labels, index=df.index)
