# Wilco Owner Data ML Exploration

This project is a lightweight workspace for exploring `Owner_20260602.csv` with reproducible Python scripts.

## Data

The source file is expected at:

```text
Owner_20260602.csv
```

The CSV is intentionally not copied into a nested project folder. It is large and should stay as the local raw extract.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## First Pass Profiling

Create a compact markdown profile without loading the whole file into memory:

```powershell
python scripts\01_profile_owner_data.py
```

Outputs:

```text
reports/owner_profile.md
```

## Starter ML: Owner Segments

Run a baseline clustering experiment over sampled owner/address/exemption features:

```powershell
python scripts\02_owner_segments.py --sample-rows 50000 --clusters 8
```

Run it over the full file:

```powershell
python scripts\02_owner_segments.py --full --clusters 8
```

Outputs:

```text
reports/owner_segments.csv
reports/owner_segmentation_report.md
reports/owner_segment_summary.csv
reports/owner_profile_segment_summary.csv
```

## Owner Type And Geography Analysis

Build interpretable tables for owner-type guesses and mailing geography:

```powershell
python scripts\03_owner_type_geo_analysis.py
```

Outputs:

```text
reports/owner_type_geo_analysis.md
reports/owner_type_summary.csv
reports/mailing_geo_summary.csv
reports/owner_type_by_geo.csv
reports/top_out_of_state_owner_places.csv
reports/top_texas_other_owner_places.csv
```

## Suggested Exploration Questions

- Which owner records look like businesses, trusts, governments, or individuals?
- Which cities/states dominate out-of-area ownership?
- Are undeliverable records concentrated in particular owner types or address-change reasons?
- What exemption combinations appear most often?
- Can owner/property records be grouped into useful operational segments?
