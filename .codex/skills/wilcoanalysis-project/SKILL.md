---
name: wilcoanalysis-project
description: Project context and workflow guide for the Wilco Property Owner Analysis repository. Use when working in this repo on the Streamlit dashboard, Williamson County owner extract processing, owner feature engineering, ML segmentation, report generation, Cloud Run deployment, README/docs, or git/deploy maintenance so Codex can avoid rediscovering the project from scratch.
---

# Wilcoanalysis Project

Use this as the first orientation pass for `c:\workspace\wilcoanalysis`.

## Project Purpose

Maintain a base-version Williamson County property-owner analysis app:

- ingest local `Owner_YYYYMMDD.csv` extracts;
- clean and feature owner records;
- infer owner type and mailing geography;
- find multiple-property owners;
- run exploratory owner segmentation;
- generate Markdown/CSV reports in `reports/`;
- display those reports in a Streamlit app;
- deploy the app to Google Cloud Run.

## Important Files

- `app.py`: Streamlit dashboard that lists and displays Markdown/CSV files from `reports/`.
- `src/owner_ml/data.py`: CSV discovery, chunked reads, basic type cleanup.
- `src/owner_ml/features.py`: owner type guesses, mailing geography, exemption features, profile labels.
- `scripts/run_owner_pipeline.py`: end-to-end report pipeline.
- `scripts/01_profile_owner_data.py`: raw extract profiling.
- `scripts/02_owner_segments.py`: exploratory ML clustering with `MiniBatchKMeans`.
- `scripts/03_owner_type_geo_analysis.py`: owner type and mailing geography summaries.
- `scripts/04_multi_property_owners.py`: owners linked to multiple distinct properties.
- `Dockerfile`, `.dockerignore`, `.gcloudignore`: Cloud Run deployment setup.
- `README.md`: user-facing project guide.

## Data And Artifacts

- Raw extracts live in the repo root as `Owner_*.csv`.
- Raw extracts are git-ignored and should not be committed.
- Generated report files in `reports/*.md` and `reports/*.csv` are git-ignored.
- `reports/.gitkeep` keeps the directory present.
- Local Cloud Run deploys include current local reports because `.gcloudignore` does not exclude `reports/`.
- GitHub-only deploys need report regeneration or artifact storage, such as Google Cloud Storage.

## Common Commands

Setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run dashboard:

```powershell
streamlit run app.py
```

Run all reports with the newest extract:

```powershell
python scripts\run_owner_pipeline.py
```

Run all reports with a specific extract:

```powershell
python scripts\run_owner_pipeline.py --data Owner_20260602.csv
```

Use sampled segmentation for faster development:

```powershell
python scripts\run_owner_pipeline.py --sample-segments 50000
```

Deploy to Cloud Run:

```powershell
gcloud.cmd run deploy wilcoanalysis `
  --source . `
  --region us-central1 `
  --project noble-kingdom-497421-f7 `
  --allow-unauthenticated
```

Use `gcloud.cmd` on Windows when PowerShell blocks the `gcloud.ps1` shim.

## Current Cloud Run Service

- Project: `noble-kingdom-497421-f7`
- Region: `us-central1`
- Service: `wilcoanalysis`
- URL: `https://wilcoanalysis-ld322r5mnq-uc.a.run.app`

## ML Context

The current ML output is exploratory unsupervised segmentation, not supervised prediction.
Avoid claiming the system predicts outcomes unless a labeled target, evaluation metrics,
and saved model artifacts have been added.

Current feature engineering includes:

- `OwnerTypeGuess`: `government`, `trust_estate`, `business`, `individual_or_unknown`;
- `MailingGeo`: `wilco_area`, `texas_other`, `out_of_state`, `unavailable`, `unknown`;
- exemption count and flags for common exemption codes;
- address age, taxing-unit count, ownership percentage, primary-owner flag, undeliverable flag;
- `OwnerProfileSegment`: interpretable labels such as `local_homestead`,
  `business_out_of_area`, `trust_estate_out_of_area`, `ag_out_of_area`, and related groups.

`scripts/02_owner_segments.py` builds a preprocessing pipeline with categorical one-hot
encoding, numeric scaling, and `MiniBatchKMeans`.

## Working Guidance

- Keep raw data and generated reports out of git unless the user explicitly changes that policy.
- Prefer updating scripts and README together when changing workflow commands.
- When changing dashboard behavior, remember that Cloud Run uses port `${PORT:-8080}` from the Dockerfile.
- After deployment config changes, deploy with `gcloud.cmd run deploy ...` and verify the public URL returns `200 OK`.
- When pushing changes, commit only intentional project files and leave ignored raw/report artifacts alone.
