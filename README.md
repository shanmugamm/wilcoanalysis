# Wilco Property Owner Analysis

This repository contains the base version of a Williamson County property-owner analysis
project. It ingests a local Wilco owner extract, creates owner and mailing-location
features, runs first-pass ML segmentation, generates reports, and exposes the outputs in a
Streamlit dashboard.

The current deployed app is hosted on Google Cloud Run:

```text
https://wilcoanalysis-ld322r5mnq-uc.a.run.app
```

## Current Capabilities

- Profiles the raw owner CSV without requiring the full file to stay in memory.
- Classifies owners with interpretable heuristics such as individual, business, trust or
  estate, and government.
- Groups mailing addresses into Wilco-area, other Texas, out-of-state, unavailable, and
  unknown geography buckets.
- Identifies owners connected to multiple distinct properties.
- Runs a baseline unsupervised ML segmentation model using `MiniBatchKMeans`.
- Produces Markdown and CSV reports under `reports/`.
- Serves generated reports through a Streamlit dashboard from Cloud Storage, with local
  `reports/` fallback for development.
- Deploys the Streamlit app to Google Cloud Run using the included `Dockerfile`.
- Can run the report pipeline as a Cloud Run Job using Cloud Storage input/output.

## Repository Layout

```text
app.py                         Streamlit reports dashboard with GCS report loading
Dockerfile                     Cloud Run container definition
requirements.txt               Python runtime dependencies
pyproject.toml                 Project metadata and tooling config
scripts/
  01_profile_owner_data.py     Raw owner-data profiling
  02_owner_segments.py         Baseline ML clustering and segment summaries
  03_owner_type_geo_analysis.py
                                Owner type and mailing geography reports
  04_multi_property_owners.py  Multiple-property owner analysis
  run_owner_pipeline_gcs_job.py
                                Cloud Run Job entrypoint for GCS input/output
  run_owner_pipeline.py        End-to-end report pipeline
src/owner_ml/
  data.py                      CSV discovery, chunked reads, and cleanup
  features.py                  Owner type, geography, exemption, and profile features
reports/                       Generated report files
```

## Data

The raw source extract is expected in the project root using this naming pattern:

```text
Owner_YYYYMMDD.csv
```

For example:

```text
Owner_20260602.csv
```

Raw extracts are intentionally ignored by git because they are large local data files.
Generated reports are also ignored by git, except for the placeholder that keeps the
`reports/` directory present.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Dashboard

Start the Streamlit dashboard locally:

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

By default, the dashboard reads Markdown and CSV report files from:

```text
gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest
```

Set a different report source with:

```powershell
$env:REPORT_SOURCE_GCS_PREFIX="gs://your-bucket/reports/latest"
streamlit run app.py
```

If the GCS source is unavailable and local `reports/` exists, the dashboard falls back to
local files for development.

## Run The Full Pipeline

Run all report steps against the newest `Owner_*.csv` file in the project root:

```powershell
python scripts\run_owner_pipeline.py
```

Use a specific extract:

```powershell
python scripts\run_owner_pipeline.py --data Owner_20260602.csv
```

Use sampled segmentation for a faster development run:

```powershell
python scripts\run_owner_pipeline.py --sample-segments 50000
```

Pipeline outputs include:

```text
reports/owner_profile.md
reports/owner_type_geo_analysis.md
reports/multi_property_owners_report.md
reports/owner_segmentation_report.md
reports/pipeline_manifest.md
```

## Individual Analysis Commands

Profile the source file:

```powershell
python scripts\01_profile_owner_data.py --data Owner_20260602.csv
```

Run owner type and mailing geography analysis:

```powershell
python scripts\03_owner_type_geo_analysis.py --data Owner_20260602.csv
```

Find owners connected to multiple properties:

```powershell
python scripts\04_multi_property_owners.py --data Owner_20260602.csv
```

Exclude organization-style owners such as LLCs, corporations, banks, governments, and similar
entities:

```powershell
python scripts\04_multi_property_owners.py --data Owner_20260602.csv --exclude-organizations
```

Run baseline ML owner segmentation:

```powershell
python scripts\02_owner_segments.py --data Owner_20260602.csv --sample-rows 50000 --clusters 8
```

Run segmentation against the full file:

```powershell
python scripts\02_owner_segments.py --data Owner_20260602.csv --full --clusters 8
```

## Feature And ML Notes

The current ML step is an exploratory clustering model, not a supervised prediction model.
It uses engineered features such as:

- owner type guess from owner-name patterns;
- mailing geography category;
- out-of-area mailing indicator;
- exemption indicators and exemption count;
- address age;
- taxing-unit count;
- ownership percentage;
- primary-owner and undeliverable flags.

The model assigns each record to an `OwnerSegment` cluster and also reports an interpretable
`OwnerProfileSegment`, such as `local_homestead`, `business_out_of_area`,
`trust_estate_out_of_area`, or `individual_out_of_area`.

## Deploy To Google Cloud Run

Deployment is now handled by GitHub Actions on every push to `main`:

```text
.github/workflows/deploy-cloud-run.yml
```

The workflow uses Google Workload Identity Federation, so no long-lived service account key
is stored in GitHub. It deploys:

- Cloud Run service: `wilcoanalysis`
- Cloud Run Job definition: `wilco-owner-pipeline`

The workflow does not execute the pipeline job automatically. Run the job manually when a new
owner extract is ready.

Manual local deploy is still available when needed:

```powershell
gcloud.cmd run deploy wilcoanalysis `
  --source . `
  --region us-central1 `
  --project noble-kingdom-497421-f7 `
  --set-env-vars REPORT_SOURCE_GCS_PREFIX=gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest `
  --allow-unauthenticated
```

On Windows PowerShell, use `gcloud.cmd` if the PowerShell script shim is blocked by local
execution policy.

GitHub Actions deployment identity:

```text
github-deployer@noble-kingdom-497421-f7.iam.gserviceaccount.com
```

Workload Identity provider:

```text
projects/921836521382/locations/global/workloadIdentityPools/github-pool/providers/github-wilcoanalysis
```

## Run The Pipeline In Cloud Run Jobs

The enterprise-friendly flow is:

```text
Cloud Storage raw CSV -> Cloud Run Job pipeline -> Cloud Storage reports -> Streamlit app
```

Create a storage bucket once:

```powershell
gcloud.cmd storage buckets create gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7 `
  --location=us-central1 `
  --project=noble-kingdom-497421-f7
```

Allow the Cloud Run Job runtime service account to read/write objects in that bucket:

```powershell
gcloud.cmd storage buckets add-iam-policy-binding gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7 `
  --member serviceAccount:921836521382-compute@developer.gserviceaccount.com `
  --role roles/storage.objectAdmin `
  --project noble-kingdom-497421-f7
```

Upload the owner extract:

```powershell
gcloud.cmd storage cp Owner_20260602.csv gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/raw/Owner_20260602.csv
```

Deploy or update the Cloud Run Job from this source tree:

```powershell
gcloud.cmd run jobs deploy wilco-owner-pipeline `
  --source . `
  --region us-central1 `
  --project noble-kingdom-497421-f7 `
  --command python `
  --args scripts/run_owner_pipeline_gcs_job.py `
  --set-env-vars OWNER_INPUT_GCS_URI=gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/raw/Owner_20260602.csv,REPORT_OUTPUT_GCS_PREFIX=gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest,OWNER_SAMPLE_SEGMENTS=50000,OWNER_EXCLUDE_ORGANIZATIONS_MULTI_PROPERTY=true `
  --memory 4Gi `
  --cpu 2 `
  --task-timeout 3600
```

Run the job:

```powershell
gcloud.cmd run jobs execute wilco-owner-pipeline `
  --region us-central1 `
  --project noble-kingdom-497421-f7 `
  --wait
```

Download generated reports if needed:

```powershell
gcloud.cmd storage cp --recursive gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest reports
```

## Important Limitations

- Report files are local generated artifacts and are not committed to git.
- A deploy from the local workspace includes the current local `reports/` files.
- A deploy from GitHub alone will need a report-generation step or a cloud storage location
  for report artifacts.
- The Streamlit app reads Cloud Storage reports by default. Local `reports/` files are now
  mainly a development fallback.
- The multi-property dashboard CSV view includes a checkbox to exclude organization-style
  owners such as LLCs, corporations, banks, governments, and similar entities.
- The current ML output is unsupervised segmentation. A true predictive model needs a
  labeled target, model evaluation metrics, and saved model artifacts.

## Good Next Steps

- Add a dashboard page focused on ML segments instead of only file browsing.
- Store generated report artifacts in Google Cloud Storage.
- Add a repeatable build or CI step that regenerates reports before deployment.
- Define a supervised prediction target if the project needs actual forecasts or scores.
- Save trained model artifacts and expose a small prediction workflow in the app.
