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
- run the report pipeline as a Cloud Run Job with Cloud Storage input/output;
- deploy the app to Google Cloud Run.

## Important Files

- `app.py`: Streamlit dashboard that lists and displays Markdown/CSV reports from GCS by
  default, with local `reports/` fallback. It has Overview, Tables, and Narratives views;
  the Tables view supports search, owner/geography filters, multi-property organization
  exclusion, column selection, sorting, row limits, charts, and filtered downloads.
- `src/owner_ml/data.py`: CSV discovery, chunked reads, basic type cleanup.
- `src/owner_ml/features.py`: owner type guesses, mailing geography, exemption features, profile labels.
- `scripts/run_owner_pipeline.py`: end-to-end report pipeline.
- `scripts/01_profile_owner_data.py`: raw extract profiling.
- `scripts/02_owner_segments.py`: exploratory ML clustering with `MiniBatchKMeans`.
- `scripts/03_owner_type_geo_analysis.py`: owner type and mailing geography summaries.
- `scripts/04_multi_property_owners.py`: owners linked to multiple distinct properties.
- `scripts/run_owner_pipeline_gcs_job.py`: Cloud Run Job wrapper that downloads an owner
  extract from GCS, runs the local pipeline, and uploads generated reports to GCS.
- `Dockerfile`, `.dockerignore`, `.gcloudignore`: Cloud Run deployment setup.
- `.github/workflows/deploy-streamlit-service.yml`: path-filtered GitHub Actions deploy for
  the Streamlit Cloud Run service; includes `app.py` and shared dashboard filter helpers.
- `.github/workflows/deploy-pipeline-job.yml`: path-filtered GitHub Actions deploy for the
  Cloud Run Job definition; it does not execute the pipeline job.
- `README.md`: user-facing project guide.

## Data And Artifacts

- Raw extracts live in the repo root as `Owner_*.csv`.
- Raw extracts are git-ignored and should not be committed.
- Generated report files in `reports/*.md` and `reports/*.csv` are git-ignored.
- `reports/.gitkeep` keeps the directory present.
- Local Cloud Run deploys include current local reports because `.gcloudignore` does not exclude `reports/`.
- GitHub-only deploys need report regeneration or artifact storage, such as Google Cloud Storage.
- Cloud Run Jobs should use `OWNER_INPUT_GCS_URI` and `REPORT_OUTPUT_GCS_PREFIX` to keep raw
  data and generated reports outside the app image.
- Multi-property reports support `--exclude-organizations` and the Cloud Run Job env var
  `OWNER_EXCLUDE_ORGANIZATIONS_MULTI_PROPERTY=true` to exclude organization-style owners
  such as LLCs, corporations, banks, governments, and similar entities.
- The Streamlit app reads from `REPORT_SOURCE_GCS_PREFIX`, defaulting to
  `gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest`.
- The job runtime identity needs `roles/storage.objectAdmin` on the artifact bucket. Current
  default service account: `921836521382-compute@developer.gserviceaccount.com`.
- GitHub Actions deployment uses Workload Identity Federation provider
  `projects/921836521382/locations/global/workloadIdentityPools/github-pool/providers/github-wilcoanalysis`
  and service account `github-deployer@noble-kingdom-497421-f7.iam.gserviceaccount.com`.

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

GitHub Actions deploys Cloud Run on every push to `main`. Manual deploy remains available:

```powershell
gcloud.cmd run deploy wilcoanalysis `
  --source . `
  --region us-central1 `
  --project noble-kingdom-497421-f7 `
  --set-env-vars REPORT_SOURCE_GCS_PREFIX=gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7/reports/latest `
  --allow-unauthenticated
```

Use `gcloud.cmd` on Windows when PowerShell blocks the `gcloud.ps1` shim.

Deploy the Cloud Run Job:

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

Execute the Cloud Run Job:

```powershell
gcloud.cmd run jobs execute wilco-owner-pipeline `
  --region us-central1 `
  --project noble-kingdom-497421-f7 `
  --wait
```

## Current Cloud Run Service

- Project: `noble-kingdom-497421-f7`
- Region: `us-central1`
- Service: `wilcoanalysis`
- URL: `https://wilcoanalysis-ld322r5mnq-uc.a.run.app`
- Pipeline job: `wilco-owner-pipeline`
- Artifact bucket: `gs://wilcoanalysis-artifacts-noble-kingdom-497421-f7`

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
- After pipeline job changes, deploy with `gcloud.cmd run jobs deploy ...`, execute with
  `gcloud.cmd run jobs execute ... --wait`, and check that reports upload to GCS.
- When pushing changes, commit only intentional project files and leave ignored raw/report artifacts alone.
