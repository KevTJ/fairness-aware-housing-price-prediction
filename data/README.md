# Data Setup

Large source and processed datasets are intentionally excluded from Git version control. This keeps the internship-facing repository lightweight while preserving a reproducible setup path.

## Quick setup

From the repository root, run:

```bash
bash scripts/get_data.sh
```

The script downloads the Cook County archive used for this project from the author's project-data repository, verifies that the ZIP contains `cook_county_train.csv`, and stores it at:

```text
data/raw/cook_county_data.zip
```

The source URL can be overridden without editing the script:

```bash
COOK_COUNTY_DATA_URL="<alternate-url>" bash scripts/get_data.sh
```

## Expected local structure

After setup:

```text
data/
├── README.md
├── raw/
│   └── cook_county_data.zip
└── processed/
```

The archive is expected to contain:

```text
cook_county_train.csv
```

The analysis reads the archive directly; the CSV does not need to remain extracted on disk.

## Why the raw data are not committed here

The Cook County archive is larger than is appropriate for an internship portfolio repository and is not part of the source code. Keeping raw and derived datasets outside Git avoids repository bloat and makes the version history focus on analytical development.

The contents of `data/raw/` and `data/processed/` are therefore ignored by Git. Selected figures and summary tables intended for review are committed under `reports/` instead.

## ACS data

Census tract-level demographic and socioeconomic variables are retrieved from the **2019 American Community Survey (ACS) 5-Year API** during the analysis.

In GitHub Codespaces, this project uses a repository-scoped Codespaces secret named:

```text
CENSUS_API_KEY
```

The notebooks read the value from the environment. The secret value itself must never be committed to the repository or printed into saved notebook output.
