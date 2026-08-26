# Data Setup

Large source and processed datasets are intentionally excluded from Git version control.

## Expected local structure

Place the Cook County data archive at:

```text
data/raw/cook_county_data.zip
```

The archive is expected to contain:

```text
cook_county_train.csv
```

Notebook 1 reads this archive directly and does not require the CSV to be extracted permanently.

## ACS data

Census tract-level demographic and socioeconomic variables are retrieved from the **2019 American Community Survey (ACS) 5-Year API** during the analysis.

If a Census API key is used, store it outside the repository and expose it through the `CENSUS_API_KEY` environment variable. Do not place credentials in notebooks or committed files.

## Version-control policy

The contents of `data/raw/` and `data/processed/` are ignored by Git to avoid committing large datasets or derived artifacts. Only this documentation and placeholder files are tracked.
