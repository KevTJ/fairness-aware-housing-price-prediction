"""Data loading and ACS enrichment utilities for the Cook County project."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ACS_URL = "https://api.census.gov/data/2019/acs/acs5"

ACS_VARIABLES = {
    "B01001_001E": "total_population",
    "B19013_001E": "median_household_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "below_poverty",
    "B03002_001E": "race_ethnicity_total",
    "B03002_003E": "white_non_hispanic",
    "B03002_004E": "black_non_hispanic",
    "B03002_006E": "asian_non_hispanic",
    "B03002_012E": "hispanic",
}

FAIRNESS_VARIABLES = [
    "median_household_income",
    "poverty_rate",
    "pct_white_non_hispanic",
    "pct_black_non_hispanic",
    "pct_asian_non_hispanic",
    "pct_hispanic",
]


def load_property_data(data_zip: str | Path) -> pd.DataFrame:
    """Load the Cook County training CSV from the local ZIP archive."""
    data_zip = Path(data_zip)
    if not data_zip.exists():
        raise FileNotFoundError(
            f"Missing {data_zip}. See data/README.md for dataset setup instructions."
        )

    with zipfile.ZipFile(data_zip, "r") as zf:
        with zf.open("cook_county_train.csv") as f:
            return pd.read_csv(f)


def add_census_geoid(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize Cook County Census tract identifiers and construct GEOIDs."""
    result = data.copy()
    result["Census Tract Code"] = (
        result["Census Tract"].astype("Int64").astype("string").str.zfill(6)
    )
    result["Census Tract GEOID"] = "17031" + result["Census Tract Code"]
    return result


def fetch_acs_fairness_data(api_key: str | None = None) -> pd.DataFrame:
    """Download and construct 2019 ACS tract-level fairness variables for Cook County."""
    if api_key is None:
        api_key = os.environ.get("CENSUS_API_KEY")

    params = {
        "get": "NAME," + ",".join(ACS_VARIABLES),
        "for": "tract:*",
        "in": "state:17 county:031",
    }
    if api_key:
        params["key"] = api_key

    response = requests.get(ACS_URL, params=params, timeout=30)
    response.raise_for_status()
    raw = response.json()

    acs = pd.DataFrame(raw[1:], columns=raw[0]).rename(columns=ACS_VARIABLES)
    acs["Census Tract GEOID"] = acs["state"] + acs["county"] + acs["tract"]

    for col in ACS_VARIABLES.values():
        acs[col] = pd.to_numeric(acs[col], errors="coerce")

    # ACS special negative estimates represent unavailable values, not valid income.
    acs.loc[acs["median_household_income"] < 0, "median_household_income"] = np.nan

    acs["poverty_rate"] = acs["below_poverty"] / acs["poverty_universe"]
    acs["pct_white_non_hispanic"] = acs["white_non_hispanic"] / acs["race_ethnicity_total"]
    acs["pct_black_non_hispanic"] = acs["black_non_hispanic"] / acs["race_ethnicity_total"]
    acs["pct_asian_non_hispanic"] = acs["asian_non_hispanic"] / acs["race_ethnicity_total"]
    acs["pct_hispanic"] = acs["hispanic"] / acs["race_ethnicity_total"]

    merge_cols = ["Census Tract GEOID", "total_population", *FAIRNESS_VARIABLES]
    return acs[merge_cols].copy()


def add_fairness_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Create balanced property-level quartile groups for post-model auditing."""
    result = data.copy()
    result["income_group"] = pd.qcut(
        result["median_household_income"],
        q=4,
        labels=["Q1: Lowest income", "Q2", "Q3", "Q4: Highest income"],
    )
    result["poverty_group"] = pd.qcut(
        result["poverty_rate"],
        q=4,
        labels=["Q1: Lowest poverty", "Q2", "Q3", "Q4: Highest poverty"],
    )

    composition_variables = {
        "pct_white_non_hispanic": "white_composition_group",
        "pct_black_non_hispanic": "black_composition_group",
        "pct_asian_non_hispanic": "asian_composition_group",
        "pct_hispanic": "hispanic_composition_group",
    }
    for variable, new_col in composition_variables.items():
        result[new_col] = pd.qcut(
            result[variable],
            q=4,
            labels=["Q1: Lowest share", "Q2", "Q3", "Q4: Highest share"],
        )
    return result


def build_analysis_dataset(data_zip: str | Path, api_key: str | None = None) -> pd.DataFrame:
    """Build the property-level dataset used for modeling and fairness auditing."""
    property_data = add_census_geoid(load_property_data(data_zip))
    acs = fetch_acs_fairness_data(api_key=api_key)
    merged = property_data.merge(
        acs,
        on="Census Tract GEOID",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    return add_fairness_groups(merged)
