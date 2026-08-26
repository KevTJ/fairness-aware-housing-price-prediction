"""Modeling-population, feature-selection, and preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COL = "Sale Price"

FAIRNESS_ONLY_COLS = [
    "total_population",
    "median_household_income",
    "poverty_rate",
    "pct_white_non_hispanic",
    "pct_black_non_hispanic",
    "pct_asian_non_hispanic",
    "pct_hispanic",
    "income_group",
    "poverty_group",
    "white_composition_group",
    "black_composition_group",
    "asian_composition_group",
    "hispanic_composition_group",
]

IDENTIFIER_COLS = [
    "Unnamed: 0",
    "PIN",
    "Deed No.",
    "Census Tract",
    "Census Tract GEOID",
]

POST_SALE_OR_FILTER_COLS = ["Most Recent Sale", "Pure Market Filter", "_merge"]
ASSESSMENT_COLS = ["Estimate (Land)", "Estimate (Building)"]
LOW_QUALITY_COLS = ["Construction Quality", "Site Desirability", "Other Improvements"]

ADDITIONAL_EXCLUSIONS = [
    "Use",
    "Modeling Group",
    "Description",
    "Age Decade",
    "Lot Size",
    "Neighborhood Code (mapping)",
    "Neigborhood Code (mapping)",  # spelling used in the source dataset
    "Sale Quarter",
    "Sale Half-Year",
    "Sale Quarter of Year",
    "Sale Half of Year",
    "Census Tract Code",
    "Town Code",
    "Neighborhood Code",
    "Town and Neighborhood",
    "Latitude",
    "Longitude",
]

NUMERIC_FEATURES = [
    "Land Square Feet",
    "Apartments",
    "Fireplaces",
    "Garage 1 Size",
    "Garage 2 Size",
    "Building Square Feet",
    "Number of Commercial Units",
    "Age",
    "Sale Year",
]

CATEGORICAL_FEATURES = [
    "Property Class",
    "Wall Material",
    "Roof Material",
    "Basement",
    "Basement Finish",
    "Central Heating",
    "Other Heating",
    "Central Air",
    "Attic Type",
    "Attic Finish",
    "Design Plan",
    "Cathedral Ceiling",
    "Garage 1 Material",
    "Garage 1 Attachment",
    "Garage 1 Area",
    "Garage 2 Material",
    "Garage 2 Attachment",
    "Garage 2 Area",
    "Porch",
    "Repair Condition",
    "Multi Code",
    "Multi Property Indicator",
    "O'Hare Noise",
    "Floodplain",
    "Road Proximity",
    "Sale Month of Year",
    "Garage Indicator",
]

PRIMARY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def define_modeling_population(data: pd.DataFrame) -> pd.DataFrame:
    """Restrict the analysis to economically meaningful pure-market transactions."""
    return data.loc[data["Pure Market Filter"] == 1].copy()


def create_temporal_splits(model_data: pd.DataFrame):
    """Create historical training, unseen-property 2019 test, and repeated-property tests."""
    train_data = model_data[model_data["Sale Year"].between(2013, 2018)].copy()
    test_2019 = model_data[model_data["Sale Year"] == 2019].copy()

    train_pins = set(train_data["PIN"])
    unseen_test = test_2019[~test_2019["PIN"].isin(train_pins)].copy()
    repeated_test = test_2019[test_2019["PIN"].isin(train_pins)].copy()
    return train_data, unseen_test, repeated_test


def build_preprocessor() -> ColumnTransformer:
    """Construct train-fitted preprocessing for the structural model."""
    numeric_preprocessor = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, NUMERIC_FEATURES),
            ("categorical", categorical_preprocessor, CATEGORICAL_FEATURES),
        ]
    )


def make_model_matrices(train_data, unseen_test, repeated_test):
    """Return feature matrices and dollar/log-scale targets."""
    X_train = train_data[PRIMARY_FEATURES].copy()
    X_test = unseen_test[PRIMARY_FEATURES].copy()
    X_test_repeated = repeated_test[PRIMARY_FEATURES].copy()

    y_train = train_data[TARGET_COL].copy()
    y_test = unseen_test[TARGET_COL].copy()
    y_test_repeated = repeated_test[TARGET_COL].copy()

    if (y_train <= 0).any():
        raise ValueError("Log-target modeling requires strictly positive training prices.")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_test_repeated": X_test_repeated,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_repeated": y_test_repeated,
        "y_train_log": np.log(y_train),
    }
