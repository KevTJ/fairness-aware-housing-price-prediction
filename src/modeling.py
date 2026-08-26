"""Reusable model fitting and prediction utilities."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

LOCATION_FEATURES = ["Latitude", "Longitude"]


def fit_dummy_baseline(y_train_log):
    """Fit a constant median baseline on the log-price target."""
    model = DummyRegressor(strategy="median")
    dummy_x = np.zeros((len(y_train_log), 1))
    model.fit(dummy_x, y_train_log)
    return model


def predict_dummy_dollars(model, n_rows: int) -> np.ndarray:
    """Generate dollar-scale predictions from the log-target dummy model."""
    dummy_x = np.zeros((n_rows, 1))
    return np.exp(model.predict(dummy_x))


def _preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )


def build_random_forest_pipeline(
    *,
    include_location: bool = False,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    max_features: str | float | None = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Construct a leakage-safe structural or location-aware Random Forest pipeline."""
    numeric_features = NUMERIC_FEATURES + (LOCATION_FEATURES if include_location else [])

    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor(numeric_features, CATEGORICAL_FEATURES)),
            ("regressor", regressor),
        ]
    )


def fit_log_target_model(model: Pipeline, X_train, y_train_log) -> Pipeline:
    """Fit a model whose target is log(Sale Price)."""
    model.fit(X_train, y_train_log)
    return model


def predict_dollars(model: Pipeline, X) -> np.ndarray:
    """Convert log-price predictions back to the original dollar scale."""
    return np.exp(model.predict(X))
