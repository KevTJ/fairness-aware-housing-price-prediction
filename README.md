# Fairness-Aware Housing Price Prediction in Cook County

A reproducible machine-learning project that predicts residential sale prices in Cook County, Illinois, and audits whether prediction errors differ systematically across neighborhood socioeconomic and demographic contexts.

## Project status

**Ongoing.** The public repository is being refactored from a single research notebook into a modular, reproducible workflow.

## Research questions

1. How accurately can residential sale prices be predicted from structural property characteristics?
2. How much additional predictive value is contributed by geographic information?
3. Do prediction errors differ systematically across neighborhood income, poverty, or demographic-composition groups?
4. Do observed disparities persist after accounting for differences in the underlying sale-price distribution?

## Fairness design

Census tract-level socioeconomic and demographic variables from the American Community Survey (ACS) are used for **post-model fairness auditing**. They are not included as direct demographic predictors in the housing-price models.

The fairness analysis evaluates both error magnitude and error direction across predefined neighborhood groups and later includes price-adjusted comparisons and robustness checks.

## Repository structure

```text
fairness-aware-housing-price-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_problem_data_and_fairness_framework.ipynb
│   └── 02_modeling_population_and_features.ipynb
├── scripts/
│   └── get_data.sh
├── src/
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── reports/
│   ├── figures/
│   └── tables/
└── docs/
```

Additional modeling, fairness-audit, and interpretation notebooks will be added as the original analysis is refactored.

## Notebook roadmap

1. **Problem, Data, and Fairness Framework** — data audit, Census geography validation, ACS integration, fairness-group construction, and evaluation design.
2. **Modeling Population and Features** — target definition, leakage review, sample construction, temporal holdout, and preprocessing.
3. **Baseline and Random Forest Models** — naive benchmark, structural Random Forest, and location-aware comparison.
4. **Gradient Boosting and Model Selection** — Histogram Gradient Boosting benchmarks and final-model selection.
5. **Fairness Audit** — socioeconomic and demographic error disparities, price adjustment, and sensitivity analysis.
6. **Model Interpretation and Error Analysis** — feature importance, residual diagnostics, and model limitations.

## Data

The large Cook County source dataset is intentionally excluded from version control. From the repository root, run:

```bash
bash scripts/get_data.sh
```

This downloads and verifies the archive into `data/raw/`, which is gitignored. See [`data/README.md`](data/README.md) for details.

## Census API configuration

ACS requests can be made without embedding credentials in the notebook. In GitHub Codespaces, provide a repository-scoped secret named `CENSUS_API_KEY`; locally, expose the same environment variable before running the analysis.

Never commit API keys, `.env` files, or URLs containing credentials.

## Environment

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Then launch Jupyter from the repository root:

```bash
jupyter lab
```

## Current scope and interpretation

This repository studies **fairness in housing-price prediction errors**. Group-level disparities are descriptive model-audit results and should not be interpreted by themselves as evidence of unlawful discrimination, causal effects, or individual-level demographic outcomes.

## License

Code in this repository is released under the MIT License. External datasets remain subject to their original licenses and terms of use.
