## Bulldozer Predictive System

This project predicts the sale price of bulldozers using historical auction data. It’s built as a complete machine learning pipeline: raw CSVs are ingested, cleaned, and transformed, models are trained and tuned, results are evaluated, and predictions can be made through a Streamlit web app.

## Project Background

The dataset is based on bulldozer sales history (the “Blue Book for Bulldozers” from Kaggle). The task is to predict SalePrice given machine specifications and auction details.

Key points to handle in this problem:

The data is time-based, so training and validation splits are done by year, not randomly.

The saledate column is transformed into useful time features such as year, month, day of week, and day of year.

Both numerical and categorical data are present and require different preprocessing.

The chosen evaluation metric is Root Mean Squared Log Error (RMSLE), which punishes large percentage errors.

How the System Works
## Data ingestion

Reads the raw train and test files.

Parses saledate and adds date-based features.

Saves the cleaned datasets for downstream processing.

## Data transformation

Splits numeric and categorical columns.

Numeric features are median-imputed.

Categorical features are imputed and then ordinal-encoded.

The preprocessing pipeline is saved for later reuse.

## Model training

Trains several regression models: linear models, decision trees, random forests, gradient boosting, XGBoost, CatBoost, etc.

Evaluates each model on the validation set using RMSLE.

Selects and saves the best-performing model.

## Model tuning

Focuses on Random Forest, which performed best among baselines.

Uses randomized search with time-series cross-validation to tune hyperparameters.

Saves the tuned Random Forest model.

## Model evaluation

Loads the tuned model and evaluates it on the validation set.

RMSLE is calculated and stored in a results file.

## Orchestration

The full pipeline is controlled in main.py, running each stage in order.

Logs track the process for reproducibility.

## Streamlit app

Users can upload their own CSVs.

The system applies the saved preprocessing pipeline and tuned model to generate predictions.

Results are displayed and can be downloaded as a new CSV.

## Results

From the latest evaluation:

Validation RMSLE: 0.261

This means predictions are usually within about 26% of the true bulldozer sale price, which is strong performance for auction data.

## Key Takeaways

Automated preprocessing with robust handling of mixed numeric and categorical data.

Feature engineering from dates adds strong predictive value.

Multiple models tested, with a tuned Random Forest emerging as the most effective.

RMSLE of 0.261 shows the model generalizes well.

A user-friendly Streamlit app makes predictions easy to generate and download.
