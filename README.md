# Mobile Price Prediction Project (Nepal Market)

## Project Overview
This project aims to extract, clean, and analyze mobile phone price data from GadgetByte Nepal (a prominent Nepalese tech news website) for various brands, including Samsung, Apple, and Xiaomi. The extracted data is then used to train a machine learning model to predict mobile phone prices based on brand, RAM, storage, and model category. The final output includes a structured CSV dataset and a trained model for price prediction.

## Project Structure and Workflow

The project follows these main steps:

1.  **Data Extraction**: Scrape mobile phone model and price information from the specified website.
2.  **Data Cleaning and Preprocessing**: Parse raw text data to extract numerical prices, RAM, storage, and clean model/brand names.
3.  **Exploratory Data Analysis (EDA)**: Visualize price distributions and brand-wise comparisons to understand the data.
4.  **Model Training**: Develop a machine learning model (RandomForestRegressor) to predict mobile prices.
5.  **Model Evaluation**: Assess the model's performance using metrics like RMSE, MAE, and R2.
6.  **Prediction**: Demonstrate how to use the trained model for new predictions.

## Installation

To run this project, you'll need Python 3.x and the following libraries:

```bash
pip install requests beautifulsoup4 pandas scikit-learn matplotlib seaborn joblib
```

## Data Extraction

Mobile phone data is extracted from `https://www.gadgetbytenepal.com/category/mobile-price-in-nepal/`. The scraping process identifies individual brand listing pages for Samsung, Apple, and Xiaomi, then extracts model names and price strings from HTML tables.

### Raw Data Example (before cleaning):

| Brand   | Model               | Price                  |
| :------ | :------------------ | :--------------------- |
| Samsung | Galaxy Z Fold 7     | Rs. 244,999 (12+256GB) |
| Apple   | iPhone 17 Pro Max   | Rs. 279,999 (12+512GB) |
| Xiaomi  | Redmi Note 15 Pro+  | NPR 66,999 (8+128GB)   |

## Data Cleaning and Feature Engineering

The raw price strings are cleaned using regular expressions to extract `Price_Min`, `Price_Max`, `ram_gb`, `storage_gb`, and a `Notes` field for variants. Brand names are normalized, and a `model_cat` feature is created to group similar models.

### Cleaned Data Example (`all_mobile_prices_nepal.csv`):

| Brand   | Model             | Price_Min | Price_Max | Notes    |
| :------ | :---------------- | :-------- | :-------- | :------- |
| Samsung | Galaxy Z Fold 7   | 244999    | 244999    | 12+256GB |
| Apple   | iPhone 17 Pro Max | 279999    | 279999    | 12+512GB |
| Xiaomi  | Redmi Note 15 Pro+| 66999     | 66999     | 8+128GB  |

## Exploratory Data Analysis (EDA)

Visualizations are generated to understand price distribution and compare prices across different brands. Key plots include:

-   `eda_outputs/eda_price_distribution.png`: Histogram of mobile prices.
-   `eda_outputs/eda_price_by_brand.png`: Box plot of prices by brand (log scale).

## Machine Learning Model

A `RandomForestRegressor` model is trained using a `Pipeline` that includes preprocessing steps for numerical and categorical features. `RandomizedSearchCV` is used for hyperparameter tuning.

### Features Used:

-   `brand_clean`: Normalized brand name (Categorical)
-   `ram_gb`: RAM in GB (Numerical)
-   `storage_gb`: Storage in GB (Numerical)
-   `model_cat`: Categorized model name (Categorical - top N models, rest as 'other')

### Model Performance (on test set):

-   **RMSE**: 53132.59 NPR
-   **MAE**: 38263.61 NPR
-   **R2 Score**: 0.6134

### Top Feature Importances:

-   `storage_gb`: 0.555
-   `brand_clean_Apple`: 0.132
-   `brand_clean_Xiaomi`: 0.063
-   `model_cat_iPhone 17 Pro Max`: 0.049
-   `model_cat_Galaxy S26 Ultra`: 0.031

## Files Generated

-   `mobile_prices_nepal_2026.csv`: Cleaned data for Samsung, Apple, and Xiaomi (first iteration).
-   `all_mobile_prices_nepal.csv` (or `all_mobile_data5.csv`): Comprehensive cleaned dataset for all scraped brands.
-   `eda_outputs/eda_price_distribution.png`: Price distribution plot.
-   `eda_outputs/eda_price_by_brand.png`: Price by brand plot.
-   `mobile_price_model.joblib`: The trained machine learning pipeline.
-   `model_metadata.json`: Metadata about the trained model and training process.
-   `feature_importances.csv`: CSV file listing feature importances from the RandomForest model.
-   `predictions_random.csv`: Sample predictions on synthetically generated data.
-   `predictions_sample.csv`: Sample predictions on data sampled from the `all_mobile_data5.csv`.

## Usage (Prediction Example)

To make a prediction using the trained model, load the `mobile_price_model.joblib` file and provide new data in the same format as the training features.

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('mobile_price_model.joblib')

# Prepare new data for prediction
new_data = {
    'brand_clean': ['Samsung'],
    'ram_gb': [6],
    'storage_gb': [1024],
    'model_cat': ['model_5'] # Or a more specific model if available in the model_cat feature set
}
X_new = pd.DataFrame(new_data)

# Make prediction
predicted_price = model.predict(X_new)[0]

print(f"Predicted Price for Samsung (6GB/1024GB, model_5): Rs. {predicted_price:,.2f} NPR")
```

This will output a predicted price, for example: `Predicted Price for Samsung (6GB/1024GB, model_5): Rs. 258,095.97 NPR`
