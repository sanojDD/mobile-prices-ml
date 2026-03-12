# 📱 Mobile Price Prediction — ML Pipeline (https://huggingface.co/spaces/Sanoj111/mobile_prices)

> Predict mobile phone prices in Nepal (NPR) from specs using an end-to-end machine learning pipeline.

---

## 📋 Project Overview

This project builds a regression model to predict smartphone prices in Nepalese Rupees (NPR) using device specifications parsed from a real-world, messy scraping dataset (`all_mobile_data5.csv`). The pipeline covers the full ML lifecycle: data parsing, feature engineering, EDA, model training, hyperparameter tuning, and inference on new data.

---

## 🗂️ Project Structure

```
mobile-price-prediction/
│
├── mobile_price_prediction.py   # Main pipeline (all steps)
├── requirements.txt             # Python dependencies
├── mobile_price_model.pkl       # Saved best model bundle
├── eda_dashboard.png            # EDA visualisation (12 plots)
├── model_evaluation.png         # Model evaluation charts
└── README.md                    # This file
```

---

## 📦 Dataset

| Field        | Description                                              |
|--------------|----------------------------------------------------------|
| `Model`      | Phone model name (e.g., Galaxy S25 Ultra)                |
| `Price`      | Raw price string (e.g., "Rs. 184,999 (12/256GB)")        |
| `Brand`      | Brand name or brand-slug string                          |
| `Price_clean`| Partially cleaned price (noisy, not directly usable)     |

**Raw records:** 127 rows — heavily messy with header rows, duplicate entries, malformed prices.  
**Clean records after parsing:** 109 usable rows.

---

## ⚙️ Pipeline Steps

### 1️⃣ Data Parsing & Preprocessing

The raw dataset is extremely messy — prices are embedded in strings like `"Rs. 184,999 (12/256GB)"` and `"NPR 199,999"`, RAM/Storage specs are inside price strings, and brand names are sometimes URL slugs.

Custom regex-based parsers handle:
- **Price extraction** — extracts 4–7 digit numbers within the realistic range (5,000–500,000 NPR)
- **RAM extraction** — handles `8GB`, `12/256`, `12+256`, `8GB+256GB` patterns
- **Storage extraction** — handles TB→GB conversion, paired patterns, standalone GB values
- **Brand normalization** — maps slugs like `samsung-galaxy-a36-price-nepal` → `Samsung`

Missing RAM/Storage values are imputed with the **median** of available values.

---

### 2️⃣ Feature Engineering

| Feature          | Description                                      |
|-----------------|--------------------------------------------------|
| `RAM_GB`         | RAM in gigabytes                                |
| `Storage_GB`     | Internal storage in gigabytes                   |
| `Is_5G`          | Binary flag: device supports 5G                 |
| `Is_Ultra`       | Binary flag: model name contains "Ultra"        |
| `Is_Pro`         | Binary flag: model name contains "Pro"          |
| `Is_Foldable`    | Binary flag: Fold or Flip form factor           |
| `RAM_x_Storage`  | Interaction term: RAM × Storage                 |
| `Log_Storage`    | log(1 + Storage) — captures diminishing returns |
| `Premium_Score`  | Weighted sum: Ultra×3 + Pro×2 + Foldable×4 + 5G×1 |
| `Brand_Encoded`  | Ordinal: Apple=3, Samsung=2, Xiaomi=1, Other=0 |

---

### 3️⃣ EDA (Exploratory Data Analysis)

The EDA dashboard (`eda_dashboard.png`) includes 12 plots:

- Price distribution histogram
- Average price by brand (bar chart)
- Brand share (pie chart)
- Price distribution by tier (box plots)
- RAM vs Price scatter (by brand)
- Storage vs Price scatter (by brand)
- Feature presence vs absence price comparison
- Correlation heatmap
- 5G vs Non-5G price comparison
- Count by price tier
- RAM distribution
- Storage distribution

**Price Tiers:**
| Tier        | Price Range (NPR)        |
|-------------|--------------------------|
| Budget      | < 20,000                 |
| Mid-Range   | 20,000 – 59,999          |
| Upper-Mid   | 60,000 – 119,999         |
| Flagship    | ≥ 120,000                |

---

### 4️⃣ Model Training

Four models were trained and evaluated with 5-fold cross-validation:

| Model                | Test R²  | MAE (NPR)  | RMSE (NPR) | CV R²  |
|---------------------|----------|------------|------------|--------|
| Ridge Regression     | 0.6776   | 41,141     | 48,503     | 0.7409 |
| Decision Tree        | 0.5677   | 38,241     | 56,167     | 0.7428 |
| Random Forest        | 0.6913   | 33,623     | 47,464     | 0.7703 |
| **Gradient Boosting**| **0.7487** | **31,371** | **42,819** | **0.7860** |

---

### 5️⃣ Hyperparameter Tuning

`GridSearchCV` was applied to the Random Forest model across:

```python
param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
    'max_features':      ['sqrt', 'log2'],
}
```

Best params: `max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=50`  
Best CV R²: **0.7690**

---

### 6️⃣ Best Model

🏆 **Gradient Boosting Regressor** was selected as the best model.

| Metric   | Value         |
|----------|---------------|
| R²       | 0.7487        |
| MAE      | NPR 31,371    |
| RMSE     | NPR 42,819    |
| CV R²    | 0.7860        |

---

### 7️⃣ Inference on New Data

The saved model bundle (`mobile_price_model.pkl`) includes the model, scaler, feature list, and brand encoder. Example predictions:

| Phone Specs                              | Predicted Price | Tier       |
|------------------------------------------|-----------------|------------|
| Budget Android (4GB/64GB)                | NPR 12,218      | Budget     |
| Mid-Range Samsung 5G (8GB/256GB)         | NPR 56,312      | Mid-Range  |
| Samsung Ultra (12GB/512GB, 5G, Ultra)    | NPR 163,363     | Flagship   |
| Apple iPhone Pro Max (12GB/1TB)          | NPR 270,262     | Flagship   |
| Samsung Z Fold (12GB/512GB, Foldable)    | NPR 128,310     | Flagship   |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python3 mobile_price_prediction.py
```

### 3. Load the saved model for inference

```python
import joblib
import pandas as pd
import numpy as np

bundle = joblib.load('mobile_price_model.pkl')
model  = bundle['model']
cols   = bundle['feature_cols']

# Example: Apple iPhone Pro, 8GB RAM, 256GB, 5G, Pro
sample = pd.DataFrame([{
    'RAM_GB': 8, 'Storage_GB': 256,
    'Is_5G': 1, 'Is_Ultra': 0, 'Is_Pro': 1, 'Is_Foldable': 0,
    'RAM_x_Storage': 8 * 256, 'Log_Storage': np.log1p(256),
    'Premium_Score': 0*3 + 1*2 + 0*4 + 1*1,
    'Brand_Encoded': 3,   # Apple=3
}])[cols]

predicted_price = model.predict(sample)[0]
print(f"Predicted Price: NPR {predicted_price:,.0f}")
```

---

## 📊 Key Insights

- **Brand** is the strongest price driver — Apple phones command a premium across all storage tiers
- **Storage** is more predictive than RAM alone; the interaction term `RAM × Storage` improves accuracy
- **Foldable** devices carry the highest premium feature score
- **5G** alone has a moderate effect; combined with Ultra/Pro it pushes prices significantly higher
- The dataset is small (109 records), which limits model generalisation — more data would improve R²

---

## 🛠️ Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 📝 Notes

- Dataset source: Nepal mobile price listings (scraped), prices in NPR
- The raw data contains significant noise — header rows mixed into data, malformed price strings, URL slugs as brand names. All cleaned via custom regex parsers.
- Model performance is limited by dataset size; with more diverse data, tree-ensemble models would perform significantly better.
