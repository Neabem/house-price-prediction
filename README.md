# 🏠 Real Estate Valuation & Property Price Prediction

An end-to-end Machine Learning regression pipeline engineered to predict continuous property selling prices.

This repository demonstrates a complete production workflow: extracting relational records and neighborhood indicators across multi-table SQLite schemas, preventing data leakage with a Scikit-Learn `ColumnTransformer`, optimizing gradient boosted decision trees (`XGBRegressor`) via 5-fold cross-validated random search (`RandomizedSearchCV`), and conducting diagnostic residual error analyses.

---

## 🛠️ Tech Stack & Tooling

* **Relational Database / SQL:** SQLite3 (`JOIN` across `houses` and `neighborhoods` tables)
* **Data Processing & Linear Algebra:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Feature Engineering & Preprocessing:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`)
* **Machine Learning & Modeling:** XGBoost (`XGBRegressor`), Scikit-Learn (`RandomizedSearchCV`, `train_test_split`)
* **Model Evaluation:** Scikit-Learn (`mean_absolute_error`, `mean_squared_error`, `r2_score`)

---

## 📊 Performance Benchmarks (Unseen Test Set)

Hyperparameter optimization across tree depth, boosting rounds, learning rate, and subsampling ratios yielded substantial performance gains across all regression metrics:

| Metric | Baseline Model (Default XGBoost) | Tuned Model (5-Fold CV Search) | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **\$36,794.02** | **\$32,508.95** | **-\$4,285.07** | **+11.65%** |
| **Root Mean Squared Error (RMSE)** | **\$50,061.54** | **\$45,184.45** | **-\$4,877.09** | **+9.74%** |
| **Coefficient of Determination ($R^2$)** | **0.6810** | **0.7401** | **+0.0591** | **+8.68%** |
| **Mean Absolute Pct Error (MAPE)** | **7.41%** | **6.52%** | **-0.89%** | **+11.98%** |

---

## 🏗️ Architecture & Engineering Workflow

### 1. Relational Data Extraction (SQL)
Property attributes and neighborhood-level indicators are partitioned across relational tables inside `real_estate.db`:
* `houses`: Primary key `house_id`, square footage (`sqft`), bedroom count, bathroom count, construction year (`year_built`), garage status, foreign key `neighborhood_id`, and sale `price`.
* `neighborhoods`: Primary key `neighborhood_id`, `neighborhood_name`, `school_rating` (1-10 scale), and `crime_rate_index`.

Executed an explicit relational `JOIN` query on `neighborhood_id`, filtering non-predictive primary and foreign keys prior to model ingestion.

```sql
SELECT 
    h.house_id,
    h.sqft,
    h.bedrooms,
    h.bathrooms,
    h.year_built,
    h.has_garage,
    n.neighborhood_name,
    n.school_rating,
    n.crime_rate_index,
    h.price
FROM houses h
JOIN neighborhoods n ON h.neighborhood_id = n.neighborhood_id
```

### 2. Leak-Free Preprocessing Architecture (`ColumnTransformer`)
Raw databases frequently contain missing records (e.g. 15% missing `sqft`, 8% missing `bedrooms`, 12% missing `has_garage`). To prevent **data leakage**, all transformation parameters are computed strictly on the training partition:
* **Numeric Stream** (`sqft`, `year_built`, `crime_rate_index`, `bedrooms`, `bathrooms`, `school_rating`): Imputed with `SimpleImputer(strategy='median')` to resist skew from luxury outlier homes, followed by `StandardScaler()` for zero-mean, unit-variance normalization.
* **Categorical Stream** (`has_garage`, `neighborhood_name`): Imputed with `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to safely handle unobserved categories during inference.

### 3. Model Training & 5-Fold Cross-Validation (`XGBRegressor`)
* **Baseline Benchmark:** Untuned `XGBRegressor(random_state=42)` yielded an unseen test MAE of **\$36,794.02**.
* **Hyperparameter Search Space:** Evaluated 30 candidate parameter combinations across 5 cross-validation folds (`RandomizedSearchCV`), tuning `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
* **Optimal Configuration:**
  * `max_depth`: 3
  * `n_estimators`: 300
  * `learning_rate`: 0.03
  * `subsample`: 0.7
  * `colsample_bytree`: 0.7

### 4. Residual Diagnostics & Explainability
* **Actual vs. Predicted Diagnostics:** Strong linear convergence along the $y = x$ identity line across all price brackets ($R^2 = 0.7401$).
* **Residual Error Distribution:** Errors $(y - \hat{y})$ are symmetrically distributed around zero (mean residual ~\$1,836), confirming no systematic positive or negative valuation bias.
* **Feature Importance:** Gradient boosting gain scores reveal that living space (`sqft`), neighborhood location (`Downtown`, `Suburbs`), and local safety (`crime_rate_index`) represent the dominant valuation drivers.

### 5. Production Pipeline Packaging
The preprocessing pipeline and optimized estimator are packaged into a unified Scikit-Learn `Pipeline` object:
```python
production_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', best_model)
])

# Direct inference on raw property dictionary
valuation = production_pipeline.predict(raw_listing_df)
```

---

## 📂 Repository Structure

```
├── House_Price_Prediction.ipynb   # Fully executed publication-grade notebook
├── real_estate.db                 # SQLite relational database (houses, neighborhoods)
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation and performance benchmarks
```

---

## 🚀 How to Run Locally

1. Clone this repository:
```bash
git clone https://github.com/Neabem/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or: pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

3. Launch Jupyter:
```bash
jupyter lab House_Price_Prediction.ipynb
```
