# 🏠 Real Estate Price Prediction (Regression Pipeline)

An end-to-end Machine Learning regression pipeline designed to predict continuous property selling prices for a real estate investment firm.

This project demonstrates a complete production-style workflow: joining multi-table relational data in SQL, handling corrupted/missing database records inside a leak-free Scikit-Learn `Pipeline`, and optimizing an `XGBRegressor` via cross-validated hyperparameter tuning (`RandomizedSearchCV`).

## 🛠️ Tech Stack
* **Database / SQL:** `SQLite3` (`JOIN` across `houses` and `neighborhoods` tables)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Machine Learning:** `Scikit-Learn` (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `RandomizedSearchCV`), `XGBoost` (`XGBRegressor`)

## 📊 Project Architecture

### 1. Relational Data Extraction (SQL)
Property records and neighborhood-level macroeconomic indicators (`crime_rate_index`, `school_rating`) were stored across separate tables inside `real_estate.db`.
* Executed a relational `JOIN` query on `neighborhood_id` to assemble the unified modeling dataset directly from SQLite.
* Dropped non-predictive primary/foreign keys (`house_id`, `neighborhood_id`) prior to modeling to prevent data leakage.

### 2. Leak-Free Imputation & Preprocessing Pipeline (Scikit-Learn)
Real-world databases contain missing values (`NULL`s). Rather than dropping rows and losing valuable signal, a multi-stage `ColumnTransformer` + `Pipeline` was engineered:
* **Numeric Pipeline** (`sqft`, `year_built`, `crime_rate_index`, `bedrooms`, `bathrooms`, `school_rating`): Imputed missing values using `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
* **Categorical Pipeline** (`has_garage`, `neighborhood_name`): Imputed missing entries using `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
* Strictly applied `fit_transform` on `X_train` and `transform` on `X_test` to guarantee zero data leakage.

### 3. Model Training & Hyperparameter Tuning (`XGBRegressor`)
* **Baseline Model:** Trained an untuned `XGBRegressor`, achieving a baseline Mean Absolute Error (MAE) of **$26,092** on unseen test properties.
* **Hyperparameter Optimization:** Configured a 5-fold cross-validated `RandomizedSearchCV` tuning `max_depth`, `n_estimators`, and `learning_rate`.
* **Final Performance:** The tuned estimator reduced unseen test error down to **$23,831 MAE**—beating the baseline by over **$2,260 per property**.

## 🚀 How to Explore
1. Clone this repository.
2. Open `House_Price_Prediction.ipynb` (or `Step_1_and_2.ipynb`) to view the SQL extraction, pipeline architecture, hyperparameter search, and evaluation metrics.
