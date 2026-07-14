# Customer Churn Prediction
**Big Data Analytics**

## Project Overview
This project aims to predict customer churn in a digital music subscription service using machine learning techniques. The dataset is sourced from the KKBOX Churn Prediction Challenge (Kaggle) and represents large-scale user behavior, transaction history, and activity logs.
The project covers the complete big data analytics pipeline:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Machine learning modeling
- Model evaluation
Two classification models are implemented:
- Logistic Regression as a baseline model  
- XGBoost as an advanced model
  
The results help companies understand churn behavior and design effective, data-driven customer retention strategies.

---

## Dataset
Source: KKBOX Churn Prediction Challenge – Kaggle
www.kaggle.com/competitions/kkbox-churn-prediction-challenge

The dataset consists of four main tables:
- `train_v2.csv` – customer churn labels (`is_churn`)
- `members_v3.csv` – customer demographic information
- `transactions_v2.csv` – subscription and payment history
- `user_logs_v2.csv` – daily user activity logs

Target Variable `is_churn`  
- `1` = churn  
- `0` = not churn  

The dataset contains millions of records (>300 MB) and is highly imbalanced, with approximately 9% churned users.

---

## Methodology

### 1️⃣ Data Preprocessing
Key preprocessing steps include:
- Invalid age values (`≤10` or `>80`) were set as missing values
- Invalid transaction records were removed
- Date columns were converted to datetime format
- A **cut-off date of April 1, 2017** was applied to prevent data leakage
- Consistent filtering between transaction and user activity data
### 2️⃣ Feature Engineering
Features were engineered from multiple data sources.
#### Transaction-based Features
- `total_transactions`
- `total_payment`
- `avg_plan_days`
- `cancel_rate`
- `auto_renew_rate`
- `avg_amt_per_day`
- `autorenew_and_not_cancel`
- `days_since_last_transaction`
- `membership_duration`
- `pct_discount_transactions`
#### Member-based Features
- `long_time_user`
#### User Activity Features
- `total_listening_secs`
- `avg_listening_secs`
- `active_days`
- `last_user_log_date`

All features were aggregated by user ID (`msno`) and merged into a final dataset. Missing values after merging were filled with zero.

**Final Dataset**
- File: `final_data.csv`
- Size: **970,960 rows × 24 features**

After preprocessing, feature engineering, and merging the four original tables, a single dataset (`final_data.csv`) was generated for model development.

| msno | is_churn | city | bd | gender | registered_via | registration_init_time | registration_duration | long_time_user | total_transactions | total_payment | avg_plan_days | cancel_rate | auto_renew_rate | last_membership_expire | avg_amt_per_day | autorenew_and_not_cancel | membership_duration | days_since_last_transaction | pct_discount_transactions | total_listening_secs | avg_listening_secs | active_days | last_user_log_date |
|------|----------|------|----|--------|----------------|------------------------|----------------------:|---------------:|-------------------:|--------------:|--------------:|------------:|-----------------:|------------------------|----------------:|--------------------------:|--------------------:|----------------------------:|--------------------------:|---------------------:|-------------------:|------------:|--------------------|
| ugx0... | 1 | 5 | 28 | male | 3 | 2013-12-23 | 1195 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 80598.557 | 7327.142 | 11 | 2017-03-29 |
| f/Nm... | 1 | 13 | 20 | male | 3 | 2013-12-23 | 1195 | 1 | 1 | 180 | 30 | 0 | 0 | 2017-04-11 | 6.00 | 0 | 31.00 | 21 | 0.00 | 6986.509 | 1164.418 | 6 | 2017-03-20 |
| zLo9... | 1 | 13 | 18 | male | 3 | 2013-12-27 | 1191 | 1 | 2 | 300 | 75 | 0 | 0 | 2017-06-15 | 1.67 | 0 | 48.00 | 18 | 0.00 | 67810.467 | 3390.523 | 20 | 2017-03-31 |
| 8iF/... | 1 | 1 | 0 | 0 | 7 | 2014-01-09 | 1178 | 1 | 10 | 1490 | 30 | 0 | 1 | 2018-01-08 | 4.97 | 1 | 685.10 | 480 | 0.00 | 0.000 | 0.000 | 0 | 0 |
| K6fj... | 1 | 13 | 35 | female | 7 | 2014-01-25 | 1162 | 1 | 8 | 792 | 30 | 0.125 | 1 | 2017-09-18 | 3.30 | 1 | 181.38 | 16 | 0.00 | 239882.241 | 15992.149 | 15 | 2017-03-31 |

Feature Description

| Feature | Description |
|---------|-------------|
| `msno` | Unique customer identifier. |
| `is_churn` | Target variable indicating customer churn (1 = churn, 0 = non-churn). |
| `city` | Customer's registered city code. |
| `bd` | Customer age. |
| `gender` | Customer gender. |
| `registered_via` | Registration channel identifier. |
| `registration_init_time` | Initial registration date. |
| `registration_duration` | Number of days since registration. |
| `long_time_user` | Indicator of whether the customer is a long-term user. |
| `total_transactions` | Total number of subscription transactions. |
| `total_payment` | Total amount paid across all transactions. |
| `avg_plan_days` | Average subscription plan duration (days). |
| `cancel_rate` | Proportion of cancelled subscriptions. |
| `auto_renew_rate` | Proportion of transactions with auto-renew enabled. |
| `last_membership_expire` | Most recent membership expiration date. |
| `avg_amt_per_day` | Average payment per subscription day. |
| `autorenew_and_not_cancel` | Indicator that auto-renew was enabled and the subscription was not cancelled. |
| `membership_duration` | Total membership duration (days). |
| `days_since_last_transaction` | Days elapsed since the last transaction. |
| `pct_discount_transactions` | Percentage of discounted transactions. |
| `total_listening_secs` | Total listening time (seconds). |
| `avg_listening_secs` | Average listening time per active day (seconds). |
| `active_days` | Number of days with listening activity. |
| `last_user_log_date` | Date of the last recorded listening activity. |

## Machine Learning Models

### Logistic Regression
Performance on the test set:
- Accuracy: **0.935**
- Precision: **0.595**
- Recall: **0.885**
- F1-Score: **0.711**
- ROC-AUC: **0.964**
  
The model achieves high recall for churn users, making it effective as a baseline model.
### Random Forest
Performance on the test set:
- Accuracy: **0.981**
- Precision: **0.892**
- Recall: **0.898**
- F1-Score: **0.895**
- ROC-AUC: **1.00**
  
Random Forest significantly outperforms Logistic Regression and effectively captures non-linear relationships between features.

---

## Key Insights
- Churn risk increases near subscription expiration dates
- Auto-renew significantly reduces churn probability
- Recent transaction activity is a stronger churn indicator than total payment
- Churn peaks during the customer evaluation phase (90–365 days)
- Long subscription plans do not always guarantee retention
- High-value customers who churn have significant financial impact

---

## Recommendations
- Implement early interventions before subscription expiration
- Encourage auto-renew activation with incentives
- Launch re-engagement campaigns for inactive users
- Provide loyalty programs for mid-term customers
- Offer more flexible subscription plans
- Personalize services for highly active users

---

