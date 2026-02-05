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
- Random Forest as an advanced model
The results help companies understand churn behavior and design effective, data-driven customer retention strategies.

---

## Dataset
Source: KKBOX Churn Prediction Challenge – Kaggle
The dataset consists of several large tables:
- `train.csv` – customer churn labels (`is_churn`)
- `members.csv` – customer demographic information
- `transactions.csv` – subscription and payment history
- `user_logs.csv` – daily user activity logs

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

---

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

