# Next Purchase Date Predictor

Predicting when an individual customer will next 
purchase a specific product using Machine Learning 
on the Instacart grocery dataset.

---

## Results

| Model | Within 1 Day | Within 3 Days | MAE |
|-------|-------------|---------------|-----|
| ANN (Best) | 30.8% | 56.3% | 6.15 days |
| XGBoost | 21.4% | 46.2% | 5.88 days |
| RNN 2-feature | 12.0% | 32.4% | 7.87 days |
| Linear Regression | 10.3% | 29.4% | 7.67 days |
| Baseline (avg) | 9.5% | 28.2% | 7.18 days |

ANN achieved **3x improvement** over baseline 
on the 1-day accuracy metric.

---

## Problem Statement

Given a customer and a product, predict how many 
days until their next purchase. This enables 
personalised marketing — sending reminders or 
offers at exactly the right time.

---

## Dataset

Instacart Market Basket Analysis (Kaggle)
- 206,000 unique customers
- 50,000 unique products  
- 3 million+ orders

Download from: https://www.kaggle.com/c/instacart-market-basket-analysis

---

## Approach

### 1. Exploratory Data Analysis
Analysed purchase patterns across customers.
Found weekly, biweekly and monthly shopping 
rhythms — confirming need for personalised 
prediction over a global average.

### 2. RFM Clustering
Built RFM (Recency, Frequency, Monetary) profiles
per customer. Applied K-Means clustering with 
k=6 (elbow method) to segment customers into 
6 behavioural groups.

### 3. Feature Engineering
Used sliding window technique on purchase gap 
sequences to create tabular features:
- t1 to t5: last 5 purchase gaps
- avg_days: historical average gap
- max_days, min_days: range of gaps
- days_since: days since last purchase
- t0: target variable (next gap)

Training set: 345,472 rows
Test set: 112,796 rows

### 4. Models Trained
- Linear Regression (per-customer polynomial)
- RNN 1-feature (custom PyTorch)
- RNN 2-feature (custom PyTorch)
- XGBoost (RandomizedSearchCV tuning)
- ANN — Keras (best model)

---

## Tech Stack

- Python 3.7
- PyTorch — custom RNN implementation
- Keras / TensorFlow — ANN
- XGBoost — gradient boosted trees
- scikit-learn — preprocessing, evaluation
- Pandas, NumPy — data manipulation
- Matplotlib, Seaborn — visualisation

---

## Project Structure

next-purchase-date-predictor/
├── notebooks/
│   ├── Data_Exploration.ipynb
│   ├── Clustering_RFM.ipynb
│   ├── Train_Linear_Regression.ipynb
│   ├── Train_RNN1.ipynb
│   ├── Train_RNN2.ipynb
│   ├── Train_XGBoost.ipynb
│   ├── Train_Neural_Network.ipynb
│   └── Results_graphs.ipynb
├── src/
│   └── MyRNN.py
├── features/
│   ├── df_features_final.pkl
│   ├── df_RFM.pkl
│   └── df_test_final.pkl
├── results/
│   └── results_rnn1_rnn2_xgb_reg1_reg2_nn_xgbh.pkl
├── requirements.txt
└── README.md