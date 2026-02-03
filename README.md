# 🛡️ E-Commerce Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning pipeline for detecting fraudulent e-commerce transactions using gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) with feature engineering, hyperparameter tuning, and production-ready model artifacts.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Key Findings](#-key-findings)
- [Data Visualizations](#-data-visualizations)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation & Execution](#-installation--execution)
- [References & Resources](#-references--resources)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🎯 Problem Statement

### The Challenge

E-commerce platforms lose billions of dollars annually to fraudulent transactions. The challenge is particularly acute in:

| Fraud Type | Description | Business Impact |
|------------|-------------|-----------------|
| **Promo Code Abuse** | Exploiting promotional offers through fake accounts | Direct revenue loss |
| **Multi-Account Fraud** | Creating multiple accounts to claim bonuses repeatedly | Marketing budget drain |
| **New Account Fraud** | Fresh accounts used for high-value fraudulent purchases | Inventory & financial loss |
| **Payment Fraud** | Stolen cards or unauthorized payment methods | Chargebacks & fees |

### Why This Is Difficult

1. **Severe Class Imbalance**: Only ~5% of transactions are fraudulent (1:18 ratio)
2. **Low Linear Correlations**: Fraud patterns are non-linear and complex
3. **Real-Time Requirements**: Decisions must be made in milliseconds
4. **Adversarial Evolution**: Fraudsters continuously adapt their techniques

---

## 💡 Solution Overview

### Approach

This project implements a **multi-model ensemble approach** using gradient-boosted decision trees, which excel at:
- Handling imbalanced datasets
- Capturing non-linear relationships
- Providing interpretable feature importance
- Fast inference for production deployment

### Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Loading   │────▶│   EDA & Data     │────▶│    Feature      │
│  & Cleaning     │     │   Quality Check  │     │   Engineering   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Deployment    │◀────│  Hyperparameter  │◀────│ Model Training  │
│   Artifacts     │     │     Tuning       │     │ (3 Algorithms)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Models Implemented

| Model | Role | Strength |
|-------|------|----------|
| **XGBoost** | Primary model (post-tuning) | Best AUC-ROC (0.8368) |
| **LightGBM** | Fast alternative | Fastest inference |
| **CatBoost** | Categorical handler | Native categorical support |

---

## 🔍 Key Findings

### 1. 🚨 New Account Fraud Risk (Critical Discovery)

> **New accounts (<7 days old) are 4.75x more likely to be fraudulent**

| Account Type | Fraud Rate | Dataset % |
|--------------|------------|-----------|
| New (<7 days) | **22.73%** | 2.14% |
| Established (≥7 days) | 4.79% | 97.86% |

**Business Implication**: Implement progressive trust systems—new accounts should have limited access to promotions and higher transaction scrutiny.

### 2. 📊 Feature Importance Rankings

The top predictors of fraud (from XGBoost model):

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | **Account Age Days** | 38.29% | Newer accounts = higher risk |
| 2 | Account Age Bin | 14.43% | Binned age groups add predictive power |
| 3 | Is Night | 11.18% | Night-time transactions riskier |
| 4 | Transaction Hour | 7.50% | Time-based patterns exist |
| 5 | Amount (log-transformed) | 7.48% | Normalized amount signal |

### 3. 📈 Performance Improvements Through Tuning

| Model | Baseline AUC | Tuned AUC | Improvement |
|-------|-------------|-----------|-------------|
| XGBoost | 0.7885 | **0.8368** | **+6.12%** |
| LightGBM | 0.7872 | 0.8313 | +5.59% |
| CatBoost | 0.7995 | 0.8238 | +3.04% |

**Recall Improvement** (catching actual fraud):
- XGBoost: 45.08% → **74.18%** (+64.55% relative improvement!)
- LightGBM: 54.51% → 73.36% (+34.59%)
- CatBoost: 60.66% → 72.54% (+19.59%)

### 4. 💳 Payment & Category Patterns

- **Payment Methods**: Different fraud rates across payment types (useful for rule-based filtering)
- **Product Categories**: Electronics historically higher risk (resale fraud potential)
- **Device Distribution**: Balanced across desktop/mobile/tablet (~33% each)

### 5. 📉 Low Linear Correlations

| Feature | Correlation with Fraud | Observation |
|---------|------------------------|-------------|
| Transaction Amount | +0.28 | Weak positive |
| Account Age | -0.14 | Weak negative |
| Transaction Hour | -0.12 | Very weak |
| Customer Age | 0.00 | No linear relationship |

**Key Insight**: Low correlations confirm that **tree-based ensemble models** are the right choice—they capture the non-linear, interaction-based fraud patterns that linear models miss.

---

## 📊 Data Visualizations

### EDA Correlation Analysis

![EDA Correlation Heatmap](eda_correlation.png)

This correlation heatmap shows the relationships between features and fraud status. Key observations:
- Low linear correlations confirm the need for non-linear models
- Account age shows a negative correlation with fraud
- Transaction amount shows a weak positive correlation

### Model Performance & Tuning Results

![Final Results Dashboard](final_results.png)

This comprehensive dashboard shows:
- **Model Comparison**: Side-by-side AUC, F1, Precision, Recall metrics
- **Tuning Results**: Before vs After hyperparameter optimization
- **Performance Improvements**: All models improved to ~0.82+ AUC
- **Best Model**: XGBoost achieved 0.8368 AUC with 74.18% Recall

---

## 🏆 Model Performance

### Final Model: Tuned XGBoost

```
Classification Report:
              precision    recall  f1-score   support

  Legitimate       0.98      0.80      0.88      4483
       Fraud       0.16      0.74      0.26       244

    accuracy                           0.79      4727
   macro avg       0.57      0.77      0.57      4727
```

### Model Selection Guide

| Use Case | Recommended Model | Reasoning |
|----------|-------------------|-----------|
| **Maximize Fraud Detection** | Tuned XGBoost | 74.18% recall - catches most fraud |
| **Minimize False Alerts** | Baseline XGBoost | Higher precision |
| **Production Speed** | LightGBM | Fastest inference time |
| **Categorical Features** | CatBoost | Native handling, no encoding needed |

---

## 📁 Project Structure

```
E-Commerce-Fraud-Detection/
│
├── 📓 Jupyter Notebook
│   └── Fraud_Detection_Pipeline_Updated.ipynb  # Complete ML pipeline notebook
│
├── 📊 Data Files
│   ├── Dataset/
│   │   ├── Fraudulent_E-Commerce_Transaction_Data.csv        # Original full dataset
│   │   └── Fraudulent_E-Commerce_Transaction_Data_2.csv      # Sample dataset
│   └── Fraudulent_E-Commerce_Transaction_Data_Cleaned.csv    # Cleaned dataset (ready for training)
│
├── 🤖 Model Artifacts (Tuned Models)
│   ├── xgboost_tuned_model.pkl      # Best performing model (AUC: 0.8368)
│   ├── lightgbm_tuned_model.pkl     # Fast alternative model
│   ├── catboost_tuned_model.pkl     # Categorical feature specialist
│   └── feature_columns.pkl          # Feature column names for inference
│
├── 📈 Results & Metrics
│   ├── feature_importance.csv       # Feature importance rankings
│   └── tuning_comparison.csv        # Before/After tuning comparison
│
├── 🖼️ Visualizations
│   ├── eda_correlation.png          # Correlation heatmap
│   └── final_results.png            # Model performance dashboard
│
├── 📄 Documentation
│   ├── README.md                    # This file
│   └── .gitignore                   # Git ignore configuration
│
└── 📁 .git/                         # Git repository
```

---

## 🚀 Installation & Execution

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook (for `.ipynb` execution)
- 8GB+ RAM recommended (for full dataset processing)
- GPU optional (for faster CatBoost training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kaustav2023/E-Commerce-Fraud-Detection.git
cd E-Commerce-Fraud-Detection
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna joblib tqdm
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna joblib tqdm
```

### Step 3: Run the Jupyter Notebook

#### Option A: Using Jupyter Notebook (Recommended)

```bash
jupyter notebook Fraud_Detection_Pipeline_Updated.ipynb
```

Then execute cells sequentially (Shift + Enter).

#### Option B: Using Google Colab

1. Upload `Fraud_Detection_Pipeline_Updated.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload the dataset from the `Dataset/` folder when prompted
3. Run all cells (Runtime → Run all)

> **Note**: The notebook was originally developed and tested on Google Colab with GPU acceleration.

### Step 4: Expected Outputs

After execution, you will have:
- ✅ EDA visualizations (`eda_correlation.png`)
- ✅ Tuned models (`xgboost_tuned_model.pkl`, `lightgbm_tuned_model.pkl`, `catboost_tuned_model.pkl`)
- ✅ Feature columns for inference (`feature_columns.pkl`)
- ✅ Performance metrics (`feature_importance.csv`, `tuning_comparison.csv`)
- ✅ Results visualization (`final_results.png`)

### Using Pre-trained Models

```python
import joblib
import pandas as pd

# Load the tuned model
model = joblib.load('xgboost_tuned_model.pkl')
feature_cols = joblib.load('feature_columns.pkl')

# Prepare your data
new_transaction = pd.DataFrame({...})  # Your transaction data
new_transaction_features = new_transaction[feature_cols]

# Predict
fraud_probability = model.predict_proba(new_transaction_features)[:, 1]
is_fraud = (fraud_probability > 0.5).astype(int)
```

---

## 📚 References & Resources

### Research Papers & Technical References

| Resource | Description | Link |
|----------|-------------|------|
| **IEEE-CIS Fraud Detection Competition** | 1st place solution with 590K transactions, feature engineering patterns | [Kaggle Discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284) |
| **IEEE-CIS Winner Code** | Complete winning implementation with XGBoost | [Kaggle Notebook](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600) |
| **AWS Real-time Fraud Detection with GNN** | Enterprise-grade GNN solution on AWS | [GitHub](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) |
| **Credit Card Fraud Detection Dataset** | Classic imbalanced fraud dataset (284K transactions) | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Promo Code Abuse Detection** | Direct promo abuse implementation | [GitHub](https://github.com/Tasha-14/PromoCode-Abuse-Detection-Model) |

### ML Frameworks Documentation

| Framework | Purpose | Documentation |
|-----------|---------|---------------|
| **XGBoost** | Gradient boosting | [Docs](https://xgboost.readthedocs.io/) |
| **LightGBM** | Fast gradient boosting | [Docs](https://lightgbm.readthedocs.io/) |
| **CatBoost** | Categorical boosting | [Docs](https://catboost.ai/docs/) |
| **Optuna** | Hyperparameter optimization | [Docs](https://optuna.readthedocs.io/) |
| **Scikit-learn** | ML utilities | [Docs](https://scikit-learn.org/) |

### Graph-Based Fraud Detection Resources

| Resource | Description | Link |
|----------|-------------|------|
| **NetworkX** | Graph analysis library | [GitHub](https://github.com/networkx/networkx) |
| **PyTorch Geometric** | GNN for fraud detection | [GitHub](https://github.com/pyg-team/pytorch_geometric) |
| **Deep Graph Library (DGL)** | Scalable GNN framework | [GitHub](https://github.com/dmlc/dgl) |
| **Amazon Fraud Detection Benchmark** | Standardized fraud datasets | [GitHub](https://github.com/amazon-science/fraud-dataset-benchmark) |

### Industry Solutions (Reference Only)

| Platform | Specialization | Link |
|----------|---------------|------|
| Sift | Promo abuse, payment fraud | [sift.com](https://sift.com/) |
| Ravelin | Referral fraud, multi-accounting | [ravelin.com](https://www.ravelin.com/) |
| Stripe Radar | Payment fraud rules engine | [stripe.com/radar](https://stripe.com/radar) |

---

## 🔮 Future Enhancements

### Planned Improvements

1. **Graph-Based Detection**
   - Implement NetworkX for referral ring detection
   - Add device/IP sharing analysis
   - Build user-device bipartite graphs

2. **Real-Time Scoring**
   - Deploy models via Flask/FastAPI
   - Add feature store integration (Feast)
   - Implement streaming with Apache Flink

3. **Deep Learning**
   - Experiment with Graph Neural Networks (GNN)
   - Add sequence models for user behavior
   - Implement attention mechanisms

4. **MLOps**
   - Model monitoring and drift detection
   - A/B testing framework
   - Automated retraining pipeline

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out via GitHub.

---

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b>
</p>
