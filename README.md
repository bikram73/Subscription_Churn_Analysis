# 📉 Subscription Churn Analysis (OTT / SaaS)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

An end-to-end churn analytics project that identifies churn drivers, predicts churn probability, and segments high-risk customers using the Telco Customer Churn dataset (adapted to OTT/SaaS PRD requirements).

## ✨ Features

- 📊 **Churn KPI Analysis**: Calculates overall churn rate and group-level churn trends.
- 🧹 **Data Cleaning Pipeline**: Handles missing values and type corrections.
- 🧠 **Feature Engineering**: Builds engagement score, tenure groups, and churn-ready features.
- 📈 **Rich Visualizations**: Tenure, usage, pricing, support behavior, and correlation heatmap.
- 🤖 **ML Modeling**: Logistic Regression and Random Forest model training + evaluation.
- 🎯 **Risk Segmentation**: Generates churn probabilities and classifies users into Low/Medium/High risk.
- 💼 **Business Insights**: Produces actionable retention recommendations.

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

## 📁 Project Structure

```text
Subscription_Churn_Analysis/
├── Telco_Customer_Churn.csv
├── README.md
├── analysis_summary.md
├── subscription_churn.csv
├── subscription_churn.ipynb
├── subscription_churn.py
├── outputs/
│   ├── churn_rate.png
│   ├── tenure_churn.png
│   ├── usage_churn.png
│   ├── charges_churn.png
│   ├── support_churn.png
│   ├── correlation_heatmap.png
│   ├── model_accuracy.png
│   └── feature_importance.png
└── requirements.txt
```

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/Subscription_Churn_Analysis.git
cd Subscription_Churn_Analysis
```

2. **Create and activate virtual environment (recommended)**

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Run Python Pipeline

```bash
python subscription_churn.py
```

This generates:

- `subscription_churn.csv` (normalized + scored dataset)
- `analysis_summary.md` (insights + model performance + recommendations)
- `outputs/*.png` (all required charts)

### Run Notebook Workflow

Open `subscription_churn.ipynb` and run all cells in order.

- Notebook visualizations are shown inline.
- Notebook cells are designed to analyze without saving charts to `outputs/`.

## 📌 Key Objectives Covered

- Calculate churn rate
- Identify major churn factors
- Segment high-risk users
- Build and evaluate prediction models
- Recommend retention strategies

## 🧾 Dataset Note

Source dataset (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The Telco dataset does not directly contain OTT telemetry fields like exact app usage frequency, last login days, and customer support call counts.

To match the PRD schema, this project engineers deterministic proxy features from available service and support columns and clearly documents this in the analysis outputs.

## 📜 License

This project is licensed under the MIT License.

- See LICENSE
