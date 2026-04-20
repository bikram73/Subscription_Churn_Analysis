# Subscription Churn Analysis (OTT / SaaS PRD Implementation)

This project implements an end-to-end churn analytics workflow using the Telco churn dataset, adapted to the OTT/SaaS PRD requirements.

## Project Structure

```
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
├── requirements.txt
```

## Objectives Covered

- Calculate churn rate
- Identify churn drivers through EDA and correlation analysis
- Segment high-risk users with churn probability scoring
- Build and evaluate churn prediction models
- Provide actionable retention recommendations

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run analysis:
   - `python subscription_churn.py`
3. Run notebook:
   - Open `subscription_churn.ipynb` and run all cells (visuals are shown inline and are not saved to `outputs/` from the notebook workflow)

## Outputs

Running `subscription_churn.py` will generate:

- `subscription_churn.csv`: normalized dataset aligned with PRD fields plus engineered features and churn probability
- `analysis_summary.md`: analysis findings, model metrics, and recommendations
- `outputs/*.png`: all required visualizations

## Data Note

The Telco dataset does not directly provide OTT telemetry fields like explicit usage frequency, last login days, and support call counts. This implementation engineers deterministic proxy features from available service and support columns, and documents this in the summary.