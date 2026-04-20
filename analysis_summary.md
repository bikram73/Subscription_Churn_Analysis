# Subscription Churn Analysis Summary

## Project Snapshot
- Dataset: Telco Customer Churn (adapted for OTT/SaaS PRD fields)
- Total customers analyzed: 7032
- Overall churn rate: 26.58%

## Key Findings
- Highest churn tenure segment: New (47.68%)
- Highest churn support-call level: 6 calls (59.97%)
- Lowest usage-frequency churn point: 9 (5.29%)
- Highest usage-frequency churn point: 3 (44.92%)

## Model Performance
- Logistic Regression Accuracy: 0.7946
- Random Forest Accuracy: 0.7584
- Random Forest Precision: 0.5491
- Random Forest Recall: 0.5080
- Random Forest F1: 0.5278
- Random Forest ROC-AUC: 0.7887

## Risk Segmentation
- High-risk users (probability >= 0.70): 1392 (19.80%)
- Segments: Low (<= 0.40), Medium (0.40-0.70), High (> 0.70)

## Top Feature Signals
- num__monthly_charges: 0.3282
- num__tenure: 0.2383
- num__customer_support_calls: 0.1190
- cat__subscription_plan_Month-to-month: 0.1061
- num__last_login_days: 0.0549

## Recommendations
- Offer targeted discounts to high-risk users on monthly contracts.
- Improve onboarding during the first 12 months to reduce early churn.
- Proactively support users with low usage and weak engagement scores.
- Prioritize service quality interventions for users with high inferred support-contact behavior.

## Notes
- usage_frequency, last_login_days, and customer_support_calls are engineered proxies because the Telco dataset does not contain direct OTT app telemetry fields.