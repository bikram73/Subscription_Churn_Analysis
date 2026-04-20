from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT_DIR / "Telco_Customer_Churn.csv"
NORMALIZED_DATA_PATH = ROOT_DIR / "subscription_churn.csv"
SUMMARY_PATH = ROOT_DIR / "analysis_summary.md"
OUTPUT_DIR = ROOT_DIR / "outputs"


def ensure_output_dir() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data() -> pd.DataFrame:
	df = pd.read_csv(RAW_DATA_PATH)

	df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
	df = df.dropna(subset=["TotalCharges"]).copy()

	df["usage_frequency"] = (
		(df["OnlineSecurity"] == "Yes").astype(int)
		+ (df["OnlineBackup"] == "Yes").astype(int)
		+ (df["DeviceProtection"] == "Yes").astype(int)
		+ (df["TechSupport"] == "Yes").astype(int)
		+ (df["StreamingTV"] == "Yes").astype(int)
		+ (df["StreamingMovies"] == "Yes").astype(int)
		+ (df["PhoneService"] == "Yes").astype(int)
		+ (df["MultipleLines"] == "Yes").astype(int)
		+ (df["InternetService"] != "No").astype(int)
	)

	inferred_login_days = (
		30 / (df["usage_frequency"] + 1)
		+ np.where(df["Contract"] == "Month-to-month", 5, 0)
		+ np.where(df["tenure"] < 6, 7, 0)
	)
	df["last_login_days"] = np.clip(np.round(inferred_login_days), 1, 60).astype(int)

	inferred_support_calls = (
		np.where(df["TechSupport"] == "No", 3, 1)
		+ np.where(df["OnlineSecurity"] == "No", 1, 0)
		+ np.where(df["OnlineBackup"] == "No", 1, 0)
		+ np.where(df["InternetService"] == "Fiber optic", 1, 0)
	)
	df["customer_support_calls"] = np.clip(inferred_support_calls, 0, 8).astype(int)

	normalized = pd.DataFrame(
		{
			"customer_id": df["customerID"],
			"subscription_plan": df["Contract"],
			"monthly_charges": df["MonthlyCharges"],
			"tenure": df["tenure"],
			"usage_frequency": df["usage_frequency"],
			"last_login_days": df["last_login_days"],
			"customer_support_calls": df["customer_support_calls"],
			"payment_method": df["PaymentMethod"],
			"churn": df["Churn"],
		}
	)

	normalized["churn_flag"] = normalized["churn"].map({"Yes": 1, "No": 0})
	normalized["engagement_score"] = normalized["usage_frequency"] / (
		normalized["last_login_days"] + 1
	)
	normalized["tenure_group"] = pd.cut(
		normalized["tenure"],
		bins=[-1, 12, 24, 72],
		labels=["New", "Mid", "Loyal"],
	)

	return normalized


def save_plot_churn_rate(df: pd.DataFrame) -> float:
	churn_rate = df["churn_flag"].mean() * 100
	counts = df["churn"].value_counts().reindex(["No", "Yes"])

	plt.figure(figsize=(6, 4))
	sns.barplot(x=counts.index, y=counts.values, palette=["#4e79a7", "#e15759"])
	plt.title(f"Customer Churn Distribution (Rate: {churn_rate:.2f}%)")
	plt.xlabel("Churn")
	plt.ylabel("Customer Count")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "churn_rate.png", dpi=150)
	plt.close()

	return churn_rate


def save_plot_tenure_churn(df: pd.DataFrame) -> pd.Series:
	churn_by_tenure = df.groupby("tenure_group", observed=False)["churn_flag"].mean() * 100

	plt.figure(figsize=(7, 4))
	sns.barplot(x=churn_by_tenure.index.astype(str), y=churn_by_tenure.values, color="#f28e2b")
	plt.title("Churn Rate by Tenure Group")
	plt.xlabel("Tenure Group")
	plt.ylabel("Churn Rate (%)")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "tenure_churn.png", dpi=150)
	plt.close()

	return churn_by_tenure


def save_plot_usage_churn(df: pd.DataFrame) -> pd.Series:
	churn_by_usage = df.groupby("usage_frequency")["churn_flag"].mean() * 100

	plt.figure(figsize=(8, 4))
	sns.lineplot(x=churn_by_usage.index, y=churn_by_usage.values, marker="o", color="#59a14f")
	plt.title("Churn Rate by Usage Frequency")
	plt.xlabel("Usage Frequency (engineered activity score)")
	plt.ylabel("Churn Rate (%)")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "usage_churn.png", dpi=150)
	plt.close()

	return churn_by_usage


def save_plot_charges_churn(df: pd.DataFrame) -> None:
	plt.figure(figsize=(7, 4))
	sns.boxplot(x="churn", y="monthly_charges", data=df, palette=["#76b7b2", "#e15759"])
	plt.title("Monthly Charges vs Churn")
	plt.xlabel("Churn")
	plt.ylabel("Monthly Charges")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "charges_churn.png", dpi=150)
	plt.close()


def save_plot_support_churn(df: pd.DataFrame) -> pd.Series:
	churn_by_support = df.groupby("customer_support_calls")["churn_flag"].mean() * 100

	plt.figure(figsize=(8, 4))
	sns.lineplot(
		x=churn_by_support.index,
		y=churn_by_support.values,
		marker="o",
		color="#edc948",
	)
	plt.title("Churn Rate by Customer Support Calls")
	plt.xlabel("Customer Support Calls (engineered estimate)")
	plt.ylabel("Churn Rate (%)")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "support_churn.png", dpi=150)
	plt.close()

	return churn_by_support


def save_plot_correlation(df: pd.DataFrame) -> None:
	corr_cols = [
		"monthly_charges",
		"tenure",
		"usage_frequency",
		"last_login_days",
		"customer_support_calls",
		"engagement_score",
		"churn_flag",
	]

	corr = df[corr_cols].corr(numeric_only=True)

	plt.figure(figsize=(8, 6))
	sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
	plt.title("Correlation Heatmap")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
	plt.close()


def train_models(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, Pipeline]:
	feature_cols = [
		"monthly_charges",
		"tenure",
		"usage_frequency",
		"last_login_days",
		"customer_support_calls",
		"subscription_plan",
		"payment_method",
	]

	X = df[feature_cols]
	y = df["churn_flag"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	numeric_cols = [
		"monthly_charges",
		"tenure",
		"usage_frequency",
		"last_login_days",
		"customer_support_calls",
	]
	categorical_cols = ["subscription_plan", "payment_method"]

	preprocess = ColumnTransformer(
		transformers=[
			("num", StandardScaler(), numeric_cols),
			("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
		]
	)

	logistic_model = Pipeline(
		steps=[
			("preprocess", preprocess),
			("model", LogisticRegression(max_iter=2000, random_state=42)),
		]
	)
	logistic_model.fit(X_train, y_train)

	rf_model = Pipeline(
		steps=[
			("preprocess", preprocess),
			(
				"model",
				RandomForestClassifier(
					n_estimators=300,
					random_state=42,
					class_weight="balanced",
				),
			),
		]
	)
	rf_model.fit(X_train, y_train)

	metrics = {}
	for model_name, model in {
		"Logistic Regression": logistic_model,
		"Random Forest": rf_model,
	}.items():
		preds = model.predict(X_test)
		probs = model.predict_proba(X_test)[:, 1]
		metrics[model_name] = {
			"accuracy": accuracy_score(y_test, preds),
			"precision": precision_score(y_test, preds),
			"recall": recall_score(y_test, preds),
			"f1": f1_score(y_test, preds),
			"roc_auc": roc_auc_score(y_test, probs),
		}

	full_proba = rf_model.predict_proba(X)[:, 1]
	scored_df = df.copy()
	scored_df["churn_probability"] = full_proba
	scored_df["risk_segment"] = pd.cut(
		scored_df["churn_probability"],
		bins=[0, 0.4, 0.7, 1.0],
		labels=["Low", "Medium", "High"],
		include_lowest=True,
	)

	return metrics, scored_df, rf_model


def save_plot_model_accuracy(metrics: dict) -> None:
	model_names = list(metrics.keys())
	accuracies = [metrics[name]["accuracy"] * 100 for name in model_names]

	plt.figure(figsize=(7, 4))
	sns.barplot(x=model_names, y=accuracies, palette=["#4e79a7", "#f28e2b"])
	plt.title("Model Accuracy Comparison")
	plt.xlabel("Model")
	plt.ylabel("Accuracy (%)")
	plt.ylim(0, 100)
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "model_accuracy.png", dpi=150)
	plt.close()


def save_plot_feature_importance(model: Pipeline, feature_df: pd.DataFrame) -> pd.DataFrame:
	preprocess = model.named_steps["preprocess"]
	rf_estimator = model.named_steps["model"]

	feature_names = preprocess.get_feature_names_out(feature_df.columns)
	importances = rf_estimator.feature_importances_

	importance_df = pd.DataFrame(
		{"feature": feature_names, "importance": importances}
	).sort_values("importance", ascending=False)

	top_features = importance_df.head(10)

	plt.figure(figsize=(9, 5))
	sns.barplot(data=top_features, y="feature", x="importance", color="#59a14f")
	plt.title("Top Feature Importances (Random Forest)")
	plt.xlabel("Importance")
	plt.ylabel("Feature")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
	plt.close()

	return importance_df


def write_summary(
	churn_rate: float,
	churn_by_tenure: pd.Series,
	churn_by_usage: pd.Series,
	churn_by_support: pd.Series,
	metrics: dict,
	scored_df: pd.DataFrame,
	importance_df: pd.DataFrame,
) -> None:
	high_risk_count = (scored_df["risk_segment"] == "High").sum()
	high_risk_pct = high_risk_count / len(scored_df) * 100

	top_tenure = churn_by_tenure.sort_values(ascending=False).head(1)
	top_support = churn_by_support.sort_values(ascending=False).head(1)

	top_rf = metrics["Random Forest"]

	summary_lines = [
		"# Subscription Churn Analysis Summary",
		"",
		"## Project Snapshot",
		"- Dataset: Telco Customer Churn (adapted for OTT/SaaS PRD fields)",
		f"- Total customers analyzed: {len(scored_df)}",
		f"- Overall churn rate: {churn_rate:.2f}%",
		"",
		"## Key Findings",
		f"- Highest churn tenure segment: {top_tenure.index[0]} ({top_tenure.iloc[0]:.2f}%)",
		f"- Highest churn support-call level: {int(top_support.index[0])} calls ({top_support.iloc[0]:.2f}%)",
		f"- Lowest usage-frequency churn point: {churn_by_usage.idxmin()} ({churn_by_usage.min():.2f}%)",
		f"- Highest usage-frequency churn point: {churn_by_usage.idxmax()} ({churn_by_usage.max():.2f}%)",
		"",
		"## Model Performance",
		f"- Logistic Regression Accuracy: {metrics['Logistic Regression']['accuracy']:.4f}",
		f"- Random Forest Accuracy: {top_rf['accuracy']:.4f}",
		f"- Random Forest Precision: {top_rf['precision']:.4f}",
		f"- Random Forest Recall: {top_rf['recall']:.4f}",
		f"- Random Forest F1: {top_rf['f1']:.4f}",
		f"- Random Forest ROC-AUC: {top_rf['roc_auc']:.4f}",
		"",
		"## Risk Segmentation",
		f"- High-risk users (probability >= 0.70): {high_risk_count} ({high_risk_pct:.2f}%)",
		"- Segments: Low (<= 0.40), Medium (0.40-0.70), High (> 0.70)",
		"",
		"## Top Feature Signals",
	]

	for _, row in importance_df.head(5).iterrows():
		summary_lines.append(f"- {row['feature']}: {row['importance']:.4f}")

	summary_lines.extend(
		[
			"",
			"## Recommendations",
			"- Offer targeted discounts to high-risk users on monthly contracts.",
			"- Improve onboarding during the first 12 months to reduce early churn.",
			"- Proactively support users with low usage and weak engagement scores.",
			"- Prioritize service quality interventions for users with high inferred support-contact behavior.",
			"",
			"## Notes",
			"- usage_frequency, last_login_days, and customer_support_calls are engineered proxies because the Telco dataset does not contain direct OTT app telemetry fields.",
		]
	)

	SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
	ensure_output_dir()

	df = load_and_prepare_data()
	churn_rate = save_plot_churn_rate(df)
	churn_by_tenure = save_plot_tenure_churn(df)
	churn_by_usage = save_plot_usage_churn(df)
	save_plot_charges_churn(df)
	churn_by_support = save_plot_support_churn(df)
	save_plot_correlation(df)

	metrics, scored_df, best_model = train_models(df)
	save_plot_model_accuracy(metrics)
	importance_df = save_plot_feature_importance(
		best_model,
		scored_df[
			[
				"monthly_charges",
				"tenure",
				"usage_frequency",
				"last_login_days",
				"customer_support_calls",
				"subscription_plan",
				"payment_method",
			]
		],
	)

	scored_df.to_csv(NORMALIZED_DATA_PATH, index=False)

	write_summary(
		churn_rate=churn_rate,
		churn_by_tenure=churn_by_tenure,
		churn_by_usage=churn_by_usage,
		churn_by_support=churn_by_support,
		metrics=metrics,
		scored_df=scored_df,
		importance_df=importance_df,
	)

	print("Project build complete.")
	print(f"Normalized dataset: {NORMALIZED_DATA_PATH}")
	print(f"Summary file: {SUMMARY_PATH}")
	print(f"Charts directory: {OUTPUT_DIR}")
	print("Random Forest metrics:")
	print({k: round(v, 4) for k, v in metrics["Random Forest"].items()})


if __name__ == "__main__":
	main()
