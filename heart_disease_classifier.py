"""
Heart Disease Prediction using Multiple Classification Algorithms
=================================================================
Algorithms implemented:
    1. Logistic Regression
    2. K-Nearest Neighbors (KNN)
    3. Decision Tree
    4. Random Forest
    5. Support Vector Machine (SVM)
    6. Gradient Boosting
    7. Naive Bayes
    8. XGBoost (if available)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DATA LOADING & GENERATION
# ─────────────────────────────────────────────

def load_heart_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the Cleveland Heart Disease dataset.
    If no path is given, generates a realistic synthetic dataset for demo.

    Columns (UCI Heart Disease Dataset):
        age       – age in years
        sex       – (1=male, 0=female)
        cp        – chest pain type (0-3)
        trestbps  – resting blood pressure (mm Hg)
        chol      – serum cholesterol (mg/dl)
        fbs       – fasting blood sugar > 120 mg/dl (1=true, 0=false)
        restecg   – resting ECG results (0-2)
        thalach   – maximum heart rate achieved
        exang     – exercise-induced angina (1=yes, 0=no)
        oldpeak   – ST depression induced by exercise
        slope     – slope of peak exercise ST segment (0-2)
        ca        – number of major vessels colored by fluoroscopy (0-3)
        thal      – thalassemia type (1=normal, 2=fixed defect, 3=reversible defect)
        target    – 0=no disease, 1=disease
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[✓] Loaded dataset from {path} — shape: {df.shape}")
        return df

    print("[!] No dataset file found. Generating synthetic Heart Disease data …")
    np.random.seed(42)
    n = 1025

    age      = np.random.randint(29, 77, n)
    sex      = np.random.choice([0, 1], n, p=[0.32, 0.68])
    cp       = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.random.normal(131, 17, n).clip(94, 200).astype(int)
    chol     = np.random.normal(246, 52, n).clip(126, 564).astype(int)
    fbs      = np.random.choice([0, 1], n, p=[0.85, 0.15])
    restecg  = np.random.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02])
    thalach  = np.random.normal(149, 22, n).clip(71, 202).astype(int)
    exang    = np.random.choice([0, 1], n, p=[0.68, 0.32])
    oldpeak  = np.round(np.random.exponential(1.0, n).clip(0, 6.2), 1)
    slope    = np.random.choice([0, 1, 2], n, p=[0.21, 0.46, 0.33])
    ca       = np.random.choice([0, 1, 2, 3, 4], n, p=[0.58, 0.22, 0.13, 0.06, 0.01])
    thal     = np.random.choice([1, 2, 3], n, p=[0.18, 0.39, 0.43])

    # Synthetic target based on risk factors
    risk = (
        0.03 * (age - 29) +
        0.5  * sex +
        0.4  * (cp == 0) +
        0.003 * (trestbps - 94) +
        0.002 * (chol - 126) +
        0.3  * fbs +
        0.4  * exang +
        0.3  * oldpeak +
        0.5  * (ca > 0) +
        0.5  * (thal == 3) -
        0.01 * (thalach - 71)
    )
    prob   = 1 / (1 + np.exp(-risk + 2))
    target = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope,
        "ca": ca, "thal": thal, "target": target
    })
    print(f"[✓] Synthetic dataset generated — shape: {df.shape}")
    print(f"    Target distribution: {df['target'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def perform_eda(df: pd.DataFrame, save_dir: str = "static") -> None:
    """Generate and save EDA visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Heart Disease — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

    # 1. Target distribution
    counts = df["target"].value_counts()
    axes[0, 0].bar(["No Disease", "Disease"], counts.values,
                   color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
    axes[0, 0].set_title("Target Distribution")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # 2. Age distribution by target
    df[df["target"] == 0]["age"].hist(ax=axes[0, 1], bins=20, alpha=0.7, color="#2ecc71", label="No Disease")
    df[df["target"] == 1]["age"].hist(ax=axes[0, 1], bins=20, alpha=0.7, color="#e74c3c", label="Disease")
    axes[0, 1].set_title("Age Distribution by Target")
    axes[0, 1].set_xlabel("Age"); axes[0, 1].legend()

    # 3. Chest pain type vs target
    cp_labels = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
    cp_counts.plot(kind="bar", ax=axes[0, 2], color=["#2ecc71", "#e74c3c"],
                   edgecolor="white", linewidth=0.5)
    axes[0, 2].set_title("Chest Pain Type vs Target")
    axes[0, 2].set_xticklabels(cp_labels[:len(cp_counts)], rotation=20, ha="right")
    axes[0, 2].legend(["No Disease", "Disease"])

    # 4. Correlation heatmap
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=axes[1, 0], mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 6})
    axes[1, 0].set_title("Feature Correlation Heatmap")

    # 5. Max heart rate vs age
    scatter_colors = ["#2ecc71" if t == 0 else "#e74c3c" for t in df["target"]]
    axes[1, 1].scatter(df["age"], df["thalach"], c=scatter_colors, alpha=0.5, s=20)
    axes[1, 1].set_title("Max Heart Rate vs Age")
    axes[1, 1].set_xlabel("Age"); axes[1, 1].set_ylabel("Max Heart Rate")

    # 6. Cholesterol distribution
    axes[1, 2].boxplot(
        [df[df["target"] == 0]["chol"], df[df["target"] == 1]["chol"]],
        labels=["No Disease", "Disease"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red", linewidth=2)
    )
    axes[1, 2].set_title("Cholesterol by Target")
    axes[1, 2].set_ylabel("Cholesterol (mg/dl)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] EDA plots saved → {save_dir}/eda_plots.png")


# ─────────────────────────────────────────────
# 3. PRE-PROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """Clean, split, and scale the dataset."""
    # Drop duplicates & nulls
    df = df.drop_duplicates().dropna()
    print(f"[✓] After cleaning — shape: {df.shape}")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[✓] Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train.values, y_test.values, scaler, X.columns.tolist()


# ─────────────────────────────────────────────
# 4. MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_models() -> dict:
    """Return dictionary of classification models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Naive Bayes":         GaussianNB(),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(use_label_encoder=False,
                                          eval_metric="logloss", random_state=42)
        print("[✓] XGBoost detected and included.")
    except ImportError:
        print("[!] XGBoost not installed — skipping.")
    return models


# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train all models, compute metrics, return summary DataFrame."""
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        cv_s = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        results.append({
            "Model":       name,
            "Accuracy":    round(acc * 100, 2),
            "F1-Score":    round(f1 * 100, 2),
            "AUC-ROC":     round(auc * 100, 2) if auc else "N/A",
            "CV Mean (%)": round(cv_s.mean() * 100, 2),
            "CV Std (%)":  round(cv_s.std() * 100, 2),
        })
        print(f"  [{name:22s}] Acc={acc*100:.1f}%  F1={f1*100:.1f}%  AUC={auc*100:.1f}%" if auc else
              f"  [{name:22s}] Acc={acc*100:.1f}%  F1={f1*100:.1f}%")

    return pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, save_dir: str = "static") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x      = np.arange(len(results_df))
    width  = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, (metric, color) in enumerate(zip(["Accuracy", "F1-Score", "AUC-ROC"], colors)):
        vals = pd.to_numeric(results_df[metric], errors="coerce")
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color,
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(50, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Algorithm Comparison — Heart Disease Prediction", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Model comparison chart saved → {save_dir}/model_comparison.png")


def plot_roc_curves(models, X_test, y_test, save_dir: str = "static") -> None:
    plt.figure(figsize=(9, 7))
    palette = plt.cm.tab10.colors

    for i, (name, model) in enumerate(models.items()):
        if not hasattr(model, "predict_proba"):
            continue
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=palette[i % 10], lw=2, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Classifiers", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] ROC curves saved → {save_dir}/roc_curves.png")


def plot_confusion_matrices(models, X_test, y_test, save_dir: str = "static") -> None:
    n_models = len(models)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        axes[i].set_title(name, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrices saved → {save_dir}/confusion_matrices.png")


def plot_feature_importance(models, feature_names, save_dir: str = "static") -> None:
    tree_models = {k: v for k, v in models.items()
                   if hasattr(v, "feature_importances_")}
    if not tree_models:
        return

    fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        imp.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
        ax.set_title(f"{name}\nFeature Importance", fontweight="bold")
        ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Feature importance saved → {save_dir}/feature_importance.png")


# ─────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING (Best Model)
# ─────────────────────────────────────────────

def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs  = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
    gs.fit(X_train, y_train)
    print(f"[✓] Best RF params: {gs.best_params_} | CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─────────────────────────────────────────────
# 8. PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_patient(model, scaler, patient_data: dict) -> dict:
    """
    Predict heart disease for a single patient.

    patient_data keys: age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal
    """
    feature_order = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    X = pd.DataFrame([patient_data])[feature_order]
    X_scaled = scaler.transform(X)

    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None

    result = {
        "prediction": "Heart Disease Detected" if pred == 1 else "No Heart Disease",
        "risk_level": "High" if pred == 1 else "Low",
    }
    if proba is not None:
        result["confidence"] = f"{max(proba) * 100:.1f}%"
        result["disease_probability"] = f"{proba[1] * 100:.1f}%"

    return result


# ─────────────────────────────────────────────
# 9. SAVE & LOAD MODEL
# ─────────────────────────────────────────────

def save_model(model, scaler, path: str = "models/best_model.joblib") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"[✓] Model saved → {path}")


def load_model(path: str = "models/best_model.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"]


# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  HEART DISEASE PREDICTION — CLASSIFICATION PIPELINE")
    print("="*60 + "\n")

    # Step 1: Load data
    df = load_heart_dataset()

    # Step 2: EDA
    print("\n[→] Running EDA …")
    perform_eda(df)

    # Step 3: Pre-process
    print("\n[→] Pre-processing …")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # Step 4: Train & evaluate
    print("\n[→] Training models …")
    models = get_models()
    results_df = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\n--- RESULTS SUMMARY ---")
    print(results_df.to_string(index=False))

    # Step 5: Visualise
    print("\n[→] Generating visualisations …")
    plot_model_comparison(results_df)
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models, feature_names)

    # Step 6: Hyperparameter tuning (Random Forest)
    print("\n[→] Tuning best model (Random Forest) …")
    best_rf = tune_random_forest(X_train, y_train)

    # Step 7: Save best model
    save_model(best_rf, scaler)

    # Step 8: Demo prediction
    print("\n[→] Demo patient prediction …")
    patient = {
        "age": 55, "sex": 1, "cp": 0, "trestbps": 145,
        "chol": 230, "fbs": 0, "restecg": 0, "thalach": 140,
        "exang": 1, "oldpeak": 2.3, "slope": 0, "ca": 1, "thal": 3
    }
    result = predict_patient(best_rf, scaler, patient)
    print(f"\n  Patient data: {patient}")
    print(f"  Prediction  : {result}")

    print("\n[✓] Pipeline complete! All outputs saved to ./static/\n")
    return models, results_df, scaler


if __name__ == "__main__":
    main()
