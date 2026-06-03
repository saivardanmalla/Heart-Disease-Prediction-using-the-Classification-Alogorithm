# ❤️ Heart Disease Prediction — Classification Algorithms

A complete, production-ready machine-learning repository that predicts heart disease using **7+ classification algorithms**, with full EDA, visualisations, hyperparameter tuning, a web API, and an interactive Jupyter notebook.

---

## 📁 Repository Structure

```
heart-disease-prediction/
├── src/
│   └── heart_disease_classifier.py   ← Core ML pipeline (all algorithms)
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb ← Step-by-step Jupyter walkthrough
├── models/
│   └── best_model.joblib              ← Saved tuned model (auto-generated)
├── static/                            ← Auto-generated charts & plots
│   ├── eda_plots.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
├── app.py                             ← Flask web API + UI
├── requirements.txt
└── README.md
```

---

## 🩺 Dataset — UCI Cleveland Heart Disease

| Feature    | Description                                    | Type        |
|------------|------------------------------------------------|-------------|
| `age`      | Age in years                                   | Numerical   |
| `sex`      | 1 = Male, 0 = Female                           | Categorical |
| `cp`       | Chest pain type (0–3)                          | Categorical |
| `trestbps` | Resting blood pressure (mmHg)                  | Numerical   |
| `chol`     | Serum cholesterol (mg/dl)                      | Numerical   |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1/0)          | Categorical |
| `restecg`  | Resting ECG results (0–2)                      | Categorical |
| `thalach`  | Maximum heart rate achieved                    | Numerical   |
| `exang`    | Exercise-induced angina (1/0)                  | Categorical |
| `oldpeak`  | ST depression induced by exercise              | Numerical   |
| `slope`    | Slope of peak exercise ST segment (0–2)        | Categorical |
| `ca`       | Number of major vessels (0–4)                  | Numerical   |
| `thal`     | Thalassemia type (1=normal, 2=fixed, 3=rev.)   | Categorical |
| `target`   | **0 = No disease, 1 = Disease**                | **Target**  |

**Download the real dataset** from:  
`https://archive.ics.uci.edu/ml/datasets/Heart+Disease`  
Save as `data/heart.csv` or pass `path='data/heart.csv'` to `load_heart_dataset()`.

---

## 🤖 Algorithms Implemented

| Algorithm             | Key Hyperparameters Tuned             |
|-----------------------|---------------------------------------|
| Logistic Regression   | `max_iter=1000`                       |
| K-Nearest Neighbors   | `n_neighbors=5`                       |
| Decision Tree         | `max_depth=5`                         |
| **Random Forest** ⭐  | `n_estimators`, `max_depth` (GridCV)  |
| Support Vector Machine| `kernel=rbf`, `probability=True`      |
| Gradient Boosting     | `n_estimators=100`                    |
| Naive Bayes           | Gaussian                              |
| XGBoost *(optional)*  | `eval_metric=logloss`                 |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline
```bash
cd heart-disease-prediction
python src/heart_disease_classifier.py
```
This will:
- Generate/load the dataset
- Run EDA and save plots to `static/`
- Train and evaluate all 7+ classifiers
- Tune the best model (Random Forest) with GridSearchCV
- Save the model to `models/best_model.joblib`
- Run a demo patient prediction

### 3. Launch the web app
```bash
python app.py
# Open http://localhost:5000
```

### 4. Open the Jupyter Notebook
```bash
jupyter notebook notebooks/Heart_Disease_Prediction.ipynb
```

---

## 📊 API Usage

**POST `/predict`** — JSON body:
```json
{
  "age": 55, "sex": 1, "cp": 0, "trestbps": 145,
  "chol": 230, "fbs": 0, "restecg": 0, "thalach": 140,
  "exang": 1, "oldpeak": 2.3, "slope": 0, "ca": 1, "thal": 3
}
```

**Response:**
```json
{
  "prediction": "Heart Disease Detected",
  "risk_level": "High",
  "confidence": "87.4%",
  "disease_probability": "79.2%",
  "model": "Random Forest (Tuned)"
}
```

---

## 📈 Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **F1-Score** — Harmonic mean of precision & recall
- **AUC-ROC** — Area under the ROC curve
- **Cross-Validation** — 5-fold stratified CV mean ± std
- **Confusion Matrix** — TP, TN, FP, FN breakdown

---

## 🔬 Pipeline Steps

```
Raw Data → EDA → Pre-processing → Train/Test Split (80/20)
    → StandardScaler → Train 7+ Models → Evaluate Metrics
    → Hyperparameter Tuning (GridSearchCV) → Save Best Model
    → Flask Web API → Real-time Prediction
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not a medical device** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

MIT License — free for personal and academic use.
