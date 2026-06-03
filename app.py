"""
Heart Disease Prediction — Flask Web API
Run: python app.py
Endpoints:
    GET  /           — Web UI
    POST /predict    — JSON prediction
    GET  /results    — Model results page
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

# ── Inline HTML template ──────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heart Disease Prediction</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  header { background: linear-gradient(135deg, #e11d48, #7c3aed); padding: 2rem; text-align: center; }
  header h1 { font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
  header p  { opacity: 0.85; margin-top: 0.4rem; }
  .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  .card { background: #1e293b; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
  label { display: block; margin-bottom: 0.3rem; font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
  input, select {
    width: 100%; padding: 0.6rem 0.9rem; background: #0f172a;
    border: 1px solid #475569; border-radius: 8px; color: #e2e8f0;
    font-size: 0.95rem; transition: border-color 0.2s;
  }
  input:focus, select:focus { outline: none; border-color: #e11d48; }
  button { width: 100%; padding: 0.9rem; background: linear-gradient(135deg, #e11d48, #7c3aed);
    border: none; border-radius: 10px; color: white; font-size: 1rem;
    font-weight: 700; cursor: pointer; margin-top: 1rem; transition: opacity 0.2s; }
  button:hover { opacity: 0.9; }
  #result { display: none; border-radius: 16px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
  .high-risk { background: rgba(220, 38, 38, 0.15); border: 2px solid #dc2626; }
  .low-risk  { background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; }
  .risk-icon { font-size: 3rem; }
  .risk-label { font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0; }
  .risk-meta  { opacity: 0.8; font-size: 0.9rem; }
  .info-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; justify-content: center; }
  .info-pill { background: #0f172a; border-radius: 20px; padding: 0.35rem 1rem; font-size: 0.85rem; border: 1px solid #334155; }
  h2 { font-size: 1.2rem; font-weight: 700; margin-bottom: 1.2rem; color: #f1f5f9; }
  .legend { font-size: 0.78rem; color: #64748b; margin-top: 0.4rem; }
</style>
</head>
<body>
<header>
  <h1>❤️ Heart Disease Prediction</h1>
  <p>Multi-Algorithm Classification · Clinical Risk Assessment</p>
</header>

<div class="container">
  <div class="card">
    <h2>Patient Information</h2>
    <div class="grid">
      <div>
        <label>Age (years)</label>
        <input type="number" id="age" value="55" min="20" max="90">
      </div>
      <div>
        <label>Sex</label>
        <select id="sex">
          <option value="1">Male</option>
          <option value="0">Female</option>
        </select>
      </div>
      <div>
        <label>Chest Pain Type</label>
        <select id="cp">
          <option value="0">Typical Angina</option>
          <option value="1">Atypical Angina</option>
          <option value="2">Non-anginal Pain</option>
          <option value="3">Asymptomatic</option>
        </select>
      </div>
      <div>
        <label>Resting BP (mmHg)</label>
        <input type="number" id="trestbps" value="130" min="80" max="220">
      </div>
      <div>
        <label>Cholesterol (mg/dl)</label>
        <input type="number" id="chol" value="240" min="100" max="600">
      </div>
      <div>
        <label>Fasting Blood Sugar &gt;120</label>
        <select id="fbs">
          <option value="0">No (&lt;= 120 mg/dl)</option>
          <option value="1">Yes (&gt; 120 mg/dl)</option>
        </select>
      </div>
      <div>
        <label>Resting ECG</label>
        <select id="restecg">
          <option value="0">Normal</option>
          <option value="1">ST-T Abnormality</option>
          <option value="2">LV Hypertrophy</option>
        </select>
      </div>
      <div>
        <label>Max Heart Rate</label>
        <input type="number" id="thalach" value="150" min="60" max="220">
      </div>
      <div>
        <label>Exercise-induced Angina</label>
        <select id="exang">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div>
        <label>ST Depression (Oldpeak)</label>
        <input type="number" id="oldpeak" value="1.0" step="0.1" min="0" max="7">
      </div>
      <div>
        <label>Slope of Peak ST</label>
        <select id="slope">
          <option value="0">Upsloping</option>
          <option value="1">Flat</option>
          <option value="2">Downsloping</option>
        </select>
      </div>
      <div>
        <label>Major Vessels (0-4)</label>
        <input type="number" id="ca" value="0" min="0" max="4">
      </div>
      <div>
        <label>Thalassemia</label>
        <select id="thal">
          <option value="1">Normal</option>
          <option value="2">Fixed Defect</option>
          <option value="3">Reversible Defect</option>
        </select>
      </div>
    </div>
    <button onclick="predict()">🔍 Predict Heart Disease Risk</button>
  </div>

  <div id="result" class="card"></div>
</div>

<script>
async function predict() {
  const fields = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"];
  const data = {};
  fields.forEach(f => { data[f] = parseFloat(document.getElementById(f).value); });

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  const json = await res.json();

  const box = document.getElementById("result");
  const isHigh = json.risk_level === "High";
  box.className = "card " + (isHigh ? "high-risk" : "low-risk");
  box.style.display = "block";
  box.innerHTML = `
    <div class="risk-icon">${isHigh ? "⚠️" : "✅"}</div>
    <div class="risk-label" style="color:${isHigh ? "#f87171" : "#34d399"}">${json.prediction}</div>
    <div class="risk-meta">Risk Level: <strong>${json.risk_level}</strong></div>
    <div class="info-row">
      <span class="info-pill">🎯 Confidence: ${json.confidence}</span>
      <span class="info-pill">💊 Disease Probability: ${json.disease_probability}</span>
      <span class="info-pill">🤖 Model: ${json.model}</span>
    </div>
    <p style="margin-top:1rem;font-size:0.8rem;color:#64748b;">
      This is a machine-learning estimate only. Always consult a licensed medical professional.
    </p>
  `;
  box.scrollIntoView({ behavior: "smooth" });
}
</script>
</body>
</html>
"""

# ── Load/train model at startup ────────────────────────────────────────────────
model  = None
scaler = None

def _init():
    global model, scaler
    try:
        from heart_disease_classifier import load_model
        model, scaler = load_model()
        print("[✓] Loaded saved model.")
    except Exception:
        print("[!] No saved model — training now …")
        from heart_disease_classifier import (
            load_heart_dataset, preprocess, get_models,
            train_and_evaluate, tune_random_forest, save_model
        )
        df = load_heart_dataset()
        X_train, X_test, y_train, y_test, sc, _ = preprocess(df)
        models = get_models()
        train_and_evaluate(models, X_train, X_test, y_train, y_test)
        model  = tune_random_forest(X_train, y_train)
        model.fit(X_train, y_train)
        scaler = sc
        save_model(model, scaler)

_init()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    feature_order = ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"]
    try:
        import pandas as pd
        X = pd.DataFrame([data])[feature_order]
        X_sc  = scaler.transform(X)
        pred  = model.predict(X_sc)[0]
        proba = model.predict_proba(X_sc)[0]

        return jsonify({
            "prediction":          "Heart Disease Detected" if pred == 1 else "No Heart Disease",
            "risk_level":          "High" if pred == 1 else "Low",
            "confidence":          f"{max(proba)*100:.1f}%",
            "disease_probability": f"{proba[1]*100:.1f}%",
            "model":               "Random Forest (Tuned)",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
