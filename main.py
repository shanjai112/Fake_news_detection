from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# ===============================
# Load dataset safely
# ===============================
DATASET_PATH = "data.csv"

def load_dataset(path):
    encodings_to_try = ["utf-8", "ISO-8859-1", "cp1252"]
    for enc in encodings_to_try:
        try:
            print(f"Trying to load with encoding: {enc}")
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            print(f"⚠️ Failed with {enc}, trying next...")
    raise ValueError("Could not read CSV file with common encodings. Please check the file.")

df = load_dataset(DATASET_PATH)

# Keep only needed columns & clean
assert "Statement" in df.columns and "Label" in df.columns, "CSV must have 'Statement' and 'Label' columns"
df = df[["Statement", "Label"]].dropna()
df["Label"] = df["Label"].astype(str).str.strip().str.capitalize()  # e.g., "real" -> "Real", "fake " -> "Fake"

print("✅ Dataset loaded")
print(df["Label"].value_counts())

# ===============================
# Train/test split (stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df["Statement"].values, df["Label"].values,
    test_size=0.2, random_state=42, stratify=df["Label"].values
)

# ===============================
# Upsample ONLY the training set
# ===============================
train_df = pd.DataFrame({"Statement": X_train, "Label": y_train})
vc = train_df["Label"].value_counts()

if len(vc) == 2:
    majority_label = vc.idxmax()
    minority_label = vc.idxmin()

    majority_df = train_df[train_df["Label"] == majority_label]
    minority_df = train_df[train_df["Label"] == minority_label]

    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=42
    )
    train_bal = pd.concat([majority_df, minority_upsampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_train_bal = train_bal["Statement"].values
    y_train_bal = train_bal["Label"].values
    print("📊 After upsampling (train only):")
    print(pd.Series(y_train_bal).value_counts())
else:
    X_train_bal, y_train_bal = X_train, y_train
    print("⚠️ Only one class in training split; skipping upsampling.")

# ===============================
# Pipeline with TF-IDF + Calibrated LinearSVC (supports predict_proba)
# ===============================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5
    )),
    ("clf", CalibratedClassifierCV(
        estimator=LinearSVC(),
        method="sigmoid",   # Platt scaling
        cv=3
    ))
])

# Train
model.fit(X_train_bal, y_train_bal)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n📊 Classification Report (true test set):")
print(classification_report(y_test, y_pred, digits=4))
print(f"✅ Accuracy: {acc:.4f}")

# ===============================
# Flask app
# ===============================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fake News Detector + Truth Retrieval</title>
  <style>
    /* --- Animated Gradient Background --- */
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(270deg, #1f1c2c, #928dab, #3a1c71, #d76d77, #ffaf7b);
      background-size: 1000% 1000%;
      animation: gradientShift 20s ease infinite;
      color: #f5f5f5;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px;
      min-height: 100vh;
      overflow-x: hidden;
    }

    @keyframes gradientShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    /* --- Title Typing Effect --- */
    h1 {
      font-size: 2.8rem;
      font-weight: bold;
      white-space: nowrap;
      overflow: hidden;
      border-right: 3px solid #fff;
      width: 0;
      animation: typing 3s steps(30, end) forwards, blink 0.8s infinite;
      background: linear-gradient(90deg, #ff8a00, #e52e71);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 25px;
    }

    @keyframes typing {
      from { width: 0; }
      to { width: 100%; }
    }

    @keyframes blink {
      50% { border-color: transparent; }
    }

    /* --- Floating Particles --- */
    .particles span {
      position: absolute;
      display: block;
      width: 8px;
      height: 8px;
      background: rgba(255,255,255,0.6);
      border-radius: 50%;
      animation: float 10s linear infinite;
    }
    .particles span:nth-child(1){ top:10%; left:20%; animation-duration: 6s; }
    .particles span:nth-child(2){ top:40%; left:80%; animation-duration: 8s; }
    .particles span:nth-child(3){ top:70%; left:50%; animation-duration: 12s; }
    .particles span:nth-child(4){ top:85%; left:15%; animation-duration: 9s; }
    .particles span:nth-child(5){ top:30%; left:60%; animation-duration: 7s; }

    @keyframes float {
      0% { transform: translateY(0) scale(1); opacity: 0.8; }
      50% { transform: translateY(-40px) scale(1.3); opacity: 1; }
      100% { transform: translateY(0) scale(1); opacity: 0.6; }
    }

    /* --- Glassmorphism Card --- */
    .card {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      padding: 25px;
      max-width: 700px;
      width: 100%;
      backdrop-filter: blur(15px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
      animation: slideUp 1s ease;
    }

    @keyframes slideUp {
      from { transform: translateY(40px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }

    /* --- Inputs & Buttons --- */
    textarea {
      width: 660px;
      height: 120px;
      border-radius: 12px;
      border: none;
      padding: 12px;
      font-size: 1rem;
      outline: none;
      transition: all 0.3s ease;
      resize: none;
    }

    textarea:focus {
      box-shadow: 0 0 20px #ff8a00;
      transform: scale(1.02);
    }

    button {
      background: linear-gradient(90deg, #ff8a00, #e52e71);
      border: none;
      padding: 12px 25px;
      margin-top: 15px;
      border-radius: 30px;
      font-size: 1rem;
      font-weight: bold;
      color: white;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      position: relative;
      overflow: hidden;
    }

    button::after {
      content: "";
      position: absolute;
      top: 0; left: -100%;
      width: 100%; height: 100%;
      background: rgba(255,255,255,0.3);
      transition: left 0.3s ease;
    }

    button:hover::after {
      left: 100%;
    }

    button:hover {
      transform: scale(1.07);
      box-shadow: 0 0 20px #e52e71;
    }

    button:active {
      transform: scale(0.95);
    }

    /* --- Headings with Underline Effect --- */
    h2, h3 {
      margin-top: 20px;
      display: inline-block;
      position: relative;
      color: #ffd369;
    }

    h2::after, h3::after {
      content: "";
      position: absolute;
      width: 0%;
      height: 3px;
      left: 0;
      bottom: -5px;
      background: #ffd369;
      transition: width 0.4s ease;
    }

    h2:hover::after, h3:hover::after {
      width: 100%;
    }

    /* --- Fade & Slide for Prediction Box --- */
    .fade-slide {
      animation: fadeSlide 1s ease forwards;
      opacity: 0;
      transform: translateX(-30px);
    }

    @keyframes fadeSlide {
      to { opacity: 1; transform: translateX(0); }
    }
  </style>
</head>
<body>
  <!-- Floating particles background -->
  <div class="particles">
    <span></span><span></span><span></span><span></span><span></span>
  </div>

  <h1>📰 Fake News Detector</h1>
  <div class="card">
    <p><b>Model Accuracy:</b> {{ accuracy }}</p>
    <form method="post">
      <textarea name="news" required placeholder="Paste news content here..."></textarea><br>
      <button type="submit">🔍 Check</button>
    </form>

    {% if prediction %}
    <div class="fade-slide">
      <h2>Prediction: {{ prediction }} ({{ confidence }}%)</h2>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    if request.method == "POST":
        text = request.form["news"]
        proba = model.predict_proba([text])[0]
        pred_idx = int(np.argmax(proba))
        pred_label = model.classes_[pred_idx]
        prediction = str(pred_label)
        confidence = round(100.0 * float(proba[pred_idx]), 2)

    return render_template_string(
        HTML_TEMPLATE,
        accuracy=round(100.0 * acc, 2),
        prediction=prediction,
        confidence=confidence
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default 5000 for local dev
    app.run(host="0.0.0.0", port=port, debug=True)

