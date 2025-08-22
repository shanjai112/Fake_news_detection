from flask import Flask, render_template, request
import re
import random

app = Flask(__name__)

# --- Fake signals ---
weak_signals = ["breaking", "urgent", "shocking"]
strong_signals = [
    "miracle", "secret", "you won't believe", "free money",
    "guaranteed", "hoax", "deep state", "banned news"
]

def analyze_headline(headline):
    score = 0
    reasons = []

    # Weak signals (low weight, used in real news too)
    for word in weak_signals:
        if word in headline.lower():
            score += 5
            reasons.append(f"Weak signal detected: '{word}' (common in real news too)")

    # Strong signals (higher weight, usually clickbait/fake)
    for word in strong_signals:
        if word in headline.lower():
            score += 20
            reasons.append(f"Strong suspicious phrase detected: '{word}'")

    # Check ALL CAPS
    if headline.isupper():
        score += 25
        reasons.append("Headline is in ALL CAPS (suspicious).")

    # Check punctuation abuse
    if "!!" in headline or "??" in headline:
        score += 15
        reasons.append("Excessive punctuation detected (!! or ??).")

    # Simulated ML classifier
    ml_fake_score = random.randint(0, 100)
    score = (score + ml_fake_score) // 2
    reasons.append(f"ML model estimated {ml_fake_score}% chance of being fake.")

    return min(score, 100), reasons

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    reasons = []
    headline = ""

    if request.method == "POST":
        headline = request.form.get("headline", "")
        if headline.strip():
            score, reasons = analyze_headline(headline)
            result = score

    return render_template("index.html", result=result, reasons=reasons, headline=headline)

if __name__ == "__main__":
    app.run(debug=True)
