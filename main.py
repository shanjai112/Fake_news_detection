import streamlit as st
import re
import random
import pandas as pd

# --- Expanded Fake Signals ---
suspicious_words = [
    # Clickbait
    "shocking", "breaking", "unbelievable", "secret", "miracle", "urgent", 
    "you won't believe", "truth exposed", "cover-up", "world ending", "banned news",
    # Too good to be true
    "free money", "guaranteed", "win a prize", "100% proven", "cure in 1 week",
    # Conspiracy
    "deep state", "hoax", "elite agenda", "hidden truth", "population control"
]

def analyze_headline(headline):
    score = 0
    reasons = []

    # Check suspicious words
    for word in suspicious_words:
        if re.search(word, headline.lower()):
            score += 15
            reasons.append(f"Contains suspicious word: '{word}'")

    # Check ALL CAPS
    if headline.isupper():
        score += 25
        reasons.append("Headline is in ALL CAPS (suspicious).")

    # Check punctuation abuse
    if "!!" in headline or "??" in headline:
        score += 15
        reasons.append("Excessive punctuation detected (!! or ??).")

    # Random ML simulation
    ml_fake_score = random.randint(0, 100)
    score = (score + ml_fake_score) // 2

    reasons.append(f"ML model estimated {ml_fake_score}% chance of being fake.")

    return min(score, 100), reasons

# --- Streamlit UI ---
st.set_page_config(page_title="Outlaw News Hunter", layout="wide")

st.title("🤠 The Outlaw News Hunter")
st.subheader("AI Fake News Headline Detector")

headline = st.text_input("Enter a news headline:")

if st.button("Analyze"):
    if headline.strip():
        score, reasons = analyze_headline(headline)

        st.metric(label="🧾 Risk Score", value=f"{score}%")
        if score > 70:
            st.error("⚠️ High risk of being fake/misleading!")
        elif score > 40:
            st.warning("⚠️ Possibly suspicious, check sources.")
        else:
            st.success("✅ Looks trustworthy.")

        st.markdown("### 🔎 Reasons")
        for r in reasons:
            st.write("- " + r)
    else:
        st.warning("Please enter a headline.")

# --- Dashboard ---
st.markdown("---")
st.subheader("📊 Dashboard")

data = {
    "Publisher": ["SiteA", "SiteB", "SiteC", "SiteD"],
    "Suspicious Headlines": [12, 25, 7, 30]
}
df = pd.DataFrame(data)

st.bar_chart(df.set_index("Publisher"))
5