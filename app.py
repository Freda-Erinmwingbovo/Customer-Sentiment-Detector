# ============================================================
# app.py — Customer Sentiment & Emotion Detector (App #2) — FINAL BEST VERSION
# Part of AI-Powered Customer Support Automation Suite
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import re
from datetime import datetime, timezone, timedelta
import streamlit.components.v1 as components

# ------------------------- NIGERIA TIME -------------------------
WAT = timezone(timedelta(hours=1))

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="Sentiment Detector",
    page_icon="❤️",
    layout="wide"
)

# ------------------------- MODEL LOADING -------------------------
@st.cache_resource
def load_sentiment_model():
    return joblib.load("sentiment_classifier_PROD.pkl")

model = load_sentiment_model()

# ------------------------- CLEAN_TEXT FUNCTION -------------------------
def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    stop_words = {
        "a", "an", "the", "and", "or", "is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "of",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
        "her", "its", "our", "their", "from", "as", "by", "be", "been", "am", "will", "can", "do", "does",
        "did", "have", "has", "had", "not", "but", "if", "then", "so", "no", "yes"
    }
    return " ".join(w for w in t.split() if w not in stop_words)

# ------------------------- LOGGING SETUP -------------------------
LOG_FILE = "data/sentiment_log.csv"
os.makedirs("data", exist_ok=True)

def safe_read_log():
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return pd.DataFrame(columns=["timestamp", "message_snippet", "sentiment", "confidence", "action"])
    try:
        return pd.read_csv(LOG_FILE)
    except:
        return pd.DataFrame(columns=["timestamp", "message_snippet", "sentiment", "confidence", "action"])

def save_and_log(message, sentiment, confidence, action):
    now = datetime.now(WAT)
    log_df = safe_read_log()
    new_row = pd.DataFrame([{
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "message_snippet": message[:100],
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "action": action
    }])
    log_df = pd.concat([log_df, new_row], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)

# ------------------------- PREDICTION FUNCTION -------------------------
def predict_sentiment(message, threshold):
    if not message.strip():
        return None

    cleaned = clean_text(message)
    decision_scores = model.decision_function([cleaned])
    exp_scores = np.exp(decision_scores - decision_scores.max())
    proba = exp_scores / exp_scores.sum()
    confidence = float(proba.max())
    sentiment = model.classes_[proba.argmax()]

    auto = confidence >= threshold
    if auto and sentiment == "negative":
        action = "NEGATIVE ALERT → Escalate immediately"
    elif auto and sentiment == "positive":
        action = "POSITIVE → Celebrate / Follow up"
    elif auto:
        action = "NEUTRAL → Standard handling"
    else:
        action = "LOW CONFIDENCE → Human review recommended"

    return sentiment, confidence, auto, action

# ------------------------- TABS -------------------------
tab1, tab2 = st.tabs(["Detector", "History"])

with tab1:
    st.title("❤️ Customer Sentiment & Emotion Detector")
    st.markdown("*Detect anger early • Celebrate happiness • Stay safe on neutral*")

    col1, col2 = st.columns([2, 1])
    with col1:
        message = st.text_area(
            "Customer Message",
            placeholder="Paste ticket, email, chat or tweet here...",
            height=220,
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("### Settings")
        threshold = st.slider(
            "Confidence Threshold",
            0.50, 0.95, 0.70, 0.05,
            help="Higher = safer (fewer auto-detections)\nLower = more sensitive"
        )

    if st.button("ANALYZE SENTIMENT", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a customer message")
        else:
            with st.spinner("Analyzing emotion..."):
                result = predict_sentiment(message, threshold)
                if result:
                    sentiment, conf, auto, action = result
                    save_and_log(message, sentiment, conf, action)

                    st.success("Analysis Complete!")
                    color = "red" if sentiment == "negative" else "orange" if sentiment == "neutral" else "green"
                    st.markdown(f"## <span style='color:{color}'>{sentiment.upper()} ({conf:.1%} confidence)</span>", unsafe_allow_html=True)
                    st.markdown(f"### {action}")

                    if auto and sentiment == "negative":
                        st.error("NEGATIVE SENTIMENT DETECTED — Consider immediate escalation!")
                        components.html("<script>alert('Negative sentiment detected!');</script>", height=0)
                    elif auto and sentiment == "positive":
                        st.success("POSITIVE SENTIMENT — Great customer experience!")
                        st.balloons()

# ------------------------- HISTORY TAB — FULL PERSISTENT LOG -------------------------
with tab2:
    st.header("Detection History")
    log_df = safe_read_log()
    if len(log_df) > 0:
        display_df = log_df.copy()
        display_df["Time"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%H:%M")
        display_df["Sentiment"] = display_df["sentiment"].str.upper()
        display_df["Confidence"] = display_df["confidence"].apply(lambda x: f"{float(x):.1%}")
        display_df["Action"] = display_df["action"].str.split(" → ").str[0]
        display_df["Message"] = display_df["message_snippet"].apply(lambda x: x[:60] + "..." if len(x) > 60 else x)
        
        # Reorder columns and sort by timestamp (now guaranteed to exist)
        display_df = display_df[["Time", "Message", "Sentiment", "Confidence", "Action"]]
        display_df = display_df.sort_values(by="timestamp", ascending=False)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        with st.expander("Admin Tools (protected)", expanded=False):
            pwd = st.text_input("Admin password", type="password", key="admin_pwd")
            if pwd == st.secrets.get("ADMIN_PASSWORD", "___NEVER___"):
                st.success("Authorized")
                csv = log_df.to_csv(index=False).encode()
                st.download_button(
                    "📥 Download full log (CSV)",
                    csv,
                    f"sentiment_log_{datetime.now(WAT).strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            elif pwd and pwd != st.secrets.get("ADMIN_PASSWORD", "___NEVER___"):
                st.error("Wrong password")
    else:
        st.info("No analyses yet → go to **Detector** tab!")

# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.image("https://em-content.zobj.net/source/skype/289/heart_2764.png", width=100)
    st.title("Sentiment Detector")
    st.caption("79.3% Accuracy • 93.7% on high-confidence")
    total_logs = len(safe_read_log())
    st.metric("Total Analyzed (all time)", total_logs)
    st.divider()
    st.info("Flags negative sentiment early → prevents churn\nCelebrates positive → boosts morale")

# ------------------------- FOOTER -------------------------
st.markdown(
    """
    <hr style="border-top: 1px solid #444;">
    <p style="text-align: center; color: #aaa; font-size: 15px;">
    Built solo by <strong>Freda Erinmwingbovo</strong> • Abuja, Nigeria • December 2025
    </p>
    """,
    unsafe_allow_html=True
)
