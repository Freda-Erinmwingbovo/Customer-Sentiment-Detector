# ============================================================
# app.py — Customer Sentiment & Emotion Detector (Final Production Version)
# Support-Specific • Negation-Aware • Production Ready
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
    return joblib.load("sentiment_classifier_support_final.pkl")

model = load_sentiment_model()

# ------------------------- FINAL SUPPORT-SPECIFIC CLEAN_TEXT -------------------------
def clean_text_support(t):
    if pd.isna(t):
        return ""
    t = str(t).lower()
    
    # Expand contractions
    contractions = {
        "can't": "can not", "cannot": "can not",
        "won't": "will not",
        "shan't": "shall not",
        "don't": "do not", "dont": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "mightn't": "might not",
        "mustn't": "must not"
    }
    for contr, exp in contractions.items():
        t = t.replace(contr, exp)
    
    t = t.replace("i've", "i have")
    t = t.replace("i'm", "i am")
    t = t.replace("i'll", "i will")
    t = t.replace("i'd", "i would")
    
    # Basic cleaning
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    # Negation tagging
    words = t.split()
    negation = False
    negation_triggers = {"not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "hardly", "barely", "scarcely", "rarely", "seldom"}
    sentence_enders = {".", "!", "?", ":", ";"}
    
    strong_negative_words = {
        "bad", "worst", "terrible", "awful", "hate", "boring", "waste", "disappointing", "poor", "problem",
        "horrible", "stupid", "dull", "rubbish", "crap", "trash", "fail", "lame", "weak", "mess", "sucks",
        "annoying", "ridiculous", "pointless", "crappy", "garbage", "ugly", "slow", "broken", "error", "bug",
        "crash", "disaster", "nightmare", "pain", "hurt", "sad", "angry", "mad", "frustrated", "issue", "issues",
        "complain", "delay", "cancel", "rude", "lost", "charge", "overcharge", "debit", "overbilling"
    }
    
    tagged_words = []
    for word in words:
        if word in negation_triggers:
            negation = True
        elif word in sentence_enders:
            negation = False
        
        if word in strong_negative_words:
            word = f"NEG_{word}"
        
        tagged_words.append(word)
    
    t = " ".join(tagged_words)
    
    stop_words = {
        "a","an","the","and","or","is","are","was","were","in","on","at","to","for","with","of","this","that","these","those",
        "i","you","he","she","it","we","they","my","your","his","her","its","our","their","from","as","by","be","been","am",
        "will","can","do","does","did","have","has","had","not","but","if","then","so","no","yes"
    }
    return " ".join(w for w in t.split() if w not in stop_words)

# ------------------------- PREDICTION FUNCTION (Fixed for SVM) -------------------------
def predict_sentiment(message, threshold):
    if not message.strip():
        return None

    cleaned = clean_text_support(message)
    
    sentiment = model.predict([cleaned])[0]
    
    # Confidence — handle Linear SVM
    decision = model.decision_function([cleaned])
    exp = np.exp(decision - decision.max())
    proba = exp / exp.sum()
    confidence = float(proba.max())
    
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

# ------------------------- REST OF THE APP (same as before) -------------------------
# (keep your current tabs, history, sidebar, footer — only the prediction function and model load changed)

# ... (rest of your app code — tabs, history, sidebar, footer)
