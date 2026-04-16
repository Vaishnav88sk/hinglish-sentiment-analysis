import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from deep_translator import GoogleTranslator
from transformers import pipeline
from helper import (
    filter, tokenize, separate_into_dict,
    transliterate_roman_to_hi, hindiStopwordsRemover,
    lemmatize_hi, postagger_hi, negation_handling_hin,
    sentiment
)

st.set_page_config(page_title="Hinglish Sentiment Analyzer", page_icon="🧠", layout="wide")

@st.cache_resource
def load_emotion():
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=1, device=-1)

emotion_model = load_emotion()

def contains_hindi(text):
    return any("\u0900" <= c <= "\u097F" for c in text)

def translate_to_english(hindi_text, original_text):
    """
    Translate to English using the Hindi text (after transliteration) for better accuracy.
    Falls back to original text if Hindi translation fails.
    """
    try:
        # First try translating the Hindi script text — this gives much better results
        # because Google Translate handles proper Hindi better than Hinglish
        result = GoogleTranslator(source="hi", target="en").translate(hindi_text)
        if result and result.strip():
            return result
    except Exception:
        pass

    try:
        # Fallback: translate original text with auto-detect
        result = GoogleTranslator(source="auto", target="en").translate(original_text)
        if result and result.strip():
            return result
    except Exception:
        pass

    return original_text

def safe_json(data):
    """Ensure all keys/values are strings for st.json display"""
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return data

# ================== UI ==================
st.title("🧠 Hinglish Sentiment Analyzer")
st.write(
    "Analyze Hindi, English, and Hinglish text using an advanced NLP pipeline with "
    "tokenization, lemmatization, POS tagging, negation handling, sentiment analysis, "
    "and AI-powered emotion detection for more accurate contextual understanding."
)

text = st.text_area("Enter Text", height=150,
                    placeholder="Yaar internet fir chala gaya\nKya bakwas traffic hai\nYe phone bahut mast hai")

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter text.")
        st.stop()

    try:
        with st.spinner("Analyzing..."):
            # ---- Step 1: Clean ----
            clean = filter(text)

            # ---- Step 2: Transliterate to Hindi ----
            if contains_hindi(clean):
                hi_text = clean
            else:
                hi_text = transliterate_roman_to_hi(clean)

            # ---- Step 3: Translate Hindi → English (not raw Hinglish) ----
            en_text = translate_to_english(hi_text, text)

            # ---- Step 4: Tokenize Hindi text ----
            tokens = tokenize(hi_text)
            tokensFinal = [t.strip() for t in tokens if t.strip()]

            # ---- Display ----
            st.subheader("📊 Pipeline Output")
            c1, c2 = st.columns(2)

            with c1:
                st.write("### Raw Input")
                st.info(text)
                st.write("### Hindi Processed Text")
                st.success(hi_text)
                st.write("### English Translation")
                st.success(en_text)
                st.write("### Tokens")
                st.code(tokensFinal)

            # ---- Step 5: NLP Pipeline ----
            if len(tokensFinal) > 0:
                d = separate_into_dict(tokensFinal)

                try:
                    stop = hindiStopwordsRemover(d)
                except Exception:
                    stop = d

                try:
                    lem = lemmatize_hi(stop)
                except Exception:
                    lem = stop

                try:
                    pos = postagger_hi(lem)
                except Exception:
                    pos = lem

                try:
                    neg = negation_handling_hin(pos)
                except Exception:
                    neg = pos

                # ---- Step 6: Sentiment (pass original texts for phrase matching) ----
                try:
                    label, score = sentiment(neg, original_roman=clean, original_hindi=hi_text)
                except Exception:
                    label, score = ("Neutral", 0)

                # ---- Step 7: Emotion (use English translation) ----
                # try:
                #     emo = emotion_model(en_text)[0][0]
                #     emotion = emo["label"].title()
                #     escore = round(emo["score"], 3)
                # except Exception:
                #     emotion = "Neutral"
                #     escore = 0.0

                # ---- Step 7: Emotion (Hybrid AI + Complaint Detection) ----
                try:
                    lower = text.lower()
                
                    if any(word in lower for word in [
                        "hang", "slow", "crash", "lag",
                        "baar baar", "again and again",
                        "freeze", "stuck"
                    ]):
                        emotion = "Frustration"
                        escore = 0.92
                
                    elif any(word in lower for word in [
                        "wow", "wah", "amazing", "mast", "awesome"
                    ]):
                        emotion = "Joy"
                        escore = 0.91
                
                    else:
                        emo = emotion_model(en_text)[0][0]
                        emotion = emo["label"].title()
                        escore = round(emo["score"], 3)
                
                except Exception:
                    emotion = "Neutral"
                    escore = 0.0

                # ---- Right column ----
                with c2:
                    st.write("### Stopword Removed")
                    st.json(safe_json(stop))
                    st.write("### Lemmatization")
                    st.json(safe_json(lem))
                    st.write("### POS Tags")
                    st.json(safe_json(pos))
                    st.write("### Negation Handling")
                    st.json(safe_json(neg))

                # ---- Final Results ----
                st.divider()
                st.subheader("🎯 Final Prediction")
                r1, r2 = st.columns(2)

                with r1:
                    st.info(f"🎭 Emotion: {emotion} ({escore})")

                with r2:
                    if label == "Positive":
                        st.success(f"😊 Sentiment: {label} ({score})")
                    elif label == "Negative":
                        st.error(f"😠 Sentiment: {label} ({score})")
                    else:
                        st.warning(f"😐 Sentiment: {label} ({score})")
            else:
                st.error("Could not process text. Please try again.")

    except Exception as e:
        st.exception(e)