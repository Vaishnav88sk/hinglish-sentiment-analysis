import os
import re
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

_hinglish_idioms = {
    # Roman Hinglish idioms → English
    "wah kya baat hai": "wow that's amazing",
    "kya baat hai": "that's amazing",
    "wah kya baat": "wow that's amazing",
    "maza aa gaya": "that was so enjoyable",
    "dil khush ho gaya": "it made me so happy",
    "kya mast hai": "this is so cool",
    "bohot badiya": "that's really great",
    "bahut badiya": "that's really great",
    "bahut acha": "that's very good",
    "bahut accha": "that's very good",
    "bohot acha": "that's very good",
    "bilkul bekar": "completely useless",
    "kuch nahi hota": "nothing works",
    "dimag kharab": "it's driving me crazy",
    "dimag ka dahi": "my mind is messed up",
    "paisa barbaad": "total waste of money",
    "time waste": "waste of time",
    "band bajao": "taught a lesson",
    "dum hai": "it takes courage",
    "jaan de di": "gave it my all",
    "aag laga di": "absolutely killed it",
    "kamaal kar diya": "that was incredible",
    "dil jeet liya": "won my heart",
    "full paisa wasool": "totally worth the money",
    "hawa tight hai": "things are tough right now",
}

_hindi_idioms = {
    # Hindi Devanagari idioms → English
    "वाह क्या बात है": "wow that's amazing",
    "क्या बात है": "that's amazing",
    "वाह क्या बात": "wow that's amazing",
    "मज़ा आ गया": "that was so enjoyable",
    "दिल खुश हो गया": "it made me so happy",
    "बहुत बढ़िया": "that's really great",
    "बहुत अच्छा": "that's very good",
    "बिल्कुल बेकार": "completely useless",
    "दिमाग खराब": "it's driving me crazy",
    "पैसा बर्बाद": "total waste of money",
    "कमाल कर दिया": "that was incredible",
    "दिल जीत लिया": "won my heart",
}


def translate_to_english(hindi_text, original_text):
    """
    General Hinglish/Hindi -> English translation.

    Strategy:
    0. Check idiom dictionary for known phrases
    1. Translate normalized Hindi text first
    2. Reject weak / overly short outputs
    3. Fallback to original text auto-detect
    4. Final fallback = original text
    """

    def clean(txt):
        return txt.strip() if txt else ""

    # ---- Idiom pre-processing ----
    roman_lower = re.sub(r"[^\w\s]", "", original_text.lower()).strip()
    roman_lower = re.sub(r"\s+", " ", roman_lower)

    # Check longest idiom matches first (greedy)
    for phrase, eng in sorted(_hinglish_idioms.items(),
                              key=lambda x: len(x[0]), reverse=True):
        if phrase in roman_lower:
            remaining = roman_lower.replace(phrase, "").strip()
            parts = []
            if remaining:
                try:
                    extra = GoogleTranslator(
                        source="auto", target="en"
                    ).translate(remaining)
                    if extra and extra.strip():
                        parts.append(extra.strip())
                except Exception:
                    parts.append(remaining)
            parts.append(eng)
            return " ".join(parts)

    # Check Hindi idiom matches
    for phrase, eng in sorted(_hindi_idioms.items(),
                              key=lambda x: len(x[0]), reverse=True):
        if phrase in hindi_text:
            remaining = hindi_text.replace(phrase, "").strip()
            parts = []
            if remaining:
                try:
                    extra = GoogleTranslator(
                        source="hi", target="en"
                    ).translate(remaining)
                    if extra and extra.strip():
                        parts.append(extra.strip())
                except Exception:
                    parts.append(remaining)
            parts.append(eng)
            return " ".join(parts)

    # ---- Standard translation ----
    try:
        result = GoogleTranslator(
            source="hi",
            target="en"
        ).translate(hindi_text)

        result = clean(result)

        # Accept only meaningful output
        if result:

            # If translator returned same Hindi text
            if result.lower() == hindi_text.lower():
                raise Exception()

            # If input has multiple words but output too short
            in_words = len(hindi_text.split())
            out_words = len(result.split())

            if in_words >= 3 and out_words <= 1:
                raise Exception()

            return result

    except Exception:
        pass

    try:
        result = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(original_text)

        result = clean(result)

        if result:
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

st.markdown(
    """
🔗 **Project Repository:** [View Source Code, Contribute & See Sample Inputs](https://github.com/your-username/your-repo)

🚀 Explore the repository for **setup instructions, documentation, contribution guide, and more sample test inputs**.
"""
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