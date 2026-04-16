import stanza
import string
import re
import csv
import os

# =====================================================
# LOAD STANZA HINDI PIPELINE
# =====================================================
nlp_hi = stanza.Pipeline(
    "hi",
    processors="tokenize,pos,lemma",
    use_gpu=False
)

# =====================================================
# CLEANING
# =====================================================
def filter(text):
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"(https?|ftp|www)\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(" +", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def separate_into_dict(tokens):
    return {i: t for i, t in enumerate(tokens)}

# =====================================================
# ROMAN HINGLISH -> HINDI
# =====================================================
roman_to_hi = {

    # Common
    "yaar": "यार",
    "yar": "यार",
    "bhai": "भाई",
    "bro": "भाई",
    "dost": "दोस्त",

    "ye": "ये",
    "yeh": "यह",
    "woh": "वह",
    "wo": "वो",

    "hai": "है",
    "hu": "हूँ",
    "ho": "हो",
    "hain": "हैं",

    "tha": "था",
    "thi": "थी",
    "the": "थे",

    "fir": "फिर",
    "phir": "फिर",

    "baar": "बार",
    "bar": "बार",

    "kya": "क्या",
    "kyu": "क्यों",
    "kyun": "क्यों",

    "bahut": "बहुत",
    "bohot": "बहुत",

    "bilkul": "बिल्कुल",

    "mera": "मेरा",
    "meri": "मेरी",
    "mere": "मेरे",
    "naam": "नाम",

    "main": "मैं",
    "mai": "मैं",

    # Time / Daily words
    "aaj": "आज",
    "kal": "कल",
    "abhi": "अभी",
    "din": "दिन",
    "raat": "रात",
    "subah": "सुबह",
    "shaam": "शाम",

    # Grammar
    "ka": "का",
    "ki": "की",
    "ke": "के",
    "me": "में",
    "mein": "में",
    "par": "पर",
    "se": "से",
    "ko": "को",
    "ne": "ने",
    "to": "तो",
    "hi": "ही",
    "bhi": "भी",

    # Positive
    "acha": "अच्छा",
    "accha": "अच्छा",
    "achha": "अच्छा",
    "mast": "मस्त",
    "amazing": "अद्भुत",
    "awesome": "शानदार",
    "tasty": "स्वादिष्ट",
    "fresh": "ताज़ा",
    "wah": "वाह",
    "wow": "वाह",
    "lit": "मस्त",
    "funny": "मजेदार",
    "op": "जबरदस्त",
    "great": "बेहतरीन",
    "nice": "अच्छा",

    # Negative
    "bekar": "बेकार",
    "bakwas": "बकवास",
    "ghatiya": "घटिया",
    "kharab": "खराब",
    "slow": "स्लो",
    "hang": "हैंग",
    "crash": "क्रैश",
    "lag": "लैग",
    "freeze": "फ्रीज़",
    "stuck": "अटका",

    # Tech
    "laptop": "लैपटॉप",
    "phone": "फोन",
    "mobile": "मोबाइल",
    "internet": "इंटरनेट",
    "wifi": "वाईफाई",
    "network": "नेटवर्क",
    "system": "सिस्टम",
    "app": "ऐप",
    "game": "गेम",
    "movie": "मूवी",
    "film": "फिल्म",
    "reel": "रील",

    # Places
    "restaurant": "रेस्टोरेंट",
    "restraurant": "रेस्टोरेंट",
    "service": "सेवा",
    "traffic": "ट्रैफिक",
    "office": "ऑफिस",
    "meeting": "मीटिंग",
    "monday": "सोमवार",

    # Food
    "khana": "खाना",
    "food": "खाना",

    # Verbs
    "chala": "चला",
    "chal": "चल",
    "gaya": "गया",
    "gayi": "गई",
    "gaye": "गए",
    "raha": "रहा",
    "rahi": "रही",
    "rahe": "रहे",
    "gir": "गिर",
    "ja": "जा",
    "hua": "हुआ",
    "select": "चयन",
    "band": "बंद",
}

def transliterate_roman_to_hi(text):
    words = text.lower().split()
    return " ".join(roman_to_hi.get(w, w) for w in words)

# =====================================================
# STOPWORDS
# =====================================================
def hindiStopwordsRemover(d):
    stop = {
        "है", "था", "थी", "थे",
        "और", "का", "की", "के",
        "में", "पर", "तो", "ही", "भी",
        "यह", "ये", "वह", "वो",
        "को", "से", "ने", "हो"
    }
    return {k: v for k, v in d.items() if v not in stop}

# =====================================================
# NLP PIPELINE
# =====================================================
def lemmatize_hi(d):
    text = " ".join(d.values())
    doc = nlp_hi(text)

    out = {}
    i = 0

    for s in doc.sentences:
        for w in s.words:
            out[i] = w.lemma if w.lemma else w.text
            i += 1

    return out


def postagger_hi(d):
    text = " ".join(d.values())
    doc = nlp_hi(text)

    out = {}
    i = 0

    for s in doc.sentences:
        for w in s.words:
            out[i] = f"{w.text}/{w.upos}"
            i += 1

    return out


def negation_handling_hin(d):
    out = {}
    negate = False

    for k, v in d.items():

        token = str(v).split("/")[0]

        if token in ["नहीं", "ना", "मत"]:
            negate = True
            continue

        if negate:
            out[k] = "!" + str(v)
            negate = False
        else:
            out[k] = str(v)

    return out

# =====================================================
# HINDI SENTIWORDNET
# =====================================================
_hinswn_pos = {}
_hinswn_neg = {}

def _load_hinswn():
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "hinSWN.csv"
    )

    if not os.path.exists(csv_path):
        return

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                syn = row.get("Synset", "")
                pos = float(row.get("Positive", 0))
                neg = float(row.get("Negative", 0))

                for word in syn.split(","):
                    w = word.strip()

                    if pos > _hinswn_pos.get(w, 0):
                        _hinswn_pos[w] = pos

                    if neg > _hinswn_neg.get(w, 0):
                        _hinswn_neg[w] = neg
    except:
        pass

_load_hinswn()

# =====================================================
# SENTIMENT DATA
# =====================================================

_positive_words = {
    "अच्छा", "मस्त", "शानदार", "स्वादिष्ट",
    "ताज़ा", "अद्भुत", "वाह", "खुश",
    "बेहतरीन", "जबरदस्त", "मजेदार",
    "सुपर"
}

_negative_words = {
    "बेकार", "बकवास", "घटिया", "खराब",
    "स्लो", "हैंग", "क्रैश", "लैग",
    "फ्रीज़", "अटका",
    "बुरा", "गंदा", "परेशान", "मुश्किल"
}

_positive_hinglish_phrases = [
    "aaj ka din bahut accha gaya",
    "bahut tasty",
    "wah kya baat hai",
    "wah kya service hai",
    "movie mast",
    "game op",
    "party lit",
    "amazing service",
    "ye phone amazing",
    "kitna taja hai",
]

_negative_hinglish_phrases = [
    "that restaurant is not good",
    "itna ghatiya khana",
    "baar baar hang",
    "baar baar crash",
    "internet slow",
    "internet fir band ho gaya",
    "system slow",
    "phone bekar",
    "service bakwas",
    "traffic bakwas",
    "app crash",
    "never coming again",
]

_positive_hi_phrases = [
    "आज का दिन बहुत अच्छा गया",
    "बहुत स्वादिष्ट",
    "वाह क्या बात है",
    "वाह क्या सेवा है",
    "मूवी मस्त",
    "गेम जबरदस्त",
    "अद्भुत सेवा",
]

_negative_hi_phrases = [
    "रेस्टोरेंट अच्छा नहीं है",
    "बहुत घटिया खाना",
    "बार बार हैंग",
    "बार बार क्रैश",
    "बहुत बेकार",
    "बिल्कुल बेकार",
    "इंटरनेट फिर बंद हो गया",
    "सेवा बकवास",
    "ट्रैफिक बकवास",
    "अच्छा नहीं",
]

# =====================================================
# FINAL SENTIMENT FUNCTION
# =====================================================
def sentiment(negation_dict, original_roman="", original_hindi=""):

    score = 0.0

    roman = original_roman.lower()
    hindi = original_hindi

    # Phrase match (Roman)
    for p in _positive_hinglish_phrases:
        if p in roman:
            score += 2

    for p in _negative_hinglish_phrases:
        if p in roman:
            score -= 2

    # Phrase match (Hindi)
    for p in _positive_hi_phrases:
        if p in hindi:
            score += 2

    for p in _negative_hi_phrases:
        if p in hindi:
            score -= 2

    # Lexicon + SWN
    for item in negation_dict.values():

        word = str(item).split("/")[0].replace("!", "")
        negated = str(item).startswith("!")

        # Hindi SWN
        pos_score = _hinswn_pos.get(word, 0)
        neg_score = _hinswn_neg.get(word, 0)

        if pos_score > 0 or neg_score > 0:
            val = pos_score - neg_score
            score += (-val if negated else val)

        # Lexicon
        if word in _positive_words:
            score += (-1 if negated else 1)

        elif word in _negative_words:
            score += (1 if negated else -1)

    # Final
    if score > 0:
        return ("Positive", round(abs(score), 2))

    elif score < 0:
        return ("Negative", round(abs(score), 2))

    return ("Neutral", 0)