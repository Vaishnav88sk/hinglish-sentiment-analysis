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

    # ---------------- Pronouns ----------------
    "mai": "मैं",
    "main": "मैं",
    "me": "मैं",
    "mera": "मेरा",
    "meri": "मेरी",
    "mere": "मेरे",
    "tu": "तू",
    "tum": "तुम",
    "aap": "आप",
    "ye": "ये",
    "yeh": "यह",
    "wo": "वो",
    "woh": "वह",
    "hum": "हम",

    # ---------------- Common Expressions ----------------
    "yaar": "यार",
    "yar": "यार",
    "bhai": "भाई",
    "bro": "भाई",
    "dost": "दोस्त",
    "arre": "अरे",
    "arey": "अरे",
    "abe": "अबे",
    "oh": "ओह",
    "wow": "वाह",
    "wah": "वाह",
    "sach": "सच",

    # ---------------- Grammar (Hinglish) ----------------
    "hai": "है",
    "hu": "हूँ",
    "ho": "हो",
    "hain": "हैं",
    "tha": "था",
    "thi": "थी",
    "the": "थे",

    "ka": "का",
    "ki": "की",
    "ke": "के",
    "ko": "को",
    "se": "से",
    "par": "पर",
    "mein": "में",
    "me": "में",
    "ne": "ने",
    "to": "तो",
    "hi": "ही",
    "bhi": "भी",

    # ---------------- English Grammar ----------------
    "i": "मैं",
    "my": "मेरा",
    "is": "है",
    "was": "था",
    "were": "थे",
    "are": "हैं",
    "am": "हूँ",
    "been": "था",
    "had": "था",
    "has": "है",
    "have": "है",
    "that": "वह",
    "this": "यह",
    "it": "यह",
    "its": "इसका",
    "not": "नहीं",
    "no": "नहीं",
    "never": "कभी नहीं",
    "very": "बहुत",
    "so": "बहुत",
    "too": "भी",
    "also": "भी",
    "but": "लेकिन",
    "and": "और",
    "or": "या",
    "with": "साथ",
    "without": "बिना",
    "for": "के लिए",
    "from": "से",
    "of": "का",
    "in": "में",
    "on": "पर",
    "at": "पर",
    "again": "फिर",
    "always": "हमेशा",
    "every": "हर",
    "really": "सच में",
    "just": "बस",
    "only": "सिर्फ",

    # ---------------- Question Words ----------------
    "kya": "क्या",
    "kyu": "क्यों",
    "kyun": "क्यों",
    "kaise": "कैसे",
    "kab": "कब",
    "kahan": "कहाँ",
    "kaun": "कौन",

    # ---------------- Time / Daily ----------------
    "aaj": "आज",
    "kal": "कल",
    "abhi": "अभी",
    "subah": "सुबह",
    "shaam": "शाम",
    "raat": "रात",
    "din": "दिन",
    "baje": "बजे",
    "bajey": "बजे",
    "clock": "बजे",
    "minute": "मिनट",
    "minutes": "मिनट",
    "hour": "घंटा",
    "hours": "घंटे",
    "pm": "पीएम",
    "am": "एएम",
    "today": "आज",
    "tomorrow": "कल",

    # ---------------- Frequency ----------------
    "fir": "फिर",
    "phir": "फिर",
    "baar": "बार",
    "bar": "बार",

    # ---------------- Positive ----------------
    "acha": "अच्छा",
    "accha": "अच्छा",
    "achha": "अच्छा",
    "mast": "मस्त",
    "badiya": "बढ़िया",
    "badhiya": "बढ़िया",
    "awesome": "शानदार",
    "amazing": "अद्भुत",
    "great": "बेहतरीन",
    "super": "सुपर",
    "nice": "अच्छा",
    "fresh": "ताज़ा",
    "tasty": "स्वादिष्ट",
    "funny": "मजेदार",
    "lit": "मस्त",
    "op": "जबरदस्त",
    "good": "अच्छा",
    "excellent": "उत्कृष्ट",
    "wonderful": "अद्भुत",
    "fantastic": "शानदार",
    "beautiful": "सुंदर",
    "lovely": "प्यारा",
    "best": "सबसे अच्छा",
    "perfect": "उत्तम",
    "happy": "खुश",
    "love": "प्यार",
    "loved": "पसंद आया",
    "liked": "पसंद आया",
    "enjoy": "मज़ा",
    "enjoyed": "मज़ा आया",
    "favorite": "पसंदीदा",
    "delicious": "स्वादिष्ट",

    # ---------------- Negative ----------------
    "bekar": "बेकार",
    "bakwas": "बकवास",
    "ghatiya": "घटिया",
    "kharab": "खराब",
    "bura": "बुरा",
    "slow": "स्लो",
    "hang": "हैंग",
    "crash": "क्रैश",
    "lag": "लैग",
    "freeze": "फ्रीज़",
    "stuck": "अटका",
    "bad": "बुरा",
    "worst": "सबसे बुरा",
    "terrible": "भयानक",
    "horrible": "भयानक",
    "awful": "बेकार",
    "poor": "घटिया",
    "boring": "उबाऊ",
    "waste": "बर्बाद",
    "useless": "बेकार",
    "hate": "नफरत",
    "hated": "नफरत",
    "annoying": "परेशान",
    "disappointed": "निराश",
    "disgusting": "घिनौना",

    # ---------------- Tech ----------------
    "laptop": "लैपटॉप",
    "phone": "फोन",
    "mobile": "मोबाइल",
    "internet": "इंटरनेट",
    "wifi": "वाईफाई",
    "network": "नेटवर्क",
    "system": "सिस्टम",
    "computer": "कंप्यूटर",
    "app": "ऐप",
    "game": "गेम",
    "reel": "रील",

    # ------------- Other ----------------
    "movie": "मूवी",
    "film": "फिल्म",
    "cinema": "सिनेमा",
    "hero": "हीरो",
    "song": "गाना",
    "scene": "सीन",

    # ---------------- Places / Things ----------------
    "restaurant": "रेस्टोरेंट",
    "restraurant": "रेस्टोरेंट",
    "service": "सेवा",
    "traffic": "ट्रैफिक",
    "office": "ऑफिस",
    "meeting": "मीटिंग",
    "party": "पार्टी",
    "scene": "सीन",

    # ---------------- Food ----------------
    "khana": "खाना",
    "food": "खाना",
    "chai": "चाय",
    "coffee": "कॉफी",

    # ---------------- Common Nouns ----------------
    "naam": "नाम",
    "baat": "बात",
    "mood": "मूड",

    # ---------------- Verbs ----------------
    "chal": "चल",
    "chala": "चला",
    "ja": "जा",
    "gaya": "गया",
    "gayi": "गई",
    "gaye": "गए",
    "hua": "हुआ",
    "ho": "हो",
    "raha": "रहा",
    "rahi": "रही",
    "rahe": "रहे",
    "gir": "गिर",
    "band": "बंद",
    "khaya": "खाया",
    "jaa": "जा",
}

def transliterate_roman_to_hi(text):
    words = text.lower().split()
    return " ".join(roman_to_hi.get(w, w) for w in words)

# =====================================================
# STOPWORDS
# =====================================================
def hindiStopwordsRemover(d):
    stop = {
        # Hindi stopwords
        "है", "था", "थी", "थे",
        "और", "का", "की", "के",
        "में", "पर", "तो", "ही", "भी",
        "यह", "ये", "वह", "वो",
        "को", "से", "ने", "हो",
        "हूँ", "हैं", "या", "लेकिन",
        "साथ", "बिना", "के लिए",
        "इसका", "मैं", "मेरा",
        "बस", "सिर्फ", "हर", "हमेशा",
        "फिर", "बहुत", "सच में",
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