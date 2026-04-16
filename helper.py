# import stanza
# import string
# import re
# import csv
# import os

# nlp_hi = stanza.Pipeline("hi", processors="tokenize,pos,lemma", use_gpu=False)

# # ================== CLEANING ==================
# def filter(text):
#     text = re.sub(r"\d+", "", text)
#     text = re.sub(r"(https?|ftp|www)\S+", "", text)
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = re.sub(" +", " ", text).strip()
#     return text

# def tokenize(text):
#     return text.split()

# def separate_into_dict(tokens):
#     return {i: t for i, t in enumerate(tokens)}

# # ================== TRANSLITERATION (Expanded) ==================
# roman_to_hi = {
#     # Common words
#     "yaar": "यार", "yar": "यार", "dost": "दोस्त", "bhai": "भाई",
#     "fir": "फिर", "phir": "फिर", "aur": "और", "bhi": "भी",
#     "kya": "क्या", "kyu": "क्यों", "kyun": "क्यों", "kaise": "कैसे",
#     "kaisa": "कैसा", "kaisi": "कैसी", "kab": "कब", "kaha": "कहाँ",
#     "ye": "ये", "yeh": "यह", "wo": "वो", "woh": "वह",
#     "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी",
#     "ho": "हो", "hota": "होता", "hoti": "होती", "hua": "हुआ",
#     "hui": "हुई", "hue": "हुए",
#     "nahi": "नहीं", "nahin": "नहीं", "nah": "नहीं", "na": "ना", "mat": "मत",
#     "bahut": "बहुत", "bohot": "बहुत", "bohat": "बहुत",
#     "bilkul": "बिल्कुल", "bilkool": "बिल्कुल",
#     "ab": "अब", "to": "तो", "toh": "तो", "hi": "ही",
#     "ka": "का", "ki": "की", "ke": "के", "me": "में", "par": "पर", "se": "से",
#     "ko": "को", "ne": "ने", "wala": "वाला", "wali": "वाली", "wale": "वाले",
#     "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
#     "tera": "तेरा", "teri": "तेरी", "tere": "तेरे",
#     "uska": "उसका", "uski": "उसकी", "unka": "उनका",
#     "sab": "सब", "sabse": "सबसे", "kuch": "कुछ", "koi": "कोई",

#     # Positive words
#     "acha": "अच्छा", "accha": "अच्छा", "achha": "अच्छा", "achchi": "अच्छी",
#     "mast": "मस्त", "maast": "मस्त",
#     "badhiya": "बढ़िया", "badiya": "बढ़िया",
#     "shandar": "शानदार", "shaandar": "शानदार",
#     "zabardast": "ज़बरदस्त", "jabardast": "ज़बरदस्त",
#     "kamaal": "कमाल", "kamal": "कमाल",
#     "sahi": "सही", "theek": "ठीक", "thik": "ठीक",
#     "khushi": "खुशी", "khush": "खुश",
#     "pyaar": "प्यार", "pyar": "प्यार",
#     "wah": "वाह", "waah": "वाह",
#     "pasand": "पसंद", "mazedaar": "मज़ेदार", "mazedar": "मज़ेदार",
#     "sundar": "सुंदर", "sunder": "सुंदर",
#     "tagda": "तगड़ा", "super": "सुपर",

#     # Negative words
#     "bekar": "बेकार", "bekaar": "बेकार",
#     "bakwas": "बकवास", "bakvaas": "बकवास",
#     "ghatiya": "घटिया", "ghatia": "घटिया",
#     "bura": "बुरा", "buri": "बुरी", "bure": "बुरे",
#     "kharab": "खराब", "kharab": "खराब",
#     "wahiyat": "वाहियात", "vahiyat": "वाहियात",
#     "ganda": "गंदा", "gandi": "गंदी", "gande": "गंदे",
#     "bekam": "बेकाम", "faltu": "फालतू", "faaltu": "फालतू",
#     "tatti": "बकवास", "chutiya": "बेकार",
#     "pareshaan": "परेशान", "pareshan": "परेशान",
#     "gussa": "गुस्सा", "naraz": "नाराज़", "naraaz": "नाराज़",
#     "dukhi": "दुखी", "udaas": "उदास", "udas": "उदास",
#     "takleef": "तकलीफ", "taklif": "तकलीफ",
#     "mushkil": "मुश्किल",

#     # Verbs / Action words
#     "chala": "चला", "chali": "चली", "chale": "चले",
#     "gaya": "गया", "gayi": "गई", "gaye": "गए",
#     "aaya": "आया", "aayi": "आई", "aaye": "आए", "aa": "आ",
#     "ja": "जा", "jaa": "जा", "gya": "गया",
#     "kar": "कर", "karo": "करो", "karna": "करना", "kiya": "किया",
#     "dekh": "देख", "dekho": "देखो", "dekha": "देखा",
#     "bol": "बोल", "bolo": "बोलो", "bola": "बोला",
#     "sun": "सुन", "suno": "सुनो", "suna": "सुना",
#     "le": "ले", "lo": "लो", "liya": "लिया", "lena": "लेना",
#     "de": "दे", "do": "दो", "diya": "दिया", "dena": "देना",
#     "mil": "मिल", "mila": "मिला", "mili": "मिली",
#     "lag": "लग", "laga": "लगा", "lagi": "लगी", "lagta": "लगता",
#     "ruk": "रुक", "ruka": "रुका", "ruki": "रुकी", "band": "बंद",
#     "tut": "टूट", "tuta": "टूटा", "tuti": "टूटी", "toota": "टूटा",
#     "gir": "गिर", "gira": "गिरा", "giri": "गिरी",
#     "chal": "चल", "chalte": "चलते",

#     # Tech / Common nouns
#     "internet": "इंटरनेट", "net": "नेट",
#     "phone": "फोन", "mobile": "मोबाइल",
#     "traffic": "ट्रैफिक", "signal": "सिग्नल",
#     "wifi": "वाईफाई", "server": "सर्वर",
#     "app": "ऐप", "game": "गेम", "movie": "मूवी", "film": "फिल्म",
#     "gaana": "गाना", "gana": "गाना",
#     "paisa": "पैसा", "paise": "पैसे", "rupay": "रुपये", "rupaye": "रुपये",
#     "kaam": "काम", "kam": "काम", "time": "टाइम",
#     "log": "लोग", "aadmi": "आदमी", "ladka": "लड़का", "ladki": "लड़की",
#     "khana": "खाना", "paani": "पानी", "chai": "चाय",
#     "ghar": "घर", "school": "स्कूल", "college": "कॉलेज",
#     "dukan": "दुकान", "shop": "शॉप",
#     "din": "दिन", "raat": "रात", "subah": "सुबह", "shaam": "शाम",
#     "kal": "कल", "aaj": "आज", "abhi": "अभी",

#     # Expressions / Fillers
#     "are": "अरे", "arey": "अरे", "arre": "अरे",
#     "oye": "ओए", "oy": "ओए",
#     "haan": "हाँ", "han": "हाँ",
#     "okay": "ठीक", "ok": "ठीक",
#     "sorry": "सॉरी", "thankyou": "धन्यवाद", "thanks": "शुक्रिया",
#     "please": "प्लीज",

#     # Adjectives / Descriptors
#     "slow": "स्लो", "fast": "फास्ट",
#     "bada": "बड़ा", "badi": "बड़ी", "bade": "बड़े",
#     "chota": "छोटा", "choti": "छोटी", "chote": "छोटे",
#     "naya": "नया", "nayi": "नई", "naye": "नए", "purana": "पुराना",
#     "zyada": "ज़्यादा", "jyada": "ज़्यादा", "kam": "कम",
#     "pura": "पूरा", "puri": "पूरी", "pure": "पूरे",
#     "alag": "अलग", "same": "सेम",

#     # Common multi-purpose
#     "raha": "रहा", "rahi": "रही", "rahe": "रहे",
#     "wapis": "वापस", "vapas": "वापस",
#     "pehle": "पहले", "baad": "बाद",
#     "saath": "साथ", "bina": "बिना",
#     "sirf": "सिर्फ", "bas": "बस",
#     "itna": "इतना", "itni": "इतनी", "kitna": "कितना", "kitni": "कितनी",

#     # Tech words
#     "laptop": "लैपटॉप",
#     "computer": "कंप्यूटर",
#     "pc": "पीसी",
#     "system": "सिस्टम",
    
#     # Issue words
#     "hang": "हैंग",
#     "freeze": "फ्रीज़",
#     "crash": "क्रैश",
#     "lag": "लैग",
#     "slow": "स्लो",
    
#     # Frequency words
#     "baar": "बार",
#     "bar": "बार",
    
#     # Actions
#     "chal": "चल",
#     "chalna": "चलना",
#     "chalraha": "चल रहा",
#     "ruk": "रुक",
#     "ruk gaya": "रुक गया"
# }

# def transliterate_roman_to_hi(text):
#     words = text.lower().split()
#     return " ".join(roman_to_hi.get(w, w) for w in words)

# # ================== STOPWORDS ==================
# def hindiStopwordsRemover(d):
#     stop = {"है","था","थे","थी","और","का","की","के","में","पर","तो","ही","भी",
#             "यह","ये","वह","वो","होना","को","ने","से","हैं","हो"}
#     return {k: v for k, v in d.items() if v not in stop}

# # ================== PIPELINE ==================
# def lemmatize_hi(d):
#     text = " ".join(d.values())
#     doc = nlp_hi(text)
#     out = {}
#     i = 0
#     for s in doc.sentences:
#         for w in s.words:
#             out[i] = w.lemma if w.lemma else w.text
#             i += 1
#     return out

# def postagger_hi(d):
#     text = " ".join(d.values())
#     doc = nlp_hi(text)
#     out = {}
#     i = 0
#     for s in doc.sentences:
#         for w in s.words:
#             out[i] = f"{w.text}/{w.upos}"
#             i += 1
#     return out

# def negation_handling_hin(d):
#     out = {}
#     negate = False
#     for k, v in d.items():
#         token = str(v).split("/")[0] if "/" in str(v) else str(v)
#         if token in ["नहीं", "ना", "मत", "not", "no"]:
#             negate = True
#             continue
#         if negate:
#             out[k] = "!" + str(v)
#             negate = False
#         else:
#             out[k] = str(v)
#     return out

# # ================== LOAD HINDI SENTIWORDNET ==================
# _hinswn_pos = {}
# _hinswn_neg = {}

# def _load_hinswn():
#     """Load Hindi SentiWordNet from hinSWN.csv"""
#     global _hinswn_pos, _hinswn_neg
#     csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hinSWN.csv")
#     if not os.path.exists(csv_path):
#         return
#     try:
#         with open(csv_path, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 synset = row.get("Synset", "")
#                 pos_score = float(row.get("Positive", 0))
#                 neg_score = float(row.get("Negative", 0))
#                 # Each synset can contain multiple words separated by comma
#                 words = [w.strip() for w in synset.split(",")]
#                 for word in words:
#                     word = word.strip()
#                     if not word:
#                         continue
#                     # Store the strongest score we find for each word
#                     if pos_score > _hinswn_pos.get(word, 0):
#                         _hinswn_pos[word] = pos_score
#                     if neg_score > _hinswn_neg.get(word, 0):
#                         _hinswn_neg[word] = neg_score
#     except Exception:
#         pass

# _load_hinswn()

# # ================== COMPREHENSIVE SENTIMENT ==================

# # --- Hinglish phrase patterns (on ORIGINAL roman text, before transliteration) ---
# # These capture idiomatic Hinglish expressions + their sentiment
# _negative_hinglish_phrases = [
#     # Internet/tech frustration
#     "chala gaya", "chal gya", "chala gya", "chali gayi", "band ho gaya",
#     "band ho gya", "band hogaya", "kaam nahi", "kaam ni", "kam nahi",
#     "kaam nhi", "nahi chal raha", "nhi chal rha", "nahi aa raha",
#     "nahi khul raha", "slow hai", "slow ho gaya", "hang ho gaya",
#     "hang ho gya", "tut gaya", "toot gaya", "gir gaya", "kharab ho gaya",
#     "kharab hogaya", "bigad gaya", "crash ho gaya",
#     # General frustration
#     "kya bakwas", "kitna bakwas", "bilkul bekar", "bahut bekar",
#     "bohot bekar", "bahut kharab", "bahut bura", "bahut ganda",
#     "ekdum bekar", "totally bekar", "full bakwas",
#     "paisa barbaad", "time waste", "time barbaad",
#     "dimag kharab", "sir dard", "tang aa gaya", "tang aa gayi",
#     "pareshan ho gaya", "bahut pareshan",
#     # "fir" (again) + negative context
#     "fir se", "phir se", "fir chala", "phir chala",
#     "fir band", "phir band", "fir kharab", "phir kharab",
#     # Disappointment
#     "umeed nahi", "koi fayda nahi", "bekar hai", "bakwas hai",
#     "ghatiya hai", "wahiyat hai", "faltu hai",
#     "pasand nahi", "acha nahi", "accha nahi",
# ]

# _positive_hinglish_phrases = [
#     "bahut acha", "bahut accha", "bahut achha", "bohot acha",
#     "bahut mast", "bahut badhiya", "bohot badhiya",
#     "bahut shandar", "bahut zabardast",
#     "kya mast", "kya badhiya", "kya shandar",
#     "full mast", "ekdum mast",
#     "dil khush", "bahut khush", "bahut pasand",
#     "kamaal hai", "kamaal ka", "kamal hai",
#     "bahut sahi", "bilkul sahi",
#     "maza aa gaya", "maja aa gaya", "maza aaya", "maja aaya",
#     "acha hai", "accha hai", "acchi hai",
#     "acha laga", "accha laga", "pasand aaya", "pasand aya",
#     "pyaar hai", "pyar hai",
#     "wah wah", "waah waah",
#     "best hai", "superb", "amazing",
# ]

# # --- Hindi word-level sentiment (comprehensive) ---
# _positive_words = {
#     "अच्छा", "अच्छी", "बढ़िया", "शानदार", "मस्त", "वाह", "कमाल",
#     "ज़बरदस्त", "सही", "ठीक", "खुश", "खुशी", "प्यार", "सुंदर",
#     "पसंद", "मज़ेदार", "तगड़ा", "सुपर", "फास्ट", "नया",
#     "शुक्रिया", "धन्यवाद",
# }

# _negative_words = {
#     "खराब", "घटिया", "बेकार", "बुरा", "बुरी", "वाहियात", "बकवास",
#     "गंदा", "गंदी", "फालतू", "परेशान", "गुस्सा", "नाराज़",
#     "दुखी", "उदास", "तकलीफ", "मुश्किल", "स्लो", "टूटा", "टूटी",
#     "बंद", "गिरा",
# }

# _negative_words.update({
#     "हैंग",
#     "फ्रीज़",
#     "क्रैश",
#     "लैग",
#     "स्लो"
# })

# # --- Hindi phrase patterns (on transliterated Hindi text) ---
# _negative_hi_phrases = [
#     "चला गया", "चली गई", "चले गए",
#     "बंद हो गया", "बंद हो गई", "बंद हो",
#     "काम नहीं", "नहीं चल", "नहीं आ",
#     "फिर चला", "फिर बंद", "फिर खराब",
#     "बहुत खराब", "बहुत बुरा", "बहुत बेकार",
#     "बिल्कुल बेकार", "पैसा बर्बाद",
#     "दिमाग खराब", "तंग आ",
#     "खराब हो गया", "टूट गया", "गिर गया",
# ]

# _positive_hi_phrases = [
#     "बहुत अच्छा", "बहुत अच्छी", "बहुत मस्त", "बहुत बढ़िया",
#     "बहुत शानदार", "बहुत ज़बरदस्त",
#     "क्या मस्त", "क्या बढ़िया", "क्या शानदार",
#     "कमाल है", "कमाल का",
#     "मज़ा आ गया", "मज़ा आया",
#     "अच्छा है", "अच्छी है", "अच्छा लगा", "पसंद आया",
#     "दिल खुश", "बहुत खुश",
#     "वाह वाह",
# ]


# def sentiment(negation_dict, original_roman="", original_hindi=""):
#     """
#     Multi-layer sentiment analysis:
#       1. Hinglish phrase detection (on original Roman text)
#       2. Hindi phrase detection (on transliterated Hindi text)
#       3. Hindi SentiWordNet lookup
#       4. Lexicon-based word matching
#       5. Negation-aware scoring
#     """
#     score = 0.0
#     roman_lower = original_roman.lower().strip()
#     hindi_text = original_hindi.strip()

#     # ---------- Layer 1: Hinglish phrase matching ----------
#     for phrase in _negative_hinglish_phrases:
#         if phrase in roman_lower:
#             score -= 2.0

#     for phrase in _positive_hinglish_phrases:
#         if phrase in roman_lower:
#             score += 2.0

#     # ---------- Layer 2: Hindi phrase matching ----------
#     for phrase in _negative_hi_phrases:
#         if phrase in hindi_text:
#             score -= 2.0

#     for phrase in _positive_hi_phrases:
#         if phrase in hindi_text:
#             score += 2.0

#     # ---------- Layer 3: Hindi SentiWordNet (on lemmatized/processed words) ----------
#     for item in negation_dict.values():
#         word = str(item).split("/")[0].replace("!", "").strip()
#         negated = str(item).startswith("!")

#         swn_pos = _hinswn_pos.get(word, 0)
#         swn_neg = _hinswn_neg.get(word, 0)

#         if swn_pos > 0.25 or swn_neg > 0.25:
#             word_score = swn_pos - swn_neg
#             if negated:
#                 word_score = -word_score
#             score += word_score

#     # ---------- Layer 4: Lexicon word match ----------
#     for item in negation_dict.values():
#         word = str(item).split("/")[0].replace("!", "").strip()
#         negated = str(item).startswith("!")

#         if word in _positive_words:
#             score += (-1.0 if negated else 1.0)
#         elif word in _negative_words:
#             score += (1.0 if negated else -1.0)

#     # ---------- Layer 5: Check original Hindi tokens for word-level sentiment ----------
#     if hindi_text:
#         for word in hindi_text.split():
#             if word in _negative_words and score >= 0:
#                 # Only nudge if phrase matching didn't already catch it
#                 score -= 0.5
#             elif word in _positive_words and score <= 0:
#                 score += 0.5

#     # ---------- Final decision ----------
#     if score > 0:
#         return ("Positive", round(abs(score), 2))
#     elif score < 0:
#         return ("Negative", round(abs(score), 2))
#     return ("Neutral", 0)



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