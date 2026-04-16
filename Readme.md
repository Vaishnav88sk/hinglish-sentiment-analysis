# 🧠 Hinglish Sentiment Analyzer

Analyze **Hindi**, **English**, and **Hinglish (Hindi + English mixed text)** using an advanced NLP pipeline with:

- Tokenization
- Stopword Removal
- Lemmatization
- POS Tagging
- Negation Handling
- Rule-Based Sentiment Analysis
- Hindi SentiWordNet Scoring
- AI Emotion Detection

Built with **Python**, **Streamlit**, **Stanza**, and **Transformers**.

---

# 🚀 Features

- Supports **Hindi text**
- Supports **English text**
- Supports **Roman Hinglish text**
- Converts Hinglish → Hindi for NLP processing
- Shows complete NLP pipeline step-by-step
- Detects Positive / Negative / Neutral sentiment
- Detects emotions like Joy, Anger, Neutral, Surprise, etc.
- Interactive Streamlit UI

---

# 🏗️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web UI |
| Stanza | Hindi NLP |
| Transformers | Emotion detection |
| deep-translator | Translation |
| Hindi SentiWordNet | Lexicon sentiment |

---

# 📂 Project Structure

```text
HinglishSentimentAnalyzer/
├── app.py
├── helper.py
├── hinSWN.csv
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Vaishnav88sk/hinglish-sentiment-analysis.git
cd hinglish-sentiment-analysis
```

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

## 4. Download Stanza Hindi Model

```bash
python -c "import stanza; stanza.download('hi')"
```

---

# ▶️ Run Project

```bash
streamlit run app.py
```

Then open: http://localhost:8501

---

# 📊 NLP Pipeline

Example Input:

```text
Laptop baar baar hang ho raha hai
```

Processing Flow:

1. Clean Text  
2. Detect Language  
3. Hinglish → Hindi Transliteration  
4. Tokenization  
5. Stopword Removal  
6. Lemmatization  
7. POS Tagging  
8. Negation Handling  
9. Sentiment Detection  
10. Emotion Detection

---

# 🧪 Sample Inputs

## 😊 Positive

```text
Ye khaana bahut tasty hai
Aaj ka din bahut accha gaya
Wah kya service hai
Movie mast thi bhai
That restaurant was amazing
```

## 😠 Negative

```text
Ye phone bilkul bekar hai
Laptop baar baar hang ho raha hai
Bakwas service hai
Itna ghatiya khaana kabhi nahi khaya
Yaar internet fir band ho gaya
```

## 😐 Neutral

```text
Mera naam Omkar hai
Meeting 5 baje hai
Aaj Monday hai
Main office ja raha hu
```

## 😮 Emotion / Surprise

```text
Are yaar! ye kya hua!
Sach me? Tu select ho gaya?
Wow! kya baat hai
```

---

# 📸 Example Output

```text
Input:
Laptop baar baar hang ho raha hai

Sentiment:
Negative 😠

Emotion:
Frustration 😩
```

---

# 🧠 How Sentiment Works

## Rule-Based Detection

Checks:

- Positive words (`अच्छा`, `मस्त`, `amazing`)
- Negative words (`बेकार`, `घटिया`, `hang`)
- Phrase patterns (`baar baar hang`, `bahut tasty`)

## Hindi SentiWordNet

Uses sentiment scores for Hindi words.

## Negation Handling

```text
अच्छा नहीं
```

becomes negative.

---

# 🎭 Emotion Detection

Uses Hugging Face model:

`j-hartmann/emotion-english-distilroberta-base`

Detects:

- Joy 😊
- Anger 😠
- Sadness 😢
- Fear 😨
- Surprise 😮
- Neutral 😐

---

# ⭐ If You Like This Project

Star the repository and share feedback.