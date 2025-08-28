from flask import Flask, request, jsonify
from flask_cors import CORS
import cohere
from textblob import TextBlob
from langdetect import detect
import spacy
from textstat import flesch_reading_ease
from nrclex import NRCLex
from newspaper import Article
import os
import subprocess
import nltk

app = Flask(__name__)
CORS(app)

# ✅ Ensure TextBlob + NLTK dependencies are available
required_corpora = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),  # 🔥 NEW: Fixes your error
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("corpora/brown", "brown"),
    ("corpora/movie_reviews", "movie_reviews"),
]

for path, name in required_corpora:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)

# ✅ Use environment variable for API key (never hardcode secrets!)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ Missing COHERE_API_KEY environment variable")
co = cohere.Client(COHERE_API_KEY)

# ✅ Load spaCy safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

@app.route('/')
def index():
    return jsonify({"message": "✅ Flask API for News Analysis is running!"})

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url']
    try:
        # 1. Extract article
        article = Article(url)
        article.download()
        article.parse()
        user_text = article.text or ""

        if not user_text.strip():
            return jsonify({'error': 'Failed to extract content from URL'}), 500

        # 2. Summarization with Cohere
        prompt = f"Summarize the following news article:\n\n{user_text}"
        summary_response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=200
        )
        summary = summary_response.generations[0].text.strip()

        # 3. Sentiment
        blob = TextBlob(summary)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # 4. Keywords (noun phrases)
        keywords = list(set(blob.noun_phrases))

        # 5. Named Entities
        doc = nlp(user_text)
        entities = list(set((ent.text, ent.label_) for ent in doc.ents))

        # 6. Emotion detection
        emotion_obj = NRCLex(user_text)
        emotions = emotion_obj.raw_emotion_scores
        dominant_emotion = max(emotions, key=emotions.get) if emotions else "neutral"

        # 7. Language
        try:
            language = detect(user_text)
        except Exception:
            language = "unknown"

        # 8. Stats
        word_count = len(user_text.split())
        readability_score = flesch_reading_ease(user_text)
        toxicity_score = round(max(0.0, -1 * polarity), 2)

        return jsonify({
            'title': article.title,
            'source': article.source_url,
            'summary': summary,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'keywords': keywords,
            'entities': entities,
            'emotion': dominant_emotion,
            'language': language,
            'word_count': word_count,
            'readability_score': readability_score,
            'toxicity_score': toxicity_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
