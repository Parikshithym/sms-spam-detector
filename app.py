
from flask import Flask, render_template, request, jsonify
import pickle, re, nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

app       = Flask(__name__)
stop_words = set(stopwords.words("english"))

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    text  = text.lower()
    text  = re.sub(r"[^a-z\s]", "", text)
    text  = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json()
    message    = data.get("message", "").strip()
    cleaned    = preprocess_text(message)
    vec        = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    prob       = model.predict_proba(vec)[0]
    return jsonify({
        "is_spam"  : bool(prediction == 1),
        "label"    : "SPAM" if prediction == 1 else "NOT SPAM",
        "spam_pct" : round(float(prob[1]) * 100, 1),
        "ham_pct"  : round(float(prob[0]) * 100, 1),
        "message"  : message
    })
