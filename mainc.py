from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import re
import random
import os
import pickle

app = Flask(__name__)
CORS(app)

# Config
THRESHOLD = 0.6
TOP_K = 5
EMBEDDING_CACHE = "sbert_embeddings_cache.pkl"

# Load SBERT model (English)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load full dataset
df = pd.read_excel("data_final.xlsx")
records = df.to_dict("records")

# Detect if Arabic based on Unicode ranges
def detect_language(text):
    if re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    return 'en'

# Get sentence embedding
def embed_text(text):
    return model.encode(text)

# Precompute or load embeddings
english_embeddings = []
arabic_embeddings = []

def embed_all():
    global english_embeddings, arabic_embeddings

    if os.path.exists(EMBEDDING_CACHE):
        print("🔄 Loading cached SBERT embeddings...")
        with open(EMBEDDING_CACHE, 'rb') as f:
            cache = pickle.load(f)
            english_embeddings = cache['en']
            arabic_embeddings = cache['ar']
        return

    print("⚙️ Generating SBERT embeddings...")
    for record in records:
        lang = record.get("Language")
        title = str(record.get("Title", "")).strip()
        abstract = str(record.get("Abstract", "")).strip()
        if not title and not abstract:
            continue

        combined_text = f"{title}. {abstract}"
        try:
            embedding = embed_text(combined_text)
            if lang == 'en':
                english_embeddings.append((embedding, record))
            else:
                arabic_embeddings.append((embedding, record))
        except Exception as e:
            print(f"❌ Error embedding: {title} — {e}")

    with open(EMBEDDING_CACHE, 'wb') as f:
        pickle.dump({
            'en': english_embeddings,
            'ar': arabic_embeddings
        }, f)
    print("✅ SBERT embeddings cached!")

embed_all()

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    query_text = data["query"]
    query_lang = detect_language(query_text)

    query_embedding = embed_text(query_text)
    pool = english_embeddings if query_lang == 'en' else arabic_embeddings

    results = []
    for emb, record in pool:
        sim = 1 - cosine(query_embedding, emb)
        if not np.isnan(sim):
            results.append((record, sim))

    top_results = sorted(results, key=lambda x: x[1], reverse=True)[:TOP_K]
    return jsonify([
        {
            "title": r["Title"],
            "link": r["Link"],
            "author": r["Author"],
            "abstract": r["Abstract"],
            "date": r["Date"],
            "language": r["Language"],
            "similarity": float(round(sim, 4))
        } for r, sim in top_results
    ])

@app.route("/random_articles", methods=["GET"])
def random_articles():
    lang = request.args.get("lang", default=None)
    filtered = [r for r in records if not lang or r["Language"] == lang]
    selected = random.sample(filtered, min(3, len(filtered)))
    return jsonify([
        {
            "title": r["Title"],
            "link": r["Link"],
            "author": r["Author"],
            "abstract": r["Abstract"],
            "date": r["Date"],
            "language": r["Language"]
        } for r in selected
    ])

if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)