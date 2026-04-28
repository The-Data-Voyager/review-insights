# Review Insights — NLP Topic Discovery & Semantic Search

An NLP pipeline that analyzes 22,000+ women's clothing reviews to discover hidden topics, classify sentiment, and find semantically similar reviews using embeddings.

## What This Project Does

1. **Text Embeddings** — Converts review text into numerical vectors using `all-MiniLM-L6-v2` sentence transformer
2. **Topic Discovery** — Uses KMeans clustering on embeddings to automatically group reviews into topics
3. **Topic Naming** — Uses TF-IDF to identify the most distinctive words in each cluster
4. **Sentiment Analysis** — Classifies reviews as Positive, Neutral, or Negative based on star ratings
5. **Interactive Visualization** — 2D scatter plot (UMAP + Plotly) with hover text and sentiment
6. **Semantic Search** — Find the most similar reviews to any input using cosine similarity
7. **Streamlit App** — Web interface for exploring clusters, searching reviews, and filtering by sentiment

## Topics Discovered

| Cluster | Topic | Reviews |
|---------|-------|---------|
| 0 | Sizing & Ordering | ~5,076 |
| 1 | Dresses & Skirts | ~5,798 |
| 2 | General Positive | ~2,987 |
| 3 | Tops & Sweaters | ~6,109 |
| 4 | Pants & Jeans | ~2,671 |

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Clustering**: scikit-learn (KMeans)
- **Topic Extraction**: TF-IDF (TfidfVectorizer)
- **Dimensionality Reduction**: UMAP
- **Visualization**: Plotly
- **Web App**: Streamlit
- **Data Processing**: pandas, numpy

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Project Structure