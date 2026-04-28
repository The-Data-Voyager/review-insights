# Review Insights — NLP Topic Discovery & Semantic Search

An NLP pipeline that analyzes 22,000+ women's clothing reviews to discover hidden topics, classify sentiment, predict ratings, and find semantically similar reviews using embeddings.

## What This Project Does

1. **Text Embeddings** — Converts review text into numerical vectors using `all-MiniLM-L6-v2` sentence transformer
2. **Topic Discovery** — Uses KMeans clustering on embeddings to automatically group reviews into topics
3. **Topic Naming** — Uses TF-IDF to identify the most distinctive words in each cluster
4. **Sentiment Analysis** — Classifies reviews as Positive, Neutral, or Negative with interactive charts
5. **Word Clouds** — Visual representation of most frequent words per topic
6. **Rating Prediction** — Logistic Regression model predicts star rating from review text
7. **Semantic Search** — Finds the most similar reviews using cosine similarity with match-strength scoring
8. **Streamlit App** — Web interface with 6 interactive tabs

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
- **Rating Prediction**: Logistic Regression
- **Dimensionality Reduction**: UMAP
- **Visualization**: Plotly, Matplotlib, WordCloud
- **Web App**: Streamlit
- **Data Processing**: pandas, numpy

## Dataset

Download the dataset from [Kaggle: Women's Clothing E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) and place the CSV file in the project root folder.

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Project Structure

```
review-insights/
├── app.py                 ← Streamlit web application (6 tabs)
├── README.md
├── requirements.txt
├── Womens Clothing E-Commerce Reviews.csv
├── notebooks/
│   └── exploration.ipynb  ← Interactive notebook
└── src/
    ├── preprocess.py      ← Data loading & cleaning
    ├── embeddings.py      ← Embedding generation
    ├── clustering.py      ← KMeans + TF-IDF topics
    ├── similarity.py      ← Semantic search
    └── visualize.py       ← Interactive charts
```