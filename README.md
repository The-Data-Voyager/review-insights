# Review Insights — NLP Topic Discovery & Semantic Search

An NLP pipeline that analyzes 22,000+ women's clothing reviews to discover hidden topics, classify sentiment, predict ratings, and find semantically similar reviews using embeddings.

## What This Project Does

1. **Text Embeddings** — Converts review text into numerical vectors using `all-MiniLM-L6-v2` sentence transformer
2. **Topic Discovery** — Uses KMeans clustering on embeddings to automatically group reviews into topics
3. **Topic Naming** — Uses TF-IDF to identify the most distinctive words in each cluster
4. **Sentiment Analysis** — Classifies reviews as Positive, Neutral, or Negative with interactive charts
5. **Word Clouds** — Visual representation of most frequent words per topic
6. **Rating Prediction** — Logistic Regression model predicts star rating from review text
7. **Semantic Search** — Finds the most similar reviews with a single vectorized cosine-similarity call and match-strength scoring
8. **Word Embeddings Comparison** — Word2Vec trained on the reviews, compared against TF-IDF and MiniLM on the same split
9. **Aspect-Based Sentiment** — Splits reviews into sentences, assigns each to an aspect (fit, fabric, colour, length, price, comfort) by cosine similarity, and scores it
10. **Per-Product Fit Report** — Fit verdict, weakest aspect and representative reviews for any product with 20+ reviews
11. **Explainable Predictions** — TF-IDF coefficients show exactly which words push a review toward "not recommended"
12. **Streamlit App** — Six tabs, ordered so the findings come first:
    Overview (headline numbers + key findings) · Product Fit Report · Find Similar
    Reviews · Topics (map, word cloud and browsing merged) · Ratings by Topic ·
    Predictor (rating, recommendation and the word-level explanation together)

## Topics Discovered

Cluster topics are named from the words that are *distinctive* to each cluster
(cluster mean TF-IDF minus overall mean TF-IDF), not from the words that are merely
frequent inside it — otherwise every cluster is described as "love, great, fit, size".

| Cluster | Topic | Reviews | Distinctive words |
|---------|-------|---------|-------------------|
| 0 | Sweaters & Shirts | 6,119 | sweater, shirt, soft, blouse, tee, cardigan |
| 1 | Dresses & Skirts | 5,793 | dress, skirt, slip, wedding, waist, flattering |
| 2 | Sizing & Ordering | 5,068 | small, xs, size, medium, petite, lbs |
| 3 | Casual Tops & Tanks | 2,991 | tops, cute, boxy, sheer, tank, peplum, cami |
| 4 | Pants & Jeans | 2,670 | pants, jeans, stretch, shorts, legs, ankle |

### Are these topics, or just departments?

Mostly departments, and the write-up says so rather than pretending otherwise:

```python
pd.crosstab(reviews["cluster"], reviews["Department Name"], normalize="index").round(2)
```

Cluster 3 is 94% Tops, cluster 1 is 83% Dresses, cluster 4 is 81% Bottoms. KMeans on
review-level embeddings largely rediscovers the `Department Name` column, because
product type is the strongest signal in clothing reviews. Real themes need a finer
unit than a whole review — which is what the aspect section below does with sentences.

k = 5 was checked rather than assumed: silhouette on the 384-dim embeddings is flat
at 0.051–0.052 for k = 5, 6 and 7, and drops to 0.032 by k = 8. Low scores are normal
for text embeddings, so k is chosen on those numbers *plus* how readable the topic
words are.

## Prediction Results

80/20 stratified split, logistic regression with `class_weight="balanced"`, scored on
the held-out 20%. The majority baseline is "always guess the most common class".

**Recommended IND** — the headline target, majority baseline 0.819:

| Features | Accuracy | Macro F1 |
|----------|----------|----------|
| TF-IDF, title + review | **0.880** | **0.822** |
| TF-IDF, review text | 0.862 | 0.799 |
| Word2Vec, averaged | 0.821 | 0.757 |
| MiniLM embeddings | 0.813 | 0.742 |

**5-class rating** — the harder task, majority baseline 0.554:

| Features | Accuracy | Macro F1 |
|----------|----------|----------|
| TF-IDF | 0.567 | 0.418 |
| Word2Vec, averaged | 0.532 | 0.408 |
| MiniLM embeddings | 0.522 | 0.385 |

Two things worth noting:

- **`Recommended IND` is the honest headline target.** 5-class rating barely clears the
  55.4% "always guess 5 stars" baseline, because neighbouring ratings are genuinely
  ambiguous in the text. Accuracy alone hides this, which is why macro F1 and the
  confusion matrix are reported alongside it.
- **TF-IDF beats both embedding types on both targets here.** Sentence embeddings win on
  *similarity search*, where meaning matters; for classification the literal words
  ("returned", "cheap", "perfect") carry more signal than 384 averaged dimensions.
  Without `class_weight="balanced"` the 5-class model looks better on accuracy
  (embeddings 0.603, TF-IDF 0.627) but worse on macro F1 — it mostly predicts 5 stars.
- **Adding the Title is the cheapest win in the project**: one line of pandas, +1.8
  points of accuracy and +2.3 points of macro F1. Titles like "Runs small!" are blunt
  where review bodies are chatty.

## Word2Vec: Useful, But Not for Sentiment

Word2Vec (100 dimensions, window 5, min_count 5, 10 epochs) trained on the 22,641
reviews learns a 4,849-word vocabulary and genuinely useful neighbourhoods:

| Word | Nearest words |
|------|---------------|
| snug | tight, **loose**, roomy, baggy, big, large |
| xs | xsp, xxsp, xxs, m, xsmall, p |
| flimsy | scratchy, rough, stiff, thin, unlined, cheap |
| returned | **kept**, exchanged, return, knew, realized |
| gorgeous | beautiful, lovely, stunning, adorable, darling |

Note "snug" next to **loose** and "returned" next to **kept**. Word2Vec learns from
context, and opposites share contexts — "it's a bit ___ around the waist" fits both
tight and loose. Averaging word vectors into one review vector therefore cancels part
of the sentiment, which is why it lands below TF-IDF in the table above.

Where it pays off is **vocabulary discovery**: `most_similar("itchy")` returns
`scratchy, stiff, flimsy, clingy, rough, bulky` — a complaint word list for free, which
is what the aspect descriptions below are built from.

## Aspect-Based Sentiment

Review-level labels hide that one reviewer can love the colour and hate the fit, so the
unit of analysis drops to the sentence: **111,864 sentences** from 22,641 reviews. Each
sentence is assigned to the nearest of six aspect descriptions by cosine similarity, and
anything below 0.25 similarity becomes "Other" (chit-chat like "i am 5'4 and 130 lbs").

Mean sentiment per aspect — the probability the recommender assigns to a review
containing that sentence:

| Aspect | Mean sentiment | Sentences |
|--------|----------------|-----------|
| Length | 0.471 | 8,217 |
| Fabric & Quality | 0.476 | 10,901 |
| Price | 0.542 | 2,871 |
| Colour | 0.640 | 11,119 |
| Fit & Size | 0.640 | 19,873 |
| Comfort | 0.666 | 30,168 |

**Length and fabric are where this retailer loses customers** — not fit, which is the
aspect the reviews talk about most. The notebook renders this as an aspect × department
heatmap, where Length in Trend (0.31) is the single worst cell.

Two honest caveats: "Comfort" is a magnet bucket whose description attracts generic
sentences, and the sentence scorer was trained on whole reviews, so it keys on words
like "disappointed" and "returned". The per-aspect *means* hold up; the single most
negative sentence in an aspect is often just a generic complaint. VADER, or a model
trained on sentences, would sharpen that end.

## Per-Product Fit Report

Fit language across the dataset: "true to size" 1,287 reviews, "size down" 659,
"size up" 488, "too big" 558, "runs large" 274, "runs small" 228, "too small" 224.

165 products have 20 or more reviews, covering 86% of all reviews — enough to report per
product. Of those 165: **75 run large, 59 are true to size, 31 run small.** The app's
Product Fit Report tab shows the verdict, the weakest aspect and the three reviews
closest to that product's average embedding.

## Explainability

The recommender is TF-IDF + logistic regression, so every coefficient is a word:

| Pushes toward NOT recommended | Pushes toward recommended |
|-------------------------------|---------------------------|
| disappointed (-5.7), wanted (-5.1), returning (-4.6), returned (-4.6), cheap (-4.1), unflattering (-3.7), huge (-3.7), excited (-3.6) | love (6.2), perfect (5.5), comfortable (4.7), great (4.5), compliments (4.4), fits (4.3), perfectly (3.8), glad (3.6) |

"wanted" and "excited" carrying *negative* weight is the interesting one: reviews that
open "I was so excited" and "I really wanted to love this" almost always end in a
complaint.

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

To run the app:

```bash
pip install -r requirements.txt
```

To run the notebook as well (adds `umap-learn` and `gensim`):

```bash
pip install -r requirements-notebook.txt
```

`requirements.txt` is what Streamlit Cloud installs, so it stays lean: it pulls
CPU-only torch (the default PyPI wheel drags in ~2.5 GB of CUDA packages) and leaves
out `umap-learn`, because the app ships the 2-D coordinates in `embeddings_2d.npy`
(177 KB) rather than recomputing them.

The notebook caches its two expensive encodes, both gitignored, and the app reuses the
first one instead of encoding at startup:

| File | Size | What it holds |
|------|------|---------------|
| `embeddings.npy` | ~33 MB | one vector per review (22,641 × 384) |
| `sentence_embeddings.npy` | ~170 MB | one vector per sentence (111,864 × 384) |
| `embeddings_2d.npy` | 177 KB | UMAP coordinates for the cluster map |

Encoding the reviews takes ~16 minutes on CPU; the sentences, being short, take ~4.

## Chart Conventions

One fixed colour per topic, used on every tab, and red/amber/green reserved for
Negative/Neutral/Positive so a hue never means two things. Both palettes were checked
with a colour-vision validator against the dark surface rather than picked by eye.

The cluster map is drawn as **one small panel per topic** instead of a single
five-colour scatter: five categorical colours cannot be separated reliably in a
scatter plot (every 5-colour subset fails the colourblind-separation floor when any
two clusters can touch), and a 3px legend dot is unreadable regardless. Panel titles
carry the identity; grey dots give the context.

Bars that compare topics show **percentages, not counts** — otherwise the largest
cluster simply has the tallest bar and the chart says nothing about sentiment.

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
│   └── Clothing_reviews_nlp.ipynb  ← Interactive notebook (end to end)
└── src/
    ├── preprocess.py      ← Data loading & cleaning
    ├── embeddings.py      ← Embedding generation
    ├── clustering.py      ← KMeans + TF-IDF topics
    ├── similarity.py      ← Semantic search
    └── visualize.py       ← Interactive charts
```