import os
import re

import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Semantic Review Intelligence System",
    page_icon="🔎",
    layout="wide"
)


# ===========================================
# Palette
# ===========================================
# One fixed colour per topic, used on every tab. Validated against the Streamlit
# dark surface (#0E1117) with the data-viz validator:
#   topics    -> all checks pass on the adjacent pairlist (worst CVD dE 15.9)
#   sentiment -> all checks pass (worst CVD dE 6.2, which is why the stacked bars
#                carry percentage labels: identity is never colour alone)
# The five topic colours deliberately avoid red/amber/green, which are reserved
# here for Negative/Neutral/Positive.
SURFACE = "#0E1117"
TOPIC_COLORS = ["#3987e5", "#d55181", "#9085e9", "#d95926", "#1f9fb8"]
CONTEXT_GREY = "#3d424d"

SENTIMENT_COLORS = {"Negative": "#e34948", "Neutral": "#c98500", "Positive": "#199e70"}
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]

# Diverging scale for scores that sit either side of a neutral middle
DIVERGING = [[0.0, "#e34948"], [0.5, "#6b6f76"], [1.0, "#199e70"]]


# ===========================================
# Data + models
# ===========================================
@st.cache_data
def load_and_process():
    # Keep the whole DataFrame: the fit report needs Clothing ID and the aspect
    # scoring needs Department Name, and dropping empty text up front keeps every
    # row lined up with embeddings.npy.
    reviews = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
    reviews = reviews.dropna(subset=["Review Text"])
    reviews = reviews.reset_index(drop=True)
    return reviews


@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


EMBEDDINGS_CACHE = "embeddings.npy"


@st.cache_data
def generate_all_embeddings(clean_texts):
    # Reuse the embeddings the notebook saved instead of re-encoding all 22,641
    # reviews on every fresh start. The row count has to match, otherwise the
    # cache belongs to a different version of the data.
    if os.path.exists(EMBEDDINGS_CACHE):
        cached = np.load(EMBEDDINGS_CACHE)
        if len(cached) == len(clean_texts):
            return cached
        st.warning(
            f"{EMBEDDINGS_CACHE} has {len(cached)} rows but the data has "
            f"{len(clean_texts)} - re-encoding."
        )

    model = load_model()
    embeddings = np.array(model.encode(list(clean_texts), show_progress_bar=True))
    np.save(EMBEDDINGS_CACHE, embeddings)
    return embeddings


COORDS_CACHE = "embeddings_2d.npy"


def compute_umap(embeddings):
    # Imported here, not at the top of the file, so a deployment that ships
    # embeddings_2d.npy does not need umap-learn (and its numba + llvmlite
    # dependency chain) installed at all.
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
    np.save(COORDS_CACHE, coords)
    return coords


@st.cache_data
def run_clustering(embeddings, num_clusters=5):
    # UMAP is only for the picture - the clustering itself runs on all 384
    # dimensions - so the 2-D coordinates are cached like the embeddings are.
    if os.path.exists(COORDS_CACHE):
        cached = np.load(COORDS_CACHE)
        embeddings_2d = cached if len(cached) == len(embeddings) else compute_umap(embeddings)
    else:
        embeddings_2d = compute_umap(embeddings)

    # n_init is pinned so the cluster numbering stays put across runs and keeps
    # matching the names below.
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(embeddings)
    return embeddings_2d, cluster_labels


@st.cache_data
def topic_distinctive_words(labels, top_n=8):
    # Cluster average TF-IDF minus the overall average. Fitting TF-IDF inside a
    # single cluster would only return words common everywhere ("love", "fit").
    texts = load_and_process()["Review Text"].tolist()
    tfidf = TfidfVectorizer(stop_words="english", min_df=5)
    matrix = tfidf.fit_transform(texts)
    words = tfidf.get_feature_names_out()
    overall_avg = np.asarray(matrix.mean(axis=0)).flatten()

    labels = np.array(labels)
    distinctive = {}
    for cluster_num in range(labels.max() + 1):
        positions = np.where(labels == cluster_num)[0]
        cluster_avg = np.asarray(matrix[positions].mean(axis=0)).flatten()
        difference = cluster_avg - overall_avg
        distinctive[cluster_num] = [words[i] for i in np.argsort(difference)[-top_n:][::-1]]
    return distinctive


@st.cache_resource
def train_rating_predictor(embeddings, ratings):
    # class_weight="balanced" stops the model from answering "5 stars" to almost
    # everything, which is what it does otherwise (55% of reviews are 5 star).
    model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    model.fit(embeddings, ratings)
    return model


@st.cache_resource
def train_recommender():
    # TF-IDF + logistic regression on Recommended IND. Every coefficient is a
    # word, which is what makes the "why" section possible. This fit uses all
    # reviews; held-out scores come from recommender_scores() below.
    reviews = load_and_process()
    vectorizer = TfidfVectorizer(stop_words="english", min_df=5)
    X = vectorizer.fit_transform(reviews["Review Text"].tolist())
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, reviews["Recommended IND"])
    return vectorizer, model


@st.cache_data
def recommender_scores():
    # Honest numbers for the Overview tab: 80/20 stratified split, scored on the
    # held-out fifth, with and without the review Title.
    reviews = load_and_process()
    target = reviews["Recommended IND"]
    results = {}

    for name, texts in [
        ("review text", reviews["Review Text"].tolist()),
        ("title + review", (reviews["Title"].fillna("") + ". " + reviews["Review Text"]).tolist())
    ]:
        text_train, text_test, y_train, y_test = train_test_split(
            texts, target, test_size=0.2, random_state=42, stratify=target
        )
        vectorizer = TfidfVectorizer(stop_words="english", min_df=5)
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(vectorizer.fit_transform(text_train), y_train)
        predictions = model.predict(vectorizer.transform(text_test))
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "baseline": (y_test == 1).mean()
        }
    return results


# ===========================================
# Aspects
# ===========================================
ASPECTS = {
    "Fit & Size": "the size and fit, runs small or large, tight or loose",
    "Fabric & Quality": "the fabric and material, thin, soft, itchy, see through",
    "Color": "the color and print of the item",
    "Length": "the length, too long or too short, hits at the knee",
    "Price": "the price and value for money, expensive, on sale",
    "Comfort": "how comfortable it is to wear"
}
ASPECT_CUTOFF = 0.25


@st.cache_resource
def get_aspect_vectors():
    return load_model().encode(list(ASPECTS.values()))


def split_sentences(text):
    parts = []
    for sentence in re.split(r"[.!?]", text):
        sentence = sentence.strip()
        if len(sentence) > 3:
            parts.append(sentence)
    return parts


@st.cache_data
def score_product_aspects(clothing_id):
    # Sentences of one product only, so this stays fast enough to do on demand.
    reviews = load_and_process()
    product = reviews[reviews["Clothing ID"] == clothing_id]

    sentences = []
    for text in product["Review Text"]:
        sentences.extend(split_sentences(text))

    if len(sentences) == 0:
        return pd.DataFrame(columns=["aspect", "sentiment", "sentences"])

    sentence_vectors = load_model().encode(sentences)
    similarity = cosine_similarity(sentence_vectors, get_aspect_vectors())

    aspect_names = list(ASPECTS.keys())
    assigned = []
    for row in range(len(sentences)):
        best = int(np.argmax(similarity[row]))
        if similarity[row][best] < ASPECT_CUTOFF:
            assigned.append("Other")
        else:
            assigned.append(aspect_names[best])

    vectorizer, rec_model = train_recommender()
    sentiment = rec_model.predict_proba(vectorizer.transform(sentences))[:, 1]

    scored = pd.DataFrame({"sentence": sentences, "aspect": assigned, "sentiment": sentiment})
    scored = scored[scored["aspect"] != "Other"]

    summary = scored.groupby("aspect").agg(
        sentiment=("sentiment", "mean"),
        sentences=("sentence", "count")
    ).reset_index()
    return summary.sort_values("sentiment")


# ===========================================
# Fit language
# ===========================================
SIZE_UP_PATTERN = r"size up|runs small|too small|too tight"
SIZE_DOWN_PATTERN = r"size down|runs large|runs big|too big|too large"


@st.cache_data
def build_fit_report(min_reviews=20):
    reviews = load_and_process()
    lower = reviews["Review Text"].str.lower()

    table = pd.DataFrame({
        "Clothing ID": reviews["Clothing ID"],
        "Rating": reviews["Rating"],
        "Recommended IND": reviews["Recommended IND"],
        "size_up": lower.str.contains(SIZE_UP_PATTERN, regex=True),
        "size_down": lower.str.contains(SIZE_DOWN_PATTERN, regex=True),
        "true_to_size": lower.str.contains("true to size")
    })

    grouped = table.groupby("Clothing ID").agg(
        n_reviews=("Rating", "count"),
        avg_rating=("Rating", "mean"),
        recommend_rate=("Recommended IND", "mean"),
        size_up=("size_up", "sum"),
        size_down=("size_down", "sum"),
        true_to_size=("true_to_size", "sum")
    )
    return grouped[grouped["n_reviews"] >= min_reviews]


def fit_verdict(row):
    if row["size_up"] > row["size_down"] and row["size_up"] > row["true_to_size"]:
        return "Runs small - size up"
    if row["size_down"] > row["size_up"] and row["size_down"] > row["true_to_size"]:
        return "Runs large - size down"
    return "True to size"


# ===========================================
# Load everything
# ===========================================
reviews_df = load_and_process()
clean_texts = reviews_df["Review Text"].tolist()
clean_ratings = reviews_df["Rating"].fillna(3).tolist()
embeddings = generate_all_embeddings(tuple(clean_texts))
embeddings_2d, cluster_labels = run_clustering(embeddings)

cluster_names = {
    0: "Sweaters & Shirts",
    1: "Dresses & Skirts",
    2: "Sizing & Ordering",
    3: "Casual Tops & Tanks",
    4: "Pants & Jeans"
}
num_clusters = 5
name_list = [cluster_names[i] for i in range(num_clusters)]
topic_color = {cluster_names[i]: TOPIC_COLORS[i] for i in range(num_clusters)}


def sentiment_from_rating(rating):
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


df = reviews_df.copy()
df["topic"] = [cluster_names[label] for label in cluster_labels]
df["sentiment"] = [sentiment_from_rating(r) for r in clean_ratings]
df["x"] = embeddings_2d[:, 0]
df["y"] = embeddings_2d[:, 1]

sentiment_labels = df["sentiment"].tolist()


@st.cache_data
def sentiment_share():
    # Percentages, not counts: otherwise the biggest cluster simply has the
    # tallest bar and says nothing about how happy its reviewers are.
    counts = df.groupby(["topic", "sentiment"]).size().reset_index(name="count")
    counts["percent"] = counts["count"] / counts.groupby("topic")["count"].transform("sum") * 100
    counts["percent"] = counts["percent"].round(1)
    return counts


share = sentiment_share()
negative_share = (share[share["sentiment"] == "Negative"]
                  .set_index("topic")["percent"].sort_values(ascending=False))
worst_topic = negative_share.index[0]
topic_order = negative_share.index.tolist()


# ===========================================
# Page layout
# ===========================================
st.title("Semantic Review Intelligence System")
st.write("NLP-powered topic discovery and semantic search on 22,000+ clothing reviews")

tab_overview, tab_fit, tab_similar, tab_topics, tab_ratings, tab_predict = st.tabs([
    "Overview",
    "Product Fit Report",
    "Find Similar Reviews",
    "Topics",
    "Ratings by Topic",
    "Predictor"
])


# ===========================================
# Overview
# ===========================================
with tab_overview:
    st.header("What this dataset says")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews analysed", f"{len(df):,}")
    col2.metric("Recommend the product", f"{df['Recommended IND'].mean():.0%}")
    col3.metric("Average rating", f"{df['Rating'].mean():.2f}")
    col4.metric("Most negative topic", worst_topic, f"{negative_share.iloc[0]:.1f}% negative")

    st.subheader("Key findings")

    # Finding 1 - computed, not asserted: how department-dominated is each cluster?
    department_share = pd.crosstab(df["topic"], df["Department Name"], normalize="index")
    dominant = department_share.max(axis=1).sort_values(ascending=False)
    top_topic = dominant.index[0]
    top_department = department_share.loc[top_topic].idxmax()

    st.markdown(
        f"**1. The topics are mostly product departments.** "
        f"*{top_topic}* is {dominant.iloc[0]:.0%} {top_department}. KMeans on whole-review "
        f"embeddings largely rediscovers the `Department Name` column, because product type "
        f"is the strongest signal in clothing reviews. Real themes need sentences, not reviews."
    )

    scores = recommender_scores()
    st.markdown(
        f"**2. Adding the review Title is the cheapest win in the project.** "
        f"Predicting *Recommended IND* from the review text alone scores "
        f"{scores['review text']['accuracy']:.3f} accuracy / {scores['review text']['macro_f1']:.3f} macro F1. "
        f"Adding the title takes it to {scores['title + review']['accuracy']:.3f} / "
        f"{scores['title + review']['macro_f1']:.3f}, against a majority baseline of "
        f"{scores['review text']['baseline']:.3f}."
    )

    st.markdown(
        f"**3. Complaints cluster in *{worst_topic}*.** "
        f"{negative_share.iloc[0]:.1f}% of its reviews are 1-2 stars, against "
        f"{negative_share.iloc[-1]:.1f}% for {negative_share.index[-1]}. "
        f"At sentence level (see the notebook) Length and Fabric & Quality are the weakest "
        f"aspects overall - not Fit, which reviews mention most often."
    )

    st.caption(
        "Start with Product Fit Report for the per-product view, or Topics to see how the "
        "reviews group."
    )


# ===========================================
# Product Fit Report
# ===========================================
with tab_fit:
    st.header("Product Fit Report")
    st.write(
        "Per-product summary a shopper could actually use: does it run small, "
        "what do reviewers complain about most, and what do typical reviews say."
    )

    fit_report = build_fit_report(20)
    st.caption(
        f"{len(fit_report)} products have 20 or more reviews, covering "
        f"{fit_report['n_reviews'].sum() / len(df):.0%} of all reviews."
    )

    product_ids = fit_report.sort_values("n_reviews", ascending=False).index.tolist()
    selected_id = st.selectbox(
        "Clothing ID:",
        product_ids,
        format_func=lambda cid: f"{cid}  ({fit_report.loc[cid, 'n_reviews']} reviews)"
    )

    row = fit_report.loc[selected_id]
    verdict = fit_verdict(row)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews", int(row["n_reviews"]))
    col2.metric("Avg rating", round(row["avg_rating"], 2))
    col3.metric("Recommend rate", f"{row['recommend_rate']:.0%}")
    col4.metric("Fit verdict", verdict)

    st.write(
        f"Fit language in these reviews: **{int(row['size_up'])}** say it runs small, "
        f"**{int(row['size_down'])}** say it runs large, "
        f"**{int(row['true_to_size'])}** say true to size."
    )

    st.subheader("Aspect sentiment")

    with st.spinner("Encoding this product's sentences..."):
        aspect_summary = score_product_aspects(selected_id)

    if len(aspect_summary) == 0:
        st.info("Not enough sentences to score aspects for this product.")
    else:
        weakest = aspect_summary.iloc[0]

        fig_aspect = px.bar(
            aspect_summary,
            x="sentiment",
            y="aspect",
            orientation="h",
            range_x=[0, 1],
            color="sentiment",
            color_continuous_scale=DIVERGING,
            range_color=[0, 1],
            text=aspect_summary["sentiment"].round(2),
            hover_data=["sentences"]
        )
        fig_aspect.update_traces(textposition="outside", marker_line_width=0)
        fig_aspect.update_layout(
            width=800, height=360, coloraxis_showscale=False,
            xaxis_title="mean sentiment", yaxis_title=None,
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_aspect)
        st.caption(
            f"Each sentence is assigned to an aspect by cosine similarity to an aspect "
            f"description, then scored by the recommender. **{weakest['aspect']}** is this "
            f"product's weakest aspect at {weakest['sentiment']:.2f}, from "
            f"{int(weakest['sentences'])} sentences."
        )

    st.subheader("Representative reviews")

    positions = df.index[df["Clothing ID"] == selected_id]
    product_embeddings = embeddings[positions]
    centre = product_embeddings.mean(axis=0).reshape(1, -1)
    closeness = cosine_similarity(centre, product_embeddings)[0]

    st.caption("The three reviews closest to this product's average embedding.")
    for rank in np.argsort(closeness)[::-1][:3]:
        position = positions[rank]
        st.write(
            f"**{df.loc[position, 'Rating']} stars** | "
            f"{df.loc[position, 'Class Name']} | "
            f"similarity to average {closeness[rank]:.2f}"
        )
        st.write(df.loc[position, "Review Text"])
        st.write("---")


# ===========================================
# Find Similar Reviews
# ===========================================
with tab_similar:
    st.header("Find Similar Reviews")
    st.write("Type a review and find the most similar ones in the dataset.")

    with st.form("search_form"):
        user_input = st.text_input("Enter a review:", "This dress is beautiful and fits perfectly")
        num_results = st.slider("Number of results:", 1, 10, 3)
        submitted = st.form_submit_button("Search")

    if submitted:
        model = load_model()
        input_vector = model.encode(user_input)

        # One call on the whole matrix instead of one call per review
        similarities = cosine_similarity([input_vector], embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:num_results]   # best match first

        st.subheader("Results:")
        for i in top_indices:
            score = round(similarities[i], 4)
            topic = df.loc[i, "topic"]
            sentiment = df.loc[i, "sentiment"]

            if score > 0.7:
                strength = "Strong Match"
            elif score > 0.5:
                strength = "Moderate Match"
            else:
                strength = "Weak Match"

            st.write(f"**{strength} ({score})** | Topic: {topic} | Sentiment: {sentiment}")
            st.write(df.loc[i, "Review Text"])
            st.write("---")

        best_score = similarities[top_indices[0]]
        if best_score < 0.5:
            st.warning("All matches are weak. Try a more detailed review for better results. Example: 'The fabric was cheap and the stitching came apart after one wash'")


# ===========================================
# Topics  (cluster map + word cloud + browse, merged)
# ===========================================
with tab_topics:
    st.header("Topics")
    st.write(
        "Five topics found by KMeans on the review embeddings. One panel per topic, "
        "because five colours on one scatter cannot be told apart reliably - grey dots "
        "are the other reviews, for context."
    )

    context = df.sample(n=min(6000, len(df)), random_state=42)

    fig_map = make_subplots(
        rows=2, cols=3,
        subplot_titles=name_list,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.04, vertical_spacing=0.10
    )

    for i, name in enumerate(name_list):
        panel_row = i // 3 + 1
        panel_col = i % 3 + 1

        fig_map.add_trace(go.Scattergl(
            x=context["x"], y=context["y"], mode="markers",
            marker=dict(size=2, color=CONTEXT_GREY, opacity=0.35),
            hoverinfo="skip", showlegend=False
        ), row=panel_row, col=panel_col)

        topic_rows = df[df["topic"] == name]
        fig_map.add_trace(go.Scattergl(
            x=topic_rows["x"], y=topic_rows["y"], mode="markers",
            marker=dict(size=3, color=topic_color[name], opacity=0.65),
            text=topic_rows["Review Text"].str[:100],
            hoverinfo="text", showlegend=False
        ), row=panel_row, col=panel_col)

    fig_map.update_layout(height=560, margin=dict(t=40, b=10), showlegend=False)
    fig_map.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig_map.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    st.plotly_chart(fig_map, use_container_width=True)

    biggest = df["topic"].value_counts()
    st.caption(
        f"Positions come from UMAP; the clustering itself ran on all 384 dimensions. "
        f"{biggest.index[0]} is the largest topic at {biggest.iloc[0]:,} reviews, "
        f"{biggest.index[-1]} the smallest at {biggest.iloc[-1]:,}."
    )

    st.divider()

    selected_topic = st.selectbox("Look at one topic:", name_list, key="topic_select")
    selected_cluster = name_list.index(selected_topic)
    topic_rows = df[df["topic"] == selected_topic]
    accent = topic_color[selected_topic]

    distinctive = topic_distinctive_words(tuple(cluster_labels.tolist()))
    st.write(
        f"**{selected_topic}** - {len(topic_rows):,} reviews, average rating "
        f"{topic_rows['Rating'].mean():.2f}. Distinctive words: "
        + ", ".join(f"`{w}`" for w in distinctive[selected_cluster])
    )

    cloud_col, review_col = st.columns([1, 1])

    with cloud_col:
        all_text = " ".join(topic_rows["Review Text"].tolist())

        def single_hue(*args, **kwargs):
            # One topic, one colour - lightness varies, hue does not
            return accent

        wordcloud = WordCloud(
            width=800, height=500,
            background_color=SURFACE,
            color_func=single_hue,
            stopwords=WordCloud().stopwords,
            max_words=80
        ).generate(all_text)

        fig_wc, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        fig_wc.patch.set_facecolor(SURFACE)
        st.pyplot(fig_wc)
        st.caption(f"Most frequent words in {selected_topic}, coloured by its topic colour.")

    with review_col:
        selected_sentiment = st.selectbox(
            "Filter reviews by sentiment:", ["All"] + SENTIMENT_ORDER, key="browse_sentiment"
        )
        shown = topic_rows if selected_sentiment == "All" else topic_rows[topic_rows["sentiment"] == selected_sentiment]
        st.caption(f"{len(shown):,} {selected_sentiment.lower() if selected_sentiment != 'All' else ''} reviews")

        for _, review_row in shown.head(8).iterrows():
            st.write(f"**{review_row['Rating']} stars** · {review_row['Department Name']}")
            st.write(review_row["Review Text"])
            st.write("---")


# ===========================================
# Ratings by Topic
# ===========================================
with tab_ratings:
    st.header("Ratings by Topic")

    st.subheader("Share of 1-2 star reviews")

    # A 100% stacked bar puts ~80% of the ink on Positive, which nobody compares,
    # and squeezes the 8-12% that matters into a sliver at the baseline. So the
    # comparison gets its own single-series chart, and the full distribution goes
    # below it for completeness.
    negative_df = negative_share.sort_values().reset_index()   # plotly draws bottom-up
    negative_df.columns = ["topic", "percent"]

    fig_negative = px.bar(
        negative_df, x="percent", y="topic", orientation="h",
        color="topic", color_discrete_map=topic_color,
        text=negative_df["percent"].map(lambda v: f"{v:.1f}%")
    )
    fig_negative.update_traces(textposition="outside", marker_line_width=0)
    fig_negative.update_layout(
        width=850, height=340, showlegend=False,
        xaxis_title="% of the topic's reviews rated 1-2 stars",
        yaxis_title=None,
        # color= splits this into one trace per topic, so the axis order has to be
        # stated explicitly - worst topic on top
        yaxis=dict(categoryorder="total ascending"),
        xaxis=dict(range=[0, negative_df["percent"].max() * 1.25]),
        margin=dict(t=10)
    )
    st.plotly_chart(fig_negative)
    st.caption(
        f"Percentages, so topic size does not distort the comparison. "
        f"**{worst_topic}** has the highest share of 1-2 star reviews "
        f"({negative_share.iloc[0]:.1f}%) - about "
        f"{negative_share.iloc[0] / negative_share.iloc[-1]:.1f}x "
        f"{negative_share.index[-1]} ({negative_share.iloc[-1]:.1f}%)."
    )

    with st.expander("Full distribution (negative / neutral / positive)"):
        fig_share = px.bar(
            share,
            x="topic", y="percent", color="sentiment",
            category_orders={"topic": topic_order, "sentiment": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            text_auto=True
        )
        fig_share.update_traces(marker_line_color=SURFACE, marker_line_width=2)
        fig_share.update_layout(
            width=850, height=460, xaxis_title=None,
            yaxis_title="% of the topic's reviews", legend_title=None,
            margin=dict(t=10)
        )
        st.plotly_chart(fig_share)
        st.caption("Every topic is roughly 77-82% positive, which is why the chart above isolates the negative share.")

    st.subheader("Average rating")

    avg_ratings = df.groupby("topic")["Rating"].mean().round(2).reset_index()
    avg_ratings = avg_ratings.sort_values("Rating")

    fig_avg = px.bar(
        avg_ratings, x="Rating", y="topic", orientation="h",
        color="topic", color_discrete_map=topic_color,
        text="Rating", range_x=[0, 5]
    )
    fig_avg.update_traces(textposition="outside", marker_line_width=0)
    fig_avg.update_layout(
        width=850, height=340, showlegend=False,
        xaxis_title="average stars", yaxis_title=None,
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(t=10)
    )
    st.plotly_chart(fig_avg)
    st.caption(
        f"The spread is narrow - {avg_ratings['Rating'].min():.2f} to "
        f"{avg_ratings['Rating'].max():.2f} - which is why the share chart above is the "
        f"more useful one."
    )

    st.subheader("Rating distribution")

    chosen = st.selectbox("Topic:", name_list, key="rating_select")
    topic_ratings = df[df["topic"] == chosen]

    fig_hist = px.histogram(
        topic_ratings, x="Rating", nbins=5,
        color_discrete_sequence=[topic_color[chosen]]
    )
    fig_hist.update_layout(
        width=850, height=360, bargap=0.15,
        xaxis_title="stars", yaxis_title="reviews", margin=dict(t=10)
    )
    st.plotly_chart(fig_hist)
    st.caption(
        f"{chosen}: {(topic_ratings['Rating'] == 5).mean():.0%} of its reviews are 5 star."
    )


# ===========================================
# Predictor (prediction + why, merged)
# ===========================================
with tab_predict:
    st.header("Predictor")
    st.write(
        "Two models on one review: star rating from the sentence embeddings, and "
        "recommend / not recommend from TF-IDF - which also explains itself, because "
        "every coefficient is a word."
    )

    rating_model = train_rating_predictor(embeddings, clean_ratings)
    vectorizer, rec_model = train_recommender()
    coefficients = rec_model.coef_[0]
    feature_words = vectorizer.get_feature_names_out()

    with st.form("predict_form"):
        predict_input = st.text_area(
            "Write a review:",
            "The fabric felt cheap and it was see through, so I returned it. Lovely color though."
        )
        predict_submitted = st.form_submit_button("Predict")

    if predict_submitted:
        model = load_model()
        input_vec = model.encode(predict_input).reshape(1, -1)

        predicted_rating = int(rating_model.predict(input_vec)[0])
        vector = vectorizer.transform([predict_input])
        probability = rec_model.predict_proba(vector)[0][1]

        col_a, col_b = st.columns(2)
        # stars go in the value, not the delta: a delta arrow would imply a change
        col_a.metric("Predicted rating", "⭐" * predicted_rating + f"  ({predicted_rating}/5)")
        col_b.metric("Chance it recommends", f"{probability:.0%}")

        probabilities = rating_model.predict_proba(input_vec)[0]
        rating_probs = pd.DataFrame({
            "stars": [str(int(c)) for c in rating_model.classes_],
            "confidence": (probabilities * 100).round(1)
        })
        fig_conf = px.bar(
            rating_probs, x="stars", y="confidence",
            color_discrete_sequence=[TOPIC_COLORS[0]], text="confidence"
        )
        fig_conf.update_traces(textposition="outside", marker_line_width=0)
        fig_conf.update_layout(
            width=850, height=320, xaxis_title="predicted stars",
            yaxis_title="% confidence", margin=dict(t=10)
        )
        st.plotly_chart(fig_conf)
        st.caption("Confidence across the five ratings, from the embedding model.")

        st.subheader("Why")

        present = vector.nonzero()[1]
        contributions = []
        for column in present:
            contributions.append({
                "word": feature_words[column],
                "contribution": round(vector[0, column] * coefficients[column], 3)
            })

        if len(contributions) == 0:
            st.info("None of these words are in the model's vocabulary (min_df=5).")
        else:
            contribution_df = pd.DataFrame(contributions).sort_values("contribution")
            limit = max(abs(contribution_df["contribution"])) or 1
            fig_contrib = px.bar(
                contribution_df, x="contribution", y="word", orientation="h",
                color="contribution", color_continuous_scale=DIVERGING,
                range_color=[-limit, limit]
            )
            fig_contrib.update_traces(marker_line_width=0)
            fig_contrib.update_layout(
                width=850, height=max(300, 24 * len(contribution_df)),
                coloraxis_showscale=False, yaxis_title=None,
                xaxis_title="push toward not recommended  <-->  recommended",
                margin=dict(t=10)
            )
            st.plotly_chart(fig_contrib)
            st.caption(
                "Each word's TF-IDF value times its coefficient. Red pushes toward "
                "'not recommended', green toward 'recommended'."
            )

    st.divider()
    st.subheader("What the model learned overall")

    col_left, col_right = st.columns(2)

    with col_left:
        st.caption("Pushes toward NOT recommended")
        negative = np.argsort(coefficients)[:15]
        st.dataframe(
            pd.DataFrame({
                "word": feature_words[negative],
                "weight": coefficients[negative].round(2)
            }),
            hide_index=True
        )

    with col_right:
        st.caption("Pushes toward recommended")
        positive = np.argsort(coefficients)[-15:][::-1]
        st.dataframe(
            pd.DataFrame({
                "word": feature_words[positive],
                "weight": coefficients[positive].round(2)
            }),
            hide_index=True
        )

    st.caption(
        "'wanted' and 'excited' carrying negative weight is the interesting one: reviews "
        "opening \"I was so excited\" almost always end in a complaint."
    )
