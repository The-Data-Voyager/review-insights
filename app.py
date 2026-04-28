import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import plotly.graph_objects as go


@st.cache_data
def load_and_process():
    reviews = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
    review_texts = reviews["Review Text"].fillna("").tolist()
    ratings = reviews["Rating"].fillna(3).tolist()

    clean_texts = []
    clean_ratings = []
    for i in range(len(review_texts)):
        if review_texts[i] != "":
            clean_texts.append(review_texts[i])
            clean_ratings.append(ratings[i])

    return clean_texts, clean_ratings


@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@st.cache_data
def generate_all_embeddings(clean_texts):
    model = load_model()
    embeddings = model.encode(clean_texts, show_progress_bar=True)
    return np.array(embeddings)


@st.cache_data
def run_clustering(embeddings, num_clusters=5):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans_model.fit_predict(embeddings)
    return embeddings_2d, cluster_labels


clean_texts, clean_ratings = load_and_process()
embeddings = generate_all_embeddings(tuple(clean_texts))
embeddings_2d, cluster_labels = run_clustering(embeddings)

cluster_names = {
    0: "Sizing & Ordering",
    1: "Dresses & Skirts",
    2: "General Positive",
    3: "Tops & Sweaters",
    4: "Pants & Jeans"
}

num_clusters = 5

# Add sentiment based on rating
sentiment_labels = []
for rating in clean_ratings:
    if rating <= 2:
        sentiment_labels.append("Negative")
    elif rating == 3:
        sentiment_labels.append("Neutral")
    else:
        sentiment_labels.append("Positive")

st.title("Review Insights")
st.write("NLP-powered topic discovery and semantic search on 22,000+ clothing reviews")

tab1, tab2, tab3 = st.tabs(["Cluster Map", "Find Similar Reviews", "Browse by Topic"])

with tab1:
    st.header("Review Clusters")
    st.write("Each dot is a review. Colors represent automatically discovered topics.")

    fig = go.Figure()

    for cluster_num in range(num_clusters):
        cluster_x = []
        cluster_y = []
        cluster_text = []

        for i in range(len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
                cluster_x.append(embeddings_2d[i, 0])
                cluster_y.append(embeddings_2d[i, 1])
                short_text = clean_texts[i][:100]
                hover = short_text + " | " + sentiment_labels[i]
                cluster_text.append(hover)

        fig.add_trace(go.Scatter(
            x=cluster_x,
            y=cluster_y,
            mode="markers",
            marker=dict(size=3, opacity=0.5),
            name=cluster_names[cluster_num],
            text=cluster_text,
            hoverinfo="text+name"
        ))

    fig.update_layout(
        width=800,
        height=600,
        legend=dict(font=dict(size=12))
    )

    st.plotly_chart(fig)

    st.subheader("Cluster Summary")
    for cluster_num in range(num_clusters):
        pos_count = 0
        neg_count = 0
        neu_count = 0
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
                if sentiment_labels[i] == "Positive":
                    pos_count = pos_count + 1
                elif sentiment_labels[i] == "Negative":
                    neg_count = neg_count + 1
                else:
                    neu_count = neu_count + 1
        total = pos_count + neg_count + neu_count
        st.write(f"**{cluster_names[cluster_num]}** → {total} reviews (Positive: {pos_count}, Neutral: {neu_count}, Negative: {neg_count})")

with tab2:
    st.header("Find Similar Reviews")
    st.write("Type a review and find the most similar ones in the dataset.")

    user_input = st.text_input("Enter a review:", "This dress is beautiful and fits perfectly")
    num_results = st.slider("Number of results:", 1, 10, 3)

    if st.button("Search"):
        model = load_model()
        input_vector = model.encode(user_input)

        similarities = []
        for vec in embeddings:
            sim = cosine_similarity([input_vector], [vec])[0][0]
            similarities.append(sim)

        indices = np.argsort(similarities)
        top_indices = indices[-num_results:]

        st.subheader("Results:")
        for i in reversed(top_indices):
            score = round(similarities[i], 4)
            topic = cluster_names[cluster_labels[i]]
            sentiment = sentiment_labels[i]
            st.write(f"**Similarity: {score}** | Topic: {topic} | Sentiment: {sentiment}")
            st.write(clean_texts[i])
            st.write("---")

with tab3:
    st.header("Browse Reviews by Topic")

    name_list = []
    for cluster_num in range(num_clusters):
        name_list.append(cluster_names[cluster_num])

    selected_topic = st.selectbox("Select a topic:", name_list)
    selected_sentiment = st.selectbox("Filter by sentiment:", ["All", "Positive", "Neutral", "Negative"])

    selected_cluster = 0
    for cluster_num in range(num_clusters):
        if cluster_names[cluster_num] == selected_topic:
            selected_cluster = cluster_num

    topic_reviews = []
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == selected_cluster:
            if selected_sentiment == "All":
                topic_reviews.append(clean_texts[i])
            elif sentiment_labels[i] == selected_sentiment:
                topic_reviews.append(clean_texts[i])

    st.write(f"Showing **{selected_sentiment}** reviews from **{selected_topic}** ({len(topic_reviews)} found)")

    for j in range(min(20, len(topic_reviews))):
        st.write(topic_reviews[j])
        st.write("---")