import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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


@st.cache_resource
def train_rating_predictor(embeddings, ratings):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(embeddings, ratings)
    return model


# ===========================================
# Load and process data
# ===========================================
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

# Build name list for dropdowns
name_list = []
for cluster_num in range(num_clusters):
    name_list.append(cluster_names[cluster_num])


# ===========================================
# Page Layout
# ===========================================
st.title("Review Insights")
st.write("NLP-powered topic discovery and semantic search on 22,000+ clothing reviews")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Cluster Map",
    "Find Similar Reviews",
    "Browse by Topic",
    "Word Clouds",
    "Sentiment Analysis",
    "Rating Predictor"
])


# ===========================================
# TAB 1: Cluster Map
# ===========================================
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


# ===========================================
# TAB 2: Find Similar Reviews
# ===========================================
with tab2:
    st.header("Find Similar Reviews")
    st.write("Type a review and find the most similar ones in the dataset.")

    with st.form("search_form"):
        user_input = st.text_input("Enter a review:", "This dress is beautiful and fits perfectly")
        num_results = st.slider("Number of results:", 1, 10, 3)
        submitted = st.form_submit_button("Search")

    if submitted:
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

            if score > 0.7:
                strength = "Strong Match"
            elif score > 0.5:
                strength = "Moderate Match"
            else:
                strength = "Weak Match"

            st.write(f"**{strength} ({score})** | Topic: {topic} | Sentiment: {sentiment}")
            st.write(clean_texts[i])
            st.write("---")

        best_score = similarities[top_indices[-1]]
        if best_score < 0.5:
            st.warning("All matches are weak. Try a more detailed review for better results. Example: 'The fabric was cheap and the stitching came apart after one wash'")


# ===========================================
# TAB 3: Browse by Topic
# ===========================================
with tab3:
    st.header("Browse Reviews by Topic")

    selected_topic = st.selectbox("Select a topic:", name_list, key="browse_select")
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


# ===========================================
# TAB 4: Word Clouds
# ===========================================
with tab4:
    st.header("Word Clouds by Topic")
    st.write("Most frequent meaningful words in each cluster.")

    selected_cloud = st.selectbox("Select a topic for word cloud:", name_list, key="cloud_select")

    cloud_cluster = 0
    for cluster_num in range(num_clusters):
        if cluster_names[cluster_num] == selected_cloud:
            cloud_cluster = cluster_num

    all_text = ""
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == cloud_cluster:
            all_text = all_text + " " + clean_texts[i]

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Set2",
        stopwords=WordCloud().stopwords,
        max_words=100
    ).generate(all_text)

    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud: {selected_cloud}", fontsize=16, color="white")
    fig_wc.patch.set_facecolor("black")
    st.pyplot(fig_wc)


# ===========================================
# TAB 5: Sentiment Analysis
# ===========================================
with tab5:
    st.header("Sentiment Analysis")
    st.write("Rating distribution across each topic cluster.")

    chart_data = []
    for i in range(len(cluster_labels)):
        row = {
            "Topic": cluster_names[cluster_labels[i]],
            "Sentiment": sentiment_labels[i],
            "Rating": clean_ratings[i]
        }
        chart_data.append(row)

    chart_df = pd.DataFrame(chart_data)

    # Sentiment distribution per cluster
    st.subheader("Sentiment Breakdown by Topic")

    sentiment_counts = chart_df.groupby(["Topic", "Sentiment"]).size().reset_index(name="Count")

    fig1 = px.bar(
        sentiment_counts,
        x="Topic",
        y="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"},
        barmode="group",
        title="Sentiment Distribution by Topic"
    )
    fig1.update_layout(width=800, height=500)
    st.plotly_chart(fig1)

    # Average rating per cluster
    st.subheader("Average Rating by Topic")

    avg_ratings = chart_df.groupby("Topic")["Rating"].mean().reset_index()
    avg_ratings["Rating"] = avg_ratings["Rating"].round(2)

    fig2 = px.bar(
        avg_ratings,
        x="Topic",
        y="Rating",
        color="Rating",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        title="Average Star Rating by Topic"
    )
    fig2.update_layout(width=800, height=400, yaxis_range=[0, 5])
    st.plotly_chart(fig2)

    # Rating distribution histogram
    st.subheader("Rating Distribution per Topic")

    selected_topic_chart = st.selectbox("Select a topic:", name_list, key="rating_select")

    topic_df = chart_df[chart_df["Topic"] == selected_topic_chart]

    fig3 = px.histogram(
        topic_df,
        x="Rating",
        nbins=5,
        color_discrete_sequence=["#3498db"],
        title=f"Rating Distribution: {selected_topic_chart}"
    )
    fig3.update_layout(width=800, height=400, bargap=0.1)
    st.plotly_chart(fig3)


# ===========================================
# TAB 6: Rating Predictor
# ===========================================
with tab6:
    st.header("Rating Predictor")
    st.write("Type a review and the model will predict its star rating (1-5).")

    rating_model = train_rating_predictor(embeddings, clean_ratings)

    with st.form("predict_form"):
        predict_input = st.text_area("Write a review:", "I love this dress! It fits perfectly and the fabric is amazing.")
        predict_submitted = st.form_submit_button("Predict Rating")

    if predict_submitted:
        model = load_model()
        input_vec = model.encode(predict_input)
        input_vec = input_vec.reshape(1, -1)

        predicted_rating = rating_model.predict(input_vec)[0]
        predicted_rating = int(predicted_rating)

        stars = ""
        for s in range(predicted_rating):
            stars = stars + "⭐"

        st.subheader(f"Predicted Rating: {stars} ({predicted_rating}/5)")

        probabilities = rating_model.predict_proba(input_vec)[0]
        st.write("**Confidence for each rating:**")
        for rating_val in range(len(probabilities)):
            actual_rating = rating_model.classes_[rating_val]
            confidence = round(probabilities[rating_val] * 100, 1)
            bar = "█" * int(confidence / 2)
            st.write(f"{'⭐' * int(actual_rating)} → {confidence}% {bar}")