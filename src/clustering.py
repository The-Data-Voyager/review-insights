# ===========================================
# clustering.py — KMeans clustering + TF-IDF topic naming
# ===========================================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import umap


def reduce_dimensions(embeddings):
    """
    Use UMAP to reduce embeddings from 384 dimensions to 2D.
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d


def run_kmeans(embeddings, num_clusters=5):
    """
    Run KMeans clustering on embeddings.
    Returns the cluster label for each review.
    """
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans_model.fit_predict(embeddings)
    return cluster_labels


def get_cluster_topics(clean_texts, cluster_labels, num_clusters=5):
    """
    Use TF-IDF to find the top 10 words for each cluster.
    """
    cluster_topics = {}

    for cluster_num in range(num_clusters):

        reviews_in_cluster = []
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
                reviews_in_cluster.append(clean_texts[i])

        if len(reviews_in_cluster) == 0:
            cluster_topics[cluster_num] = ["(empty)"]
            continue

        tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = tfidf.fit_transform(reviews_in_cluster)

        words = tfidf.get_feature_names_out()
        avg_scores = tfidf_matrix.mean(axis=0)
        avg_scores = np.array(avg_scores).flatten()
        top_positions = np.argsort(avg_scores)[-10:]

        top_words = []
        for pos in top_positions:
            top_words.append(words[pos])

        cluster_topics[cluster_num] = top_words

    return cluster_topics


def print_cluster_summary(cluster_labels, cluster_names, sentiment_labels, num_clusters=5):
    """
    Print review count and sentiment breakdown per cluster.
    """
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
        print(f"{cluster_names[cluster_num]} → {total} reviews (Pos: {pos_count}, Neu: {neu_count}, Neg: {neg_count})")