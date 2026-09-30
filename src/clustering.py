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
    n_init is pinned so the cluster numbering is reproducible across runs.
    """
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(embeddings)
    return cluster_labels


def get_cluster_topics(clean_texts, cluster_labels, num_clusters=5, top_n=10):
    """
    Find the words that are DISTINCTIVE to each cluster.

    TF-IDF is fitted once on all reviews, then each cluster's average score is
    compared with the overall average. Fitting a separate TF-IDF inside each
    cluster only finds words that are common there ("love", "great", "fit",
    "size" in every cluster), not words that tell the clusters apart.
    """
    cluster_labels = np.asarray(cluster_labels)

    tfidf = TfidfVectorizer(stop_words="english", min_df=5)
    tfidf_matrix = tfidf.fit_transform(clean_texts)
    words = tfidf.get_feature_names_out()

    overall_avg = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    cluster_topics = {}

    for cluster_num in range(num_clusters):

        positions = np.where(cluster_labels == cluster_num)[0]

        if len(positions) == 0:
            cluster_topics[cluster_num] = ["(empty)"]
            continue

        cluster_avg = np.asarray(tfidf_matrix[positions].mean(axis=0)).flatten()

        difference = cluster_avg - overall_avg
        top_positions = np.argsort(difference)[-top_n:][::-1]

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