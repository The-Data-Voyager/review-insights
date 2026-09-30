# ===========================================
# similarity.py — Find similar reviews
# ===========================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_reviews(input_review, model, clean_texts, clean_embeddings, cluster_labels, cluster_names, sentiment_labels, top_n=3):
    """
    Given a new review, find the most similar reviews.
    Returns a list of dictionaries with text, score, topic, and sentiment.
    """
    input_vector = model.encode(input_review)

    # One call on the whole matrix instead of one call per review
    similarities = cosine_similarity([input_vector], clean_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]   # best match first

    results = []
    for i in top_indices:
        result = {
            "text": clean_texts[i],
            "score": round(similarities[i], 4),
            "topic": cluster_names[cluster_labels[i]],
            "sentiment": sentiment_labels[i]
        }
        results.append(result)

    return results