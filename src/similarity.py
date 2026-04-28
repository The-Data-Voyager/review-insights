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

    similarities = []
    for vec in clean_embeddings:
        sim = cosine_similarity([input_vector], [vec])[0][0]
        similarities.append(sim)

    indices = np.argsort(similarities)
    top_indices = indices[-top_n:]

    results = []
    for i in reversed(top_indices):
        result = {
            "text": clean_texts[i],
            "score": round(similarities[i], 4),
            "topic": cluster_names[cluster_labels[i]],
            "sentiment": sentiment_labels[i]
        }
        results.append(result)

    return results