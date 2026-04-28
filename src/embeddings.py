# ===========================================
# embeddings.py — Generate text embeddings
# ===========================================
import numpy as np
from sentence_transformers import SentenceTransformer


def load_model():
    """
    Load the sentence transformer model.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def generate_embeddings(model, texts):
    """
    Convert a list of texts into numerical vectors.
    Returns a numpy array of shape (num_texts, 384).
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings)
    return embeddings