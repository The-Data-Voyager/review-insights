# ===========================================
# preprocess.py — Load and clean the dataset
# ===========================================
import pandas as pd


def load_reviews(filepath):
    """
    Load the CSV file and return the dataframe.
    """
    reviews = pd.read_csv(filepath)
    return reviews


def clean_reviews(reviews):
    """
    Extract review texts and ratings, remove empty reviews.
    Returns clean texts, clean ratings, and original review texts.
    """
    review_texts = reviews["Review Text"].fillna("").tolist()
    ratings = reviews["Rating"].fillna(3).tolist()

    clean_texts = []
    clean_ratings = []
    for i in range(len(review_texts)):
        if review_texts[i] != "":
            clean_texts.append(review_texts[i])
            clean_ratings.append(ratings[i])

    print("Original reviews:", len(review_texts))
    print("After removing empty:", len(clean_texts))

    return review_texts, clean_texts, clean_ratings


def get_sentiment_labels(clean_ratings):
    """
    Convert numerical ratings to sentiment labels.
    1-2 = Negative, 3 = Neutral, 4-5 = Positive
    """
    sentiment_labels = []
    for rating in clean_ratings:
        if rating <= 2:
            sentiment_labels.append("Negative")
        elif rating == 3:
            sentiment_labels.append("Neutral")
        else:
            sentiment_labels.append("Positive")
    return sentiment_labels