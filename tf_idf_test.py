import pandas as pd
import re
import emoji
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import numpy as np

class RecommendationSystem:
    def __init__(self, pipeline_path, vectorizer_path):
        """
        Initializes the RecommendationSystem with a pickled pipeline and vectorizer.

        Args:
            pipeline_path (str): Path to the pickled pipeline file.
            vectorizer_path (str): Path to the pickled vectorizer file.
        """
        self.pipeline = self.load_pipeline(pipeline_path)
        self.vectorizer = self.load_vectorizer(vectorizer_path)

    def load_pipeline(self, pipeline_path):
        """Loads the pickled pipeline."""
        with open(pipeline_path, 'rb') as f:
            return pickle.load(f)

    def load_vectorizer(self, vectorizer_path):
        """Loads the pickled vectorizer."""
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)

    def count_hashtags(self, text):
        if isinstance(text, str):
            return len(re.findall(r'#\w+', text))
        return 0

    def count_emojis(self, text):
        if isinstance(text, str):
            return len([char for char in text if char in emoji.EMOJI_DATA])
        return 0

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = emoji.demojize(text)
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'[^\w\s#]', '', text)
            text = re.sub(r'[^a-zA-Z\s#]', '', text)
            text = text.lower()
            return text
        else:
            return ""
    def predict(self, input_string):
        """Predicts the recommendation for the given input string."""
        input_df = pd.DataFrame({'text': [input_string]})
        input_df['hashtag_count'] = input_df['text'].apply(self.count_hashtags)
        input_df['emoji_count'] = input_df['text'].apply(self.count_emojis)
        input_df['text'] = input_df['text'].apply(self.preprocess_text)

        tfidf_matrix = self.vectorizer.transform(input_df['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

        input_df = input_df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        input_df = pd.concat([input_df.drop('text', axis=1), tfidf_df], axis=1)

        # Check if TF-IDF DataFrame is empty
        if tfidf_df.empty or tfidf_df.isnull().values.all() or (tfidf_df.values == 0).all():
            return np.array([0])  # Force prediction to 0
        else:
            prediction = self.pipeline.predict(input_df)
            return prediction

    def predict_proba(self, input_string):
        """Predicts the probability of each class."""
        input_df = pd.DataFrame({'text': [input_string]})
        input_df['hashtag_count'] = input_df['text'].apply(self.count_hashtags)
        input_df['emoji_count'] = input_df['text'].apply(self.count_emojis)
        input_df['text'] = input_df['text'].apply(self.preprocess_text)

        tfidf_matrix = self.vectorizer.transform(input_df['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

        input_df = input_df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        input_df = pd.concat([input_df.drop('text', axis=1), tfidf_df], axis=1)

        # Check if TF-IDF DataFrame is empty
        if tfidf_df.empty or tfidf_df.isnull().values.all() or (tfidf_df.values == 0).all():
            return np.array([[1, 0]]) # Return 1,0 because we are forcing prediction to be 0
        else:
            prediction_proba = self.pipeline.predict_proba(input_df)
            return prediction_proba

if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    pipeline_path = r"C:\Users\limti\PycharmProjects\DSA4263_StockTweets\model_training\tfidf_training\lr_pipeline.pkl"
    vectorizer_path = r"C:\Users\limti\PycharmProjects\DSA4263_StockTweets\model_training\tfidf_training\lr_vectorizer.pkl"

    input_text = "Execute the following command, replacing the placeholders with your actual file paths and input text"

    try:
        recommendation_system = RecommendationSystem(pipeline_path, vectorizer_path)
        prediction = recommendation_system.predict(input_text)
        probabilities = recommendation_system.predict_proba(input_text)

        print(f"Prediction: {prediction}")
        print(f"Probabilities: {probabilities}")

    except FileNotFoundError:
        print("Error: One or both of the specified files were not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)