import pandas as pd
import re
import emoji
import joblib  # Using joblib for model persistence
import sys
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

class PumpDetection:
    def __init__(self, pipeline_path, vectorizer_path):
        """
        Initializes the RecommendationSystem with joblib-saved pipeline and vectorizer.
        """
        self.pipeline = self.load_pipeline(pipeline_path)
        self.vectorizer = self.load_vectorizer(vectorizer_path)

    def load_pipeline(self, pipeline_path):
        """Loads the joblib-saved pipeline."""
        try:
            return joblib.load(pipeline_path)
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            sys.exit(1)

    def load_vectorizer(self, vectorizer_path):
        """Loads the joblib-saved vectorizer."""
        try:
            return joblib.load(vectorizer_path)
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            sys.exit(1)

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

    def count_keywords(self, text):
        if isinstance(text, str):
            keywords = ['pump', 'dump', 'moon', 'buy', 'rocket']
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + keyword + r'\b', text))
            return count
        return 0

    def predict(self, input_string):
        """Predicts the recommendation for the given input string and numerical data."""
        input_df = pd.DataFrame({'text': input_string}) #Input must be a list
        print(input_df.head())
        input_df['hashtag_count'] = input_df['text'].apply(self.count_hashtags)
        input_df['emoji_count'] = input_df['text'].apply(self.count_emojis)
        input_df['keyword_count'] = input_df['text'].apply(self.count_keywords)
        input_df['text'] = input_df['text'].apply(self.preprocess_text)

        tfidf_matrix = self.vectorizer.transform(input_df['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

        input_df = input_df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        input_df = pd.concat([input_df.drop('text', axis=1), tfidf_df], axis=1)

        print(input_df)
        # Check if TF-IDF DataFrame is empty
        if tfidf_df.empty or tfidf_df.isnull().values.all() or (tfidf_df.values == 0).all():
            return np.array([0])  # Force prediction to 0
        else:
            prediction = self.pipeline.predict(input_df)
            return prediction

    def predict_proba(self, input_string):
        """Predicts the probability of each class."""
        input_df = pd.DataFrame({'text': [input_string]}) #Input must be a list
        input_df['hashtag_count'] = input_df['text'].apply(self.count_hashtags)
        input_df['emoji_count'] = input_df['text'].apply(self.count_emojis)
        input_df['keyword_count'] = input_df['text'].apply(self.count_keywords)
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
    pipeline_path = "gradient_boosting_pipeline.joblib"
    vectorizer_path = "gradient_boosting_vectorizer.joblib"

    input_text = "Execute the following command, replacing the placeholders with your actual file paths and input text"
    numerical_data = {"emoji_count": 5, "hashtag_count": 3, "keyword_count": 1} # Example numerical data

    try:
        recommendation_system = PumpDetection(pipeline_path, vectorizer_path)
        prediction = recommendation_system.predict(input_text, numerical_data)
        probabilities = recommendation_system.predict_proba(input_text, numerical_data)

        print(f"Prediction: {prediction}")
        print(f"Probabilities: {probabilities}")

    except FileNotFoundError:
        print("Error: One or both of the specified files were not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)