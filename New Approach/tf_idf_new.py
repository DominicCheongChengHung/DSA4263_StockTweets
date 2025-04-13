import pandas as pd
import re
import emoji
import joblib
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from deep_translator import GoogleTranslator
from langdetect import detect

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

    def remove_emojis(self, text):
        """Removes emojis from text."""
        return ''.join(c for c in text if c not in emoji.EMOJI_DATA)

    def translate_non_english(self, text, target_language='en'):
        """Detects non-English text and translates it using deep-translator."""
        try:
            cleaned_text = self.remove_emojis(text)
            source_language = detect(cleaned_text)
            if source_language != target_language:
                translated_text = GoogleTranslator(source='auto', target=target_language).translate(cleaned_text)
                return translated_text
            else:
                return text  # No translation needed
        except Exception as e:
            print(f"Translation Error: {e}")
            return text  # Return original text on error

    def count_dollar_capital_tickers(self, text):
        """Counts dollar signs followed by capitalized letters of length > 3."""
        return len(re.findall(r'\$([A-Z]{4,})\b', text))

    def process_text_column(self, df_1, text_column='text'):
        """Removes emojis and translates non-English text."""
        df = df_1.copy()
        df[text_column] = df[text_column].apply(self.remove_emojis)
        df[text_column] = df[text_column].apply(self.translate_non_english)
        return df

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
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'\$([A-Z]{4,})\b', 'stock', text)
            text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
            return text
        else:
            return ""

    def count_keywords(self, text):
        if isinstance(text, str):
            keywords = [
                'pump', 'dump', 'bullish', 'bearish', 'to the moon', 'short squeeze', 'hold', 'buy', 'sell',
                'rocket', 'moonshot', 'gains', 'profit', 'stock tip', 'insider info', 'guaranteed profits',
                'easy money', 'get rich quick', 'massive gains', 'skyrocket', 'explode', 'surge', 'rally',
                'squeeze', 'diamond hands', 'ape in', 'yolo', 'fomo', 'bagholder', 'bag holders',
                'squeeze play', 'market manipulation', 'hype', 'hype train', 'hot stock', 'next big thing',
                'get in now', 'before it\'s too late', 'don\'t miss out', 'limited time offer',
                'secret tip', 'insider trading', 'whale alert', 'breakout', 'uptrend', 'downtrend',
                'price target', 'PT', 'shares', 'calls', 'puts', 'options', 'leverage', 'margin',
                'DD', 'due diligence', 'fundamentals', 'technical analysis', 'TA', 'chart', 'volume',
                '10x', '100x', '1000x', 'millionaire', 'billionaire', 'tendies', 'stonks',
                'meme stock', 'penny stock', 'low float', 'OTC', 'over the counter',
                'insane gains', 'major catalyst', 'catalyst', 'breakout imminent', 'buy the dip',
                'diamond hands', 'paper hands', 'long term hold', 'squeeze it', 'squeeze em',
                'pamp it', 'pumping', 'dumping', 'bagholding', 'moon', 'rocket ship',
                'squeeze it', 'squeeze em', 'pamp it', 'pumping', 'dumping', 'bagholding', 'moon', 'rocket ship',
                'get in', 'get out', 'take profit', 'stop loss', 'limit order', 'market order', 'stop order'
            ]
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower()))
            return count
        return 0

    def count_url_mentions(self, text):
        return len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

    def predict(self, input_string, transform=True):
        """Predicts the recommendation for the given input string."""
        if type(input_string) == str:
            input_df = pd.DataFrame({'text': [input_string]})
        else:
            input_df = pd.DataFrame({'text': input_string})

        input_df['hashtag_count'] = input_df['text'].apply(self.count_hashtags)
        input_df['emoji_count'] = input_df['text'].apply(self.count_emojis)
        input_df['keyword_count'] = input_df['text'].apply(self.count_keywords)
        input_df['stock_ticker_count'] = input_df['text'].apply(self.count_dollar_capital_tickers)
        input_df['url_count'] = input_df['text'].apply(self.count_url_mentions)
        input_df = self.process_text_column(input_df)
        input_df['text'] = input_df['text'].apply(self.preprocess_text)

        tfidf_matrix = self.vectorizer.transform(input_df['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

        input_df = input_df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        input_df = pd.concat([input_df.drop('text', axis=1), tfidf_df], axis=1)

        if tfidf_df.empty or tfidf_df.isnull().values.all() or (tfidf_df.values == 0).all():
            return np.array([0])
        else:
            prediction = self.pipeline.predict(input_df)
            return prediction

    def predict_proba(self, input_string):
        """Predicts the probability of each class."""
        if type(input_string) == str:
            input_df = pd.DataFrame({'text': [input_string]})
        else:
            input_df = pd.DataFrame({'text': input_string})

        input_df['hashtag_count'] = input_df['text'].apply(self.count_hashtags)
        input_df['emoji_count'] = input_df['text'].apply(self.count_emojis)
        input_df['keyword_count'] = input_df['text'].apply(self.count_keywords)
        input_df['stock_ticker_count'] = input_df['text'].apply(self.count_dollar_capital_tickers)
        input_df['url_count'] = input_df['text'].apply(self.count_url_mentions)
        input_df = self.process_text_column(input_df)
        input_df['text'] = input_df['text'].apply(self.preprocess_text)

        tfidf_matrix = self.vectorizer.transform(input_df['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

        input_df = input_df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        input_df = pd.concat([input_df.drop('text', axis=1), tfidf_df], axis=1)

        if tfidf_df.empty or tfidf_df.isnull().values.all() or (tfidf_df.values == 0).all():
            return np.array([[1, 0]])
        else:
            prediction_proba = self.pipeline.predict_proba(input_df)
            return prediction_proba

if __name__ == "__main__":
    pipeline_path = "model/gradient_boosting_pipeline.joblib"
    vectorizer_path = "model/gradient_boosting_vectorizer.joblib"

    input_text = "To the moon, bitches #Elon #Millionaire #ROCKETROCKET"

    try:
        recommendation_system = PumpDetection(pipeline_path, vectorizer_path)
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