import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(os.path.dirname(BASE_DIR), "data", "scored_tweets_final_translated_with_network_analytics.csv")
model_path = os.path.join(BASE_DIR, "model", "network_analysis_XGBoost_model.joblib")

model = joblib.load(model_path)

def get_network_analysis(user_screenname):
    """
    Performs network analysis on tweets for a specific user, using precalculated
    network metrics from a CSV, and makes predictions using a loaded XGBoost model.
    Calculates predictions based on the unique values of the required columns
    for the given user.  Handles missing columns by using the average.

    Args:
        user_screenname (str): The screen name of the user to analyze.

    Returns:
        pandas.DataFrame: A DataFrame with the analysis results, with one row
                          containing the prediction for the unique combination of
                          network metrics for the user, or None on error.
    """
    try:
        # 1. Load data from CSV
        df = pd.read_csv(csv_file_path)

        # 2. Filter by user.screen_name
        filtered_df = df[df['user.screen_name'] == user_screenname]

        # 3. Check if any data remains after filtering
        if filtered_df.empty:
            print(f"Error: No tweets found for user: {user_screenname}")
            return None

        # 4. Define required columns (XGBoost model features)
        required_columns = [
            "degree_centrality",
            "betweenness_centrality",
            "eigenvector_centrality",
            "mentions_by_others",
            "handles_mentioned",
            "frequency_change_1d",
            "frequency_change_3d",
            "retweet_count"
        ]
        print(filtered_df.columns)
        # 5. Handle missing columns
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}.  Imputing with average values.")
            for col in missing_columns:
                # Calculate the average from the original, unfiltered DataFrame
                if col in df.columns:
                    avg_value = df[col].mean()
                    filtered_df[col] = avg_value  # Assign the average to the filtered DataFrame
                else:
                    print(f"Error: Column '{col}' not found in the data.  Cannot impute.")
                    return None # return None, because a column needed for the model is not in the data.

        # 6. Get unique values of required columns
        unique_values_df = filtered_df[required_columns].drop_duplicates()

        # 7. Prepare data for prediction
        X = unique_values_df

        # 8. Make predictions
        predictions = model.predict(X)

        return predictions[0]

    except Exception as e:
        print(f"An error occurred during network analysis: {e}")
        return None
