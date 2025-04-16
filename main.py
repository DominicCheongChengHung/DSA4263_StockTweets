from flask import Flask, jsonify, request
import os
import sys

# Add the 'codebase' directory to the Python path so we can import the modules
CODEBASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CODEBASE_DIR)

from codebase.text_analysis import get_text_analysis
from codebase.network_analysis import get_network_analysis
app = Flask(__name__)

@app.route('/analyze_tweet', methods=['POST'])
def analyze_tweet():
    """
    Endpoint to analyze a single tweet for pump-and-dump characteristics.
    Uses the text_analysis module.
    """
    try:
        data = request.get_json()  # Get the JSON data from the request
        tweet_text = data.get('tweet_text')  # Extract the tweet text

        if not tweet_text:
            return jsonify({'error': 'tweet_text is required'}), 400  # Return a 400 Bad Request if tweet_text is missing

        prediction = get_text_analysis(tweet_text)  # Use the function from text_analysis.py
        prediction = int(prediction)
        if prediction is None:
            return jsonify({'error': 'Failed to analyze tweet'}), 500  # Handle errors from the analysis function

        return jsonify({'prediction': prediction}), 200  # Return the prediction as JSON

    except Exception as e:
        print(f"Error in /analyze_tweet: {e}")
        return jsonify({'error': f'An error occurred: {e}'}), 500  # Handle other exceptions


@app.route('/analyze_user_network', methods=['POST'])
def analyze_user_network():
    """
    Endpoint to analyze a user's network metrics.
    Uses the network_analysis module.
    """
    try:
        data = request.get_json()  # Get the JSON data from the request
        user_screenname = data.get('user_screenname')  # Extract the tweet text

        if not user_screenname:
            return jsonify({'error': 'user_screenname is required'}), 400  # Return 400 if missing

        prediction = get_network_analysis(user_screenname)  # Use the function from network_analysis.py

        if prediction is None:
            return jsonify({'error': 'Failed to analyze user network'}), 500  # Handle errors

        prediction = int(prediction)

        return jsonify({'prediction': prediction}), 200  # Return the prediction as JSON

    except Exception as e:
        print(f"Error in /analyze_user_network: {e}")
        return jsonify({'error': f'An error occurred: {e}'}), 500  # Handle exceptions


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Start the Flask development server
