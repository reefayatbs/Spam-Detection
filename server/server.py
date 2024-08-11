from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Global variables for model and transformers
__model = None
__vectorizer = None
__tfidf_transformer = None

def process_text(message):
    """Preprocess text by removing punctuation and stopwords."""
    puncs = string.punctuation
    redun = stopwords.words('english')
    words = word_tokenize(message)
    words_clean = [word for word in words if word not in puncs and word.lower() not in redun]
    return words_clean

def load_artifacts():
    """Load model, vectorizer, and tfidf transformer from disk."""
    logging.info("Loading saved artifacts...start")

    global __vectorizer
    with open("./artifacts/vectorizer.pkl", "rb") as f:
        __vectorizer = pickle.load(f)

    global __tfidf_transformer
    with open("./artifacts/tfidf_transformer.pkl", "rb") as f:
        __tfidf_transformer = pickle.load(f)

    global __model
    with open("./artifacts/model.pkl", 'rb') as f:
        __model = pickle.load(f)

    logging.info("Loading saved artifacts...done")

def isSpam(message):
    """Predict if the given message is spam."""
    sentence = [message]

    # Transform message using vectorizer and tfidf transformer
    message_bow = __vectorizer.transform(sentence)
    message_tfidf = __tfidf_transformer.transform(message_bow)

    # Predict and return result
    prediction = __model.predict(message_tfidf)
    return prediction[0] == 'spam'

@app.route('/check_spam', methods=['POST'])
def check_spam():
    """API endpoint to check if a message is spam."""
    if not request.json or 'message' not in request.json:
        return jsonify({'error': 'No message provided'}), 400
     
    message = request.json.get('message')
    
    if not isinstance(message, str):
        return jsonify({'error': 'Invalid message format'}), 400
     
    result = isSpam(message)
    return jsonify({'is_spam': result})

@app.route('/home', methods=['GET'])
def test():
    return jsonify({'model': 'Naive Bayes'}), 200


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True, port=8080)

