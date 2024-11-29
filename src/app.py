# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import re
import string
import time
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer, Random Forest model, and TF-IDF vectorizer
lemmatizer = WordNetLemmatizer()
LogisticRegressionModel = LogisticRegression(random_state=64)
vectorization = TfidfVectorizer()

def clean_text(text):
    """
    Clean and preprocess the input text.
    This function performs various text cleaning operations to prepare the text for analysis.
    """
    text = text.lower()                                                 # Converts text to lowercase
    text = re.sub(r"\[.*?\]", "", text)                                 # Removes text within square brackets
    text = re.sub(r"\W", " ", text)                                     # Removes non-word characters
    text = re.sub(r"https?://\S+\s?", "", text)                         # Removes URLs
    text = re.sub(r"<.*?>", "", text)                                   # Removes HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)    # Removes punctuation
    text = re.sub(r"\n", " ", text)                                     # Removes newline characters
    text = re.sub(r"\w*\d\w*", "", text)                                # Removes words containing numbers
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    # Removing stop words, lemmatizing and recombining the words into a single string
    return text

def load_data(fake_data_path, true_data_path):
    """
    Load and combine fake and true news datasets.
    """
    fake_data = pd.read_csv(fake_data_path)
    true_data = pd.read_csv(true_data_path)
    fake_data["class"] = 0  # Label fake news as 0
    true_data["class"] = 1  # Label true news as 1
    data = pd.concat([fake_data, true_data], axis=0)
    return data

def preprocess_data(data):
    """
    Preprocess the combined dataset.
    This function cleans the text data and splits it into training and testing sets.
    """
    df = data.drop(['title', 'subject', 'date'], axis=1)  # Remove unnecessary columns
    df = df.sample(frac=1)  # Shuffle the dataset
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    df['text'] = df['text'].apply(clean_text)  # Clean the text data
    X = df['text']
    y = df['class']
    return train_test_split(X, y, test_size=0.2)  # Split data into training and testing sets

def vectorize_data(X_train, X_test):
    """
    Convert text data to TF-IDF vectors.
    """
    X_train = vectorization.fit_transform(X_train)
    X_test = vectorization.transform(X_test)
    return X_train, X_test

def train_model(X_train, y_train):
    """
    Train the Random Forest model on the vectorized data.
    """
    LogisticRegressionModel.fit(X_train, y_train)

def fake_news_prediction(news):
    """
    Predict whether a given news article is fake or real.
    """
    news_article = {"text":[news]}
    news_df = pd.DataFrame(news_article)
    news_df["text"] = news_df["text"].apply(clean_text)  # Clean the input text
    new_X_test = news_df["text"]
    new_X_test = vectorization.transform(new_X_test)  # Vectorize the input text
    
    predictionLR = LogisticRegressionModel.predict(new_X_test)
    return predictionLR

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_news():
    news = request.form['news']
    prediction = fake_news_prediction(news)
    result = "Provided News Is Real" if prediction == 1 else "Provided News Is Fake"
    return result

if __name__ == "__main__":
    start_time = time.time()  # Start timing the execution

    # Load and preprocess the data
    data = load_data('D:\\GTU Sem-5\\MLP\\Project\\Data\\Fake.csv', 'D:\\GTU Sem-5\\MLP\\Project\\Data\\True.csv')

    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train, X_test = vectorize_data(X_train, X_test)

    # Train the model
    train_model(X_train, y_train)

    # Test the model with a sample news article
    news = "NASA has confirmed that the Earth will experience 15 days of complete darkness next month due to a rare planetary alignment. Scientists claim that Jupiter and Venus will align in such a way that their combined gravitational pull will block out the Sun's rays from reaching the Earth. Governments worldwide are urging citizens to stock up on food and supplies for this unprecedented event."
    prediction = fake_news_prediction(news)

    # Print the result
    if prediction == 1:
        print("Provided News Is Real....")
    elif prediction == 0:
        print("Provided News Is Fake")

    # Print the total execution time
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    # Run the Flask app
    app.run(debug=True)