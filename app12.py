import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords

# Load the data
@st.cache_data
def load_data():
    # Replace with the correct path to your CSV file if it's not in the same folder
    data = pd.read_csv("spam.csv", encoding="latin-1")
    return data

# Preprocess the data
@st.cache_data
def preprocess_data(data):
    # Rename columns
    data = data.rename(columns={"v1": "label", "v2": "message"})
    
    # Drop unwanted columns (e.g., 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4')
    data = data.drop(columns=[col for col in data.columns if "Unnamed" in col], errors='ignore')
    
    # Map labels to binary (spam=1, ham=0)
    data["label"] = data["label"].map({"spam": 1, "ham": 0})
    
    # Check for missing data
    missing_values = data.isnull().sum()
    if missing_values.any():
        st.warning("Missing values found!")
        data = data.fillna({'message': '', 'label': 0})
    
    return data

# Clean the text data (remove punctuation, stopwords)
def clean_text(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Train a model
@st.cache_resource
def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    return model, vectorizer

# Display a message prediction with explanation
def predict_message_with_explanation(model, vectorizer, message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    prediction_proba = model.predict_proba(message_vectorized)[0]
    explanation = f"Spam Probability: {prediction_proba[1]*100:.2f}%"
    return ("Spam" if prediction == 1 else "Not Spam", explanation)

# Frontend App
st.title("SMS Spam Detection App")

# Load data and preprocess
data = load_data()
processed_data = preprocess_data(data)

# Display the dataset
if st.checkbox("Show dataset"):
    st.write(processed_data.head())

# Display dataset summary
st.write("Dataset Summary:")
st.write(processed_data.describe())
st.write(f"Spam Count: {processed_data['label'].sum()}, Ham Count: {len(processed_data) - processed_data['label'].sum()}")

# Visualize data distribution
fig, ax = plt.subplots()
sns.countplot(x="label", data=processed_data, ax=ax)
ax.set_title("Distribution of Spam vs. Ham Messages")
st.pyplot(fig)

# Clean text data
processed_data["message"] = processed_data["message"].apply(clean_text)

# Split data into features (X) and labels (y)
X = processed_data["message"]
y = processed_data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model, vectorizer = train_model(X_train, y_train)

# Display model accuracy
train_accuracy = model.score(vectorizer.transform(X_train), y_train)
test_accuracy = model.score(vectorizer.transform(X_test), y_test)

st.write(f"Training Accuracy: {train_accuracy * 100:.2f}%")
st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# User Input for Message Prediction
user_input = st.text_area("Enter a message to classify as spam or not:")

if user_input:
    prediction, explanation = predict_message_with_explanation(model, vectorizer, user_input)
    st.write(f"Prediction: {prediction}")
    st.write(explanation)
