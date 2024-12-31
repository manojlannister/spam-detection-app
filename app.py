import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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
    st.write(f"Missing values before cleaning: {missing_values}")
    
    if missing_values.any():
        st.warning("Missing values found!")
        
        # Handle missing values by filling them with empty strings for 'message' and 0 for 'label'
        data = data.fillna({'message': '', 'label': 0})
        
        # Verify if there are still missing values after filling
        missing_values_after = data.isnull().sum()
        st.write(f"Missing values after cleaning: {missing_values_after}")
    
    return data



# Train a model
@st.cache_resource
def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    return model, vectorizer

# Display a message prediction
def predict_message(model, vectorizer, message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction == 1 else "Not Spam"

# Frontend App
st.title("SMS Spam Detection App")

# Load data and preprocess
data = load_data()
processed_data = preprocess_data(data)

# Display the dataset
if st.checkbox("Show dataset"):
    st.write(processed_data.head())

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
    prediction = predict_message(model, vectorizer, user_input)
    st.write(f"Prediction: {prediction}")

