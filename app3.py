import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------------
# Load and preprocess the dataset
# -----------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    return data

@st.cache_data
def preprocess_data(data):
    data = data.rename(columns={"v1": "label", "v2": "message"})
    data = data.drop(columns=[col for col in data.columns if "Unnamed" in col], errors='ignore')
    data["label"] = data["label"].map({"spam": 1, "ham": 0})
    data = data.fillna({'message': '', 'label': 0})
    return data

# -----------------------------------
# Train Model
# -----------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    return model, vectorizer

# -----------------------------------
# Predict Message (On Button Click)
# -----------------------------------
def predict_message(model, vectorizer, message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return "🚨 Spam" if prediction == 1 else "✅ Not Spam"

# -----------------------------------
# Streamlit App
# -----------------------------------
st.set_page_config(page_title="Real-Time Spam Detection", page_icon="📩")
st.title("📩 Real-Time Spam Detection")

# Load and prepare data
data = load_data()
processed_data = preprocess_data(data)

# Split data
X = processed_data["message"]
y = processed_data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model, vectorizer = train_model(X_train, y_train)

# Accuracy Display (smaller and cleaner)
st.subheader("Model Accuracy:")
train_acc = model.score(vectorizer.transform(X_train), y_train) * 100
test_acc = model.score(vectorizer.transform(X_test), y_test) * 100
st.write(f"- **Training Accuracy:** {train_acc:.2f}%")
st.write(f"- **Testing Accuracy:** {test_acc:.2f}%")

st.divider()

# User input for message
st.subheader("Test a Message:")
user_input = st.text_area("Enter a message to check if it's spam or not:")

if st.button("🔍 Detect"):
    if user_input.strip():
        prediction = predict_message(model, vectorizer, user_input)
        st.success(f"**Prediction:** {prediction}")
    else:
        st.warning("Please enter a message first.")
