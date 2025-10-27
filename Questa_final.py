import datetime
import os
import re
import string
import json
import csv
import random
import pickle
import nltk
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Set up Streamlit configuration
st.set_page_config(page_title="Questa Chatbot", page_icon="🤖")

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}]", " ", text)  # Replace punctuation with space
    text = re.sub(r"\d+", " ", text)  # Remove digits
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return " ".join(tokens)

# Cache Heavy Resources
@st.cache_resource
def load_intents_and_train_model():
    # Load intents
    file_path = "./cleaned_intents.json"
    if not os.path.exists(file_path):
        st.error("Error: Cleaned intents file not found!")
        st.stop()

    with open(file_path, 'r') as file:
        intents = json.load(file)

    # Preprocess intents
    tags, patterns = [], []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(preprocess_text(pattern))

    # Vectorize and tune the model
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
    X = vectorizer.fit_transform(patterns)
    y = tags

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning for RandomForest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [20, 30, 50],
        'class_weight': ['balanced']
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=1)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    return intents, vectorizer, best_model

intents, vectorizer, clf = load_intents_and_train_model()

# Chatbot response function
def chatbot(input_text):
    # Preprocess user input
    processed_input = preprocess_text(input_text)
    
    # Vectorize the preprocessed input
    input_text_vector = vectorizer.transform([processed_input])
    
    # Predict intent tag
    predicted_tag = clf.predict(input_text_vector)[0]
    
    # Find response with original casing from JSON
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses']), predicted_tag
    
    return "I'm sorry, I don't understand.", "unknown"

# User input processing
def process_user_input():
    user_input = st.session_state.get("user_input", "").strip()
    if user_input:
        st.session_state["messages"].append({"user": True, "text": user_input})

        response, predicted_tag = chatbot(user_input)
        st.session_state["messages"].append({"user": False, "text": response})

        # Log the message
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = [user_input, response, predicted_tag, timestamp]
        st.session_state["log_buffer"].append(log_entry)

        log_file = "chat_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "Response", "Predicted Tag", "Timestamp"])

        with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(log_entry)

        st.session_state["user_input"] = ""

# Main function
def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "log_buffer" not in st.session_state:
        st.session_state["log_buffer"] = []

    st.sidebar.image("qu.png", caption="Questa Chatbot Logo", use_container_width=True)
    st.sidebar.markdown("### Navigate")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Home":
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to Questa Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Your intelligent conversational assistant. Start a chat below!</p>", unsafe_allow_html=True)

        for message in st.session_state["messages"]:
            alignment = "right" if message["user"] else "left"
            bg_color = "#DCF8C6" if message["user"] else "#F0F0F0"
            emoji = "🙂" if message["user"] else "🤖"
            st.markdown(f"""
            <div style="text-align: {alignment}; margin-bottom: 10px;">
                <span style="background-color: {bg_color}; padding: 10px; border-radius: 10px; max-width: 70%; display: inline-flex; align-items: center;">
                    <span style="margin-right: 5px;">{emoji}</span> {message['text']}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.text_input(
            "You:", placeholder="Type your message...", key="user_input", on_change=process_user_input
        )

    elif choice == "Conversation History":
        st.markdown("<h2 style='color: #4CAF50;'>Conversation History</h2>", unsafe_allow_html=True)
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader, None)
                for row in csv_reader:
                    st.markdown(f"""
                    <div style="margin-bottom: 20px; border: 1px solid #4CAF50; padding: 10px; border-radius: 5px;">
                        <p><strong>User:</strong> {row[0]}</p>
                        <p><strong>Questa:</strong> {row[1]}</p>
                        <p><strong>Timestamp:</strong> {row[2]}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No conversation history found.")

    elif choice == "About":
        st.markdown("<h2 style='color: #4CAF50;'>About Questa Chatbot</h2>", unsafe_allow_html=True)
        st.write("""
        Questa is an intelligent chatbot designed to interact with users using Natural Language Processing (NLP) techniques. 
        It is powered by a Random Forest model to classify user intents and provide meaningful responses.
        """)
        
if __name__ == "__main__":
    main()
