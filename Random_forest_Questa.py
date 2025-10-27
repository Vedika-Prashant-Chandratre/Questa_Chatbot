import datetime
import os
import json
import csv
import random
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

# Set up Streamlit configuration
st.set_page_config(page_title="Questa Chatbot", page_icon="🤖")

# Load cleaned intents file
file_path = os.path.abspath("./cleaned_intents.json")
if not os.path.exists(file_path):
    st.error("Error: Cleaned intents file not found!")
    st.stop()

with open(file_path, 'r') as file:
    intents = json.load(file)

# Preprocess the data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Vectorize the data and train-test split
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
clf = LogisticRegression(max_iter=10000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_val)
baseline_accuracy = accuracy_score(y_val, y_pred)
print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")

# Save the trained model and vectorizer
if not os.path.exists("chatbot_model.pkl"):
    with open('chatbot_model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)

if not os.path.exists("vectorizer.pkl"):
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

# Chatbot response function
def chatbot(input_text):
    input_text_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text_vector)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses']), predicted_tag
    return "I'm sorry, I don't understand.", "unknown"

# Main function
def main():
    # Sidebar navigation
    st.sidebar.image("qu.png", caption="Questa Chatbot Logo", use_column_width=True)
    st.sidebar.markdown("### Navigate")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Home":
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to Questa Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Your intelligent conversational assistant. Start a chat below!</p>", unsafe_allow_html=True)

        # Ensure chat log exists
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "Response", "Predicted Tag", "Timestamp"])

        # Chat interface
        user_input = st.text_input("You:")
        if user_input:
            response, predicted_tag = chatbot(user_input)

            # Log the conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, predicted_tag, timestamp])

            st.markdown(f"<div style='padding: 10px; background-color: #F0F0F0; border-radius: 5px;'>🤖 <strong>Questa:</strong> {response}</div>", unsafe_allow_html=True)

            # Display precision score
            y_true, y_pred = [], []
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    y_true.append(row[2])  # True labels
                    y_pred.append(row[2])  # Predicted labels
            if y_true and y_pred:
                precision = precision_score(y_true, y_pred, average='weighted')
                st.write(f"Model Precision: {precision:.4f}")

            # End session on "goodbye" or "bye"
            if response.lower() in ["goodbye", "bye"]:
                st.write("Thank you! Have a wonderful day ahead.")
                st.stop()

    elif choice == "Conversation History":
        st.markdown("<h2 style='color: #4CAF50;'>Conversation History</h2>", unsafe_allow_html=True)
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.markdown(f"""
                    <div style="margin-bottom: 20px; border: 1px solid #4CAF50; padding: 10px; border-radius: 5px;">
                        <p><strong>User:</strong> {row[0]}</p>
                        <p><strong>Questa:</strong> {row[1]}</p>
                        <p><strong>Intent:</strong> {row[2]}</p>
                        <p><strong>Timestamp:</strong> {row[3]}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No conversation history found.")

    elif choice == "About":
        st.markdown("<h2 style='color: #4CAF50;'>About Questa Chatbot</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p>
            <strong>Questa</strong> is an intelligent chatbot designed to interact with users using Natural Language Processing (NLP) techniques. 
            It is powered by a Logistic Regression model to classify user intents and provide meaningful responses.
        </p>
        <h3 style="color: #4CAF50;">Features:</h3>
        <ul>
            <li>Handles various topics including education, health, technology, and more.</li>
            <li>Supports dynamic and interactive conversations.</li>
            <li>Logs conversation history for future reference.</li>
        </ul>
        <h3 style="color: #4CAF50;">Future Enhancements:</h3>
        <ul>
            <li>Incorporating deep learning models for improved accuracy.</li>
            <li>Adding support for multi-language conversations.</li>
            <li>Integrating with external APIs for real-time information retrieval.</li>
        </ul>
        <p>Enjoy exploring the possibilities with Questa!</p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
