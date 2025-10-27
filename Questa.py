import datetime
import os
import json
import csv
import random
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up Streamlit configuration
st.set_page_config(page_title="Questa Chatbot", page_icon="🤖")

# Step 1: Cache Heavy Resources
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
            patterns.append(pattern)

    # Vectorize and train the model
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(patterns)
    y = tags

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_val)
    print(f"Model Accuracy: {accuracy_score(y_val, y_pred):.4f}")

    return intents, vectorizer, clf

intents, vectorizer, clf = load_intents_and_train_model()

# Step 2: Chatbot Response Function
def chatbot(input_text):
    input_text_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text_vector)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses']), predicted_tag
    return "I'm sorry, I don't understand.", "unknown"

def process_user_input():
    user_input = st.session_state.get("user_input", "").strip()
    if user_input:
        # Append user message
        st.session_state["messages"].append({"user": True, "text": user_input})

        # Generate bot response
        response, predicted_tag = chatbot(user_input)
        st.session_state["messages"].append({"user": False, "text": response})

        # Log the message
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["log_buffer"].append([user_input, response, predicted_tag, timestamp])

        # Write to log file every 10 messages
        if len(st.session_state["log_buffer"]) >= 10:
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(st.session_state["log_buffer"])
            st.session_state["log_buffer"].clear()

        # Clear input box
        st.session_state["user_input"] = ""

def main():
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "log_buffer" not in st.session_state:
        st.session_state["log_buffer"] = []

    # Sidebar navigation
    st.sidebar.image("qu.png", caption="Questa Chatbot Logo", use_container_width=True)
    st.sidebar.markdown("### Navigate")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Home":
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to Questa Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Your intelligent conversational assistant. Start a chat below!</p>", unsafe_allow_html=True)

        # Display chat messages dynamically
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

        # User input
        st.text_input(
            "You:", placeholder="Type your message...", key="user_input", on_change=process_user_input
        )

    elif choice == "Conversation History":
        st.markdown("<h2 style='color: #4CAF50;'>Conversation History</h2>", unsafe_allow_html=True)
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader, None)  # Skip the header row if present
                for row in csv_reader:
                    st.markdown(f"""
                    <div style="margin-bottom: 20px; border: 1px solid #4CAF50; padding: 10px; border-radius: 5px;">
                        <p><strong>User:</strong> {row[0]}</p>
                        <p><strong>Questa:</strong> {row[1]}</p>
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
            It is powered by a Random Forest model to classify user intents and provide meaningful responses.
        </p>
        <h3 style="color: #4CAF50;">Features:</h3>
        <ul>
            <li>Handles various topics including education, health, technology, and more.</li>
            <li>Supports dynamic and interactive conversations.</li>
            <li>Logs conversation history for future reference.</li>
        </ul>
        """)

if __name__ == "__main__":
    main()