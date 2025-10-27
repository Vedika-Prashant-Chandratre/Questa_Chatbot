import json
import re
import os

# Step 1: Load the JSON File
def load_json(file_path):
    """Loads and validates the JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, list):
        return data  # JSON root is already a list
    elif "intents" in data and isinstance(data["intents"], list):
        return data["intents"]  # JSON root contains "intents" key
    else:
        raise TypeError("The JSON root should be a list or a dictionary with an 'intents' key.")

# Step 2: Define a Cleaning Function
def clean_text(text):
    """Cleans the input text by removing punctuation and extra spaces."""
    
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Step 3: Process Intents
def clean_intents(intents):
    """Cleans and validates intents data."""
    for intent in intents:
        # Clean and deduplicate patterns
        if 'patterns' in intent:
            intent['patterns'] = list({clean_text(pattern) for pattern in intent['patterns']})
        else:
            print(f"Warning: Intent '{intent.get('tag', 'unknown')}' is missing 'patterns' key!")

        # Clean and deduplicate responses
        if 'responses' in intent:
            intent['responses'] = list({clean_text(response) for response in intent['responses']})
        else:
            print(f"Warning: Intent '{intent.get('tag', 'unknown')}' is missing 'responses' key!")

    # Validate data
    for intent in intents:
        if not intent.get('patterns') or not intent.get('responses'):
            print(f"Warning: Intent '{intent.get('tag', 'unknown')}' has empty patterns or responses!")
    
    return intents

# Step 4: Save Cleaned Data
def save_cleaned_intents(intents, output_file_path):
    """Saves cleaned intents to a specified file."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(intents, file, indent=4)
    print(f"Cleaned intents saved to {output_file_path}")

# Main Execution
def main():
    input_file_path = os.path.abspath("./in.json")
    cleaned_file_path = r"d:\Users\DELL\Desktop\Questa chatbot\cleaned_intents.json"

    # Load, clean, and save intents
    intents = load_json(input_file_path)
    cleaned_intents = clean_intents(intents)
    save_cleaned_intents(cleaned_intents, cleaned_file_path)

if __name__ == "__main__":
    main()
