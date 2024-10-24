from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import pickle
import random

app = Flask(__name__)

# Load the trained model and data
with open("data.pickle", "rb") as f:
    words, labels, _, _ = pickle.load(f)

model = tf.keras.models.load_model("chatbot_model.h5")

with open("college_data.json", "r") as file:
    intents = json.load(file)

stemmer = LancasterStemmer()

def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def classify(sentence):
    bag = preprocess_input(sentence)
    try:
        results = model.predict(np.array([bag]))[0]
        max_index = np.argmax(results)
        tag = labels[max_index]
        return tag
    except Exception as e:
        print(f"Error classifying input: {e}")
        return "error"

def get_response(tag):
    if tag == "error":
        return "I'm having trouble processing your request. Please try again later."
    
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# New route to serve the FAQ data
@app.route('/get_faqs', methods=['GET'])
def get_faqs():
    try:
        faq_questions = []
        for intent in intents['intents']:
            faq_questions.extend(intent['patterns'])
        return jsonify(faq_questions)
    except Exception as e:
        print(f"Error retrieving FAQs: {e}")
        return jsonify([])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'response': "Please provide a message."})

        tag = classify(user_message)
        response = get_response(tag)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error processing response: {e}")
        return jsonify({'response': "I'm having trouble processing your request. Please try again later."})

if __name__ == '__main__':
    app.run(debug=True)
