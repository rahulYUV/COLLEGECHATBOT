# chatbot_app.py

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import random
import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr

                                        # # Download necessary NLTK data
                                        # def download_nltk_data():
                                        #     try:
                                        #         nltk.data.find('tokenizers/punkt')
                                        #     except LookupError:
                                        #         print("Downloading necessary NLTK data...")
                                        #         nltk.download('punkt')

                                        # download_nltk_data()
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading necessary NLTK data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
    print("NLTK data check complete.")

# Call this function at the start of your script


stemmer = LancasterStemmer()

class DataPreprocessor:
    def __init__(self):
        self.words = []
        self.labels = []
        self.docs_patterns = []
        self.docs_tags = []

    def load_data(self, filename):
        with open(filename, "r") as file:
            self.data = json.load(file)

    def tokenize_and_stem(self):
        for intent in self.data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                stemmed_tokens = [stemmer.stem(word.lower()) for word in tokens]
                self.words.extend(stemmed_tokens)
                self.docs_patterns.append(stemmed_tokens)
                self.docs_tags.append(intent["tag"])
                
                if intent["tag"] not in self.labels:
                    self.labels.append(intent["tag"])

        self.words = sorted(list(set([stemmer.stem(w.lower()) for w in self.words])))
        self.labels = sorted(self.labels)

    def create_training_data(self):
        training = []
        output = []
        empty_output_row = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(self.docs_patterns):
            bag = []
            stemmed_words = [stemmer.stem(w.lower()) for w in doc]
            for w in self.words:
                bag.append(1) if w in stemmed_words else bag.append(0)
            
            output_row = empty_output_row[:]
            output_row[self.labels.index(self.docs_tags[x])] = 1
            training.append(bag)
            output.append(output_row)

        return np.array(training), np.array(output)

    def preprocess(self, filename):
        self.load_data(filename)
        self.tokenize_and_stem()
        training, output = self.create_training_data()
        return self.words, self.labels, training, output

def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_shape, activation='softmax')
    ])
    
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, training_data, output_data):
    model.fit(training_data, output_data, epochs=100, batch_size=8, verbose=1)
    model.save("chatbot_model.h5")
    print("Model training complete and saved as 'chatbot_model.h5'.")

def preprocess_input(sentence, words):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def classify(sentence, words, labels):
    model = tf.keras.models.load_model("chatbot_model.h5")
    bag = preprocess_input(sentence, words)
    results = model.predict(np.array([bag]))[0]
    max_index = np.argmax(results)
    tag = labels[max_index]
    return tag

def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Speak now.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an error with the speech recognition service.")
            return None

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot")
        master.geometry("400x500")

        self.chat_display = scrolledtext.ScrolledText(master, state='disabled')
        self.chat_display.pack(expand=True, fill='both')

        self.msg_entry = tk.Entry(master)
        self.msg_entry.pack(fill='x', pady=10)
        self.msg_entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack()

        self.voice_button = tk.Button(master, text="Voice Input", command=self.voice_input)
        self.voice_button.pack()

        # Load the trained model and data
        with open("data.pickle", "rb") as f:
            self.words, self.labels, _, _ = pickle.load(f)

        with open("college_data.json", "r") as file:
            self.intents = json.load(file)

    def send_message(self, event=None):
        msg = self.msg_entry.get()
        self.msg_entry.delete(0, 'end')

        if msg != '':
            self.chat_display.config(state='normal')
            self.chat_display.insert('end', "You: " + msg + '\n')
            
            # Get chatbot response
            tag = classify(msg, self.words, self.labels)
            response = get_response(tag, self.intents)
            
            self.chat_display.insert('end', "Chatbot: " + response + '\n\n')
            self.chat_display.config(state='disabled')
            self.chat_display.yview('end')

    def voice_input(self):
        text = speech_to_text()
        if text:
            self.msg_entry.insert(0, text)
            self.send_message()

def main():
    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor()

    # Preprocess the data
    words, labels, training, output = preprocessor.preprocess("college_data.json")

    # Save processed data to pickle file
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    # Build and train the model
    model = build_model(len(training[0]), len(output[0]))
    train_model(model, training, output)

    # Start GUI
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()