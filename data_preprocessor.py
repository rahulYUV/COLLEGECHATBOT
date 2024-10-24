# data_preprocessor.py

import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.stemmer = LancasterStemmer()
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
                stemmed_tokens = [self.stemmer.stem(word.lower()) for word in tokens]
                self.words.extend(stemmed_tokens)
                self.docs_patterns.append(stemmed_tokens)
                self.docs_tags.append(intent["tag"])
                
                if intent["tag"] not in self.labels:
                    self.labels.append(intent["tag"])

        self.words = sorted(list(set([self.stemmer.stem(w.lower()) for w in self.words])))
        self.labels = sorted(self.labels)

    def create_training_data(self):
        training = []
        output = []
        empty_output_row = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(self.docs_patterns):
            bag = []
            stemmed_words = [self.stemmer.stem(w.lower()) for w in doc]
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

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    words, labels, training, output = preprocessor.preprocess("college_data.json")
    print("Preprocessing complete.")
    print(f"Number of words: {len(words)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Shape of training data: {training.shape}")
    print(f"Shape of output data: {output.shape}")