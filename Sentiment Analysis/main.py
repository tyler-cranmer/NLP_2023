from typing import List
import spacy
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import math
from util import load_train_data
nlp = spacy.load('en_core_web_sm')

def split_dataset(texts, labels):
    """
    Split the dataset randomly into 80% training and 20% development set
    Make sure the splits have the same label distribution
    """
    z = zip(texts, labels)
    df = pd.DataFrame(z, columns=['review', 'label'])
    X = df['review']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return list(X_train), list(y_train), list(X_test), list(y_test)


def pre_process(text: str) -> List[str]:
    """
    remove stopwords and lemmatize and return an array of lemmas
    """
    doc = nlp(text)
    filtered_words = []
    for token in doc:
        if token.is_stop != True:
            filtered_words.append(token.lemma_)

    return filtered_words


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    tp = sum((true_labels == 1) & (predicted_labels == 1))
    tn = sum((true_labels == 0) & (predicted_labels == 0))
    fn = sum((true_labels == 1) & (predicted_labels == 0))
    fp = sum((true_labels == 0) & (predicted_labels == 1))
    return (tp + tn) / (tp + tn + fn + fp)


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    tp = sum((true_labels == 1) & (predicted_labels == 1))
    fp = sum((true_labels == 0) & (predicted_labels == 1))

    prec = tp / (tp + fp)
    return prec


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    tp = sum((true_labels == 1) & (predicted_labels == 1))
    fn = sum((true_labels == 1) & (predicted_labels == 0))

    return tp / (tp + fn)


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall
    """

    prec = precision(predicted_labels, true_labels)
    rec = recall(predicted_labels, true_labels)

    return (2 * prec * rec) / (prec + rec)

class NaiveBayesClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_word_counter = {}
        self.label_words_count = {}

    def fit(self, texts, labels):
        """
        1. Group samples by their labels
        2. Preprocess each text
        3. Count the words of the text for each label
        """
        POSITIVE_LABEL = 1
        NEGATIVE_LABEL = 0
        pos_samples = [text for text, label in zip(texts, labels) if label == POSITIVE_LABEL]
        neg_samples = [text for text, label in zip(texts, labels) if label == NEGATIVE_LABEL]
        filtered_pos_samples = pre_process(" ".join(pos_samples))
        filtered_neg_samples = pre_process(" ".join(neg_samples))
        pos_counter = dict(Counter(filtered_pos_samples))
        neg_counter = dict(Counter(filtered_neg_samples))

        self.label_word_counter[POSITIVE_LABEL] = pos_counter
        self.label_word_counter[NEGATIVE_LABEL] = neg_counter
        self.label_words_count[POSITIVE_LABEL] = len(filtered_pos_samples)
        self.label_words_count[NEGATIVE_LABEL] = len(filtered_neg_samples)

    def predict(self, texts):
        """
        1. Preprocess the texts
        2. Predict the class by using the likelihood with Bayes Method and Laplace Smoothing
        """

        # Preprocess the input texts
        filtered_texts = [pre_process(text) for text in texts]

        # Initialize a list to store the predicted labels for each text
        predictions = []

        for text in filtered_texts:
            # Initialize variables to store the log probabilities for each class
            log_probs = {}

            for label, total_word_count in self.label_words_count.items():
                log_prob = 0.0  # Initialize the log probability for the current class

                for word in text:
                    # Calculate the log probability of the word for the current class using Laplace smoothing
                    word_count_in_class = self.label_word_counter[label].get(word, 0)
                    label_word_count = len(self.label_word_counter[label])
                    word_probability = (word_count_in_class + 1) / (total_word_count + label_word_count)

                    # Add the log probability of the word to the current class's log probability
                    log_prob += math.log(word_probability)

                # Calculate the final log probability for the current class
                log_prob += math.log(self.label_words_count[label] / sum(self.label_words_count.values()))

                # Store the log probability for the current class
                log_probs[label] = log_prob

            # Determine the predicted class as the one with the highest log probability
            predicted_label = max(log_probs, key=log_probs.get)
            predictions.append(predicted_label)

        return predictions


if __name__ == "__main__":
    pos_datapath = "data/hotelPosT-train.txt"
    neg_datapath = "data/hotelNegT-train.txt"
    all_texts, all_labels = load_train_data(pos_datapath, neg_datapath)
    train_texts, train_labels, dev_texts, dev_labels = split_dataset(all_texts, all_labels)
    naive_bayes_classifier = NaiveBayesClassifier(num_classes=2)
    naive_bayes_classifier.fit(train_texts, train_labels)
    testset_predictions_nb = naive_bayes_classifier.predict(dev_texts)
    print('Naive Bayes F1:', f1_score(testset_predictions_nb, dev_labels))