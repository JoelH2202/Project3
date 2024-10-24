import csv
import random
import math
from collections import defaultdict
from typing import List, Tuple
import numpy as np

class DataHandler:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def read_csv(self) -> List[List[str]]:
        dataset = []
        with open(self.filepath, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header row
            for row in reader:
                dataset.append(row)
        return dataset

    def train_test_split(self, dataset: List[List[str]], test_size: float = 0.2) -> Tuple[List[List[str]], List[List[str]]]:
        if not dataset:
            raise ValueError("Dataset is empty. Cannot split into train and test sets.")

        random.shuffle(dataset)
        split_index = int(len(dataset) * (1 - test_size))
        train_set = dataset[:split_index]
        test_set = dataset[split_index:]
        return train_set, test_set

    def separate_features_labels(self, dataset: List[List[str]]) -> Tuple[List[List[float]], List[str]]:
        if not dataset:
            raise ValueError("Dataset is empty. Cannot separate features and labels.")

        features = []
        labels = []
        for row in dataset:
            features.append([float(value) for value in row[:-1]])
            labels.append(row[-1])
        return features, labels


class NaiveBayesClassifier:
    def __init__(self):
        self.prior_probabilities = {}
        self.mean_variance = {}

    def fit(self, X_train: List[List[float]], y_train: List[str]):
        self.calculate_prior_probabilities(y_train)
        self.calculate_mean_variance(X_train, y_train)

    def predict(self, X_test: List[List[float]]) -> List[str]:
        return [self.predict_single(features) for features in X_test]

    def predict_single(self, input_features: List[float]) -> str:
        probabilities = {label: math.log(prob) for label, prob in self.prior_probabilities.items()}
        for label in self.prior_probabilities:
            for i in range(len(input_features)):
                mean, variance = self.mean_variance[label][i]
                probabilities[label] += math.log(self.gaussian_probability(input_features[i], mean, variance))
        return max(probabilities, key=probabilities.get)

    def accuracy(self, y_test: List[str], y_pred: List[str]) -> float:
        correct = sum(1 for true, pred in zip(y_test, y_pred) if true == pred)
        return correct / len(y_test)

    def classification_report(self, y_test: List[str], y_pred: List[str]):
        true_positive = defaultdict(int)
        false_positive = defaultdict(int)
        false_negative = defaultdict(int)
        total_counts = defaultdict(int)
        labels = set(y_test + y_pred)

        for true, pred in zip(y_test, y_pred):
            total_counts[true] += 1
            if true == pred:
                true_positive[true] += 1
            else:
                false_positive[pred] += 1
                false_negative[true] += 1

        print("Classification Report:")
        for label in labels:
            tp = true_positive[label]
            fp = false_positive[label]
            fn = false_negative[label]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = tp / total_counts[label] if total_counts[label] > 0 else 0

            print(f"Class {label}:")
            print(f"  Precision: {precision:.2f}")
            print(f"  Recall: {recall:.2f}")
            print(f"  F1-score: {f1_score:.2f}")
            print(f"  Accuracy: {accuracy:.2f}")

    def calculate_prior_probabilities(self, y_train: List[str]):
        label_counts = defaultdict(int)
        for label in y_train:
            label_counts[label] += 1
        self.prior_probabilities = {label: count / len(y_train) for label, count in label_counts.items()}

    def calculate_mean_variance(self, X_train: List[List[float]], y_train: List[str]):
        separated_data = defaultdict(list)
        for features, label in zip(X_train, y_train):
            separated_data[label].append(features)

        self.mean_variance = {}
        for label, features_list in separated_data.items():
            features_array = np.array(features_list)
            means = np.mean(features_array, axis=0)
            variances = np.var(features_array, axis=0)
            self.mean_variance[label] = list(zip(means, variances))

    def gaussian_probability(self, x: float, mean: float, variance: float) -> float:
        exponent = math.exp(-((x - mean) ** 2 / (2 * variance)))
        return (1 / (math.sqrt(2 * math.pi * variance))) * exponent


class SVM:
    def __init__(self, learning_rate=0.001, epochs=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.weights = []
        self.bias = 0.0

    def fit(self, X: List[List[float]], y: List[str]):
        n_samples, n_features = len(X), len(X[0])
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        y_transformed = np.array([1 if label == "1" else -1 for label in y])

        for _ in range(self.epochs):
            for idx, sample in enumerate(X):
                condition = y_transformed[idx] * (np.dot(sample, self.weights) + self.bias)
                if condition >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(sample, y_transformed[idx]))
                    self.bias -= self.learning_rate * y_transformed[idx]

    def predict(self, X: List[List[float]]) -> List[str]:
        linear_output = np.dot(X, self.weights) + self.bias
        return ["1" if i >= 0 else "0" for i in linear_output]

    def accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    def classification_report(self, y_true: List[str], y_pred: List[str]):
        true_positive = defaultdict(int)
        false_positive = defaultdict(int)
        false_negative = defaultdict(int)
        total_counts = defaultdict(int)
        labels = set(y_true + y_pred)

        for true, pred in zip(y_true, y_pred):
            total_counts[true] += 1
            if true == pred:
                true_positive[true] += 1
            else:
                false_positive[pred] += 1
                false_negative[true] += 1

        print("Classification Report:")
        for label in labels:
            tp = true_positive[label]
            fp = false_positive[label]
            fn = false_negative[label]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = tp / total_counts[label] if total_counts[label] > 0 else 0

            print(f"Class {label}:")
            print(f"  Precision: {precision:.2f}")
            print(f"  Recall: {recall:.2f}")
            print(f"  F1-score: {f1_score:.2f}")
            print(f"  Accuracy: {accuracy:.2f}")


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X_train: List[List[float]], y_train: List[str]):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: List[List[float]]) -> List[str]:
        return [self.predict_single(features) for features in X_test]

    def predict_single(self, input_features: List[float]) -> str:
        distances = [(self.euclidean_distance(input_features, x), label) for x, label in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda x: x[0])
        vote_count = defaultdict(int)
        for i in range(self.k):
            vote_count[distances[i][1]] += 1
        return max(vote_count, key=vote_count.get)

    def accuracy(self, y_test: List[str], y_pred: List[str]) -> float:
        correct = sum(1 for true, pred in zip(y_test, y_pred) if true == pred)
        return correct / len(y_test)

    def classification_report(self, y_test: List[str], y_pred: List[str]):
        true_positive = defaultdict(int)
        false_positive = defaultdict(int)
        false_negative = defaultdict(int)
        total_counts = defaultdict(int)
        labels = set(y_test + y_pred)

        for true, pred in zip(y_test, y_pred):
            total_counts[true] += 1
            if true == pred:
                true_positive[true] += 1
            else:
                false_positive[pred] += 1
                false_negative[true] += 1

        print("Classification Report:")
        for label in labels:
            tp = true_positive[label]
            fp = false_positive[label]
            fn = false_negative[label]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = tp / total_counts[label] if total_counts[label] > 0 else 0

            print(f"Class {label}:")
            print(f"  Precision: {precision:.2f}")
            print(f"  Recall: {recall:.2f}")
            print(f"  F1-score: {f1_score:.2f}")
            print(f"  Accuracy: {accuracy:.2f}")

    def euclidean_distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# Main function
def main():
    data_handler = DataHandler("diabetes.csv")
    dataset = data_handler.read_csv()

    if not dataset:
        print("Dataset is empty after reading from file. Exiting.")
        return

    print("First 5 rows of the dataset:")
    for row in dataset[:5]:
        print(row)

    train_set, test_set = data_handler.train_test_split(dataset)

    train_features, train_labels = data_handler.separate_features_labels(train_set)
    test_features, test_labels = data_handler.separate_features_labels(test_set)

    choice = int(input("Choose classifier (1: Naive Bayes, 2: SVM, 3: KNN): "))

    if choice == 1:
        classifier = NaiveBayesClassifier()
        classifier.fit(train_features, train_labels)
        test_predictions = classifier.predict(test_features)
        acc = classifier.accuracy(test_labels, test_predictions)
        print(f"Accuracy: {acc:.2f}")
        classifier.classification_report(test_labels, test_predictions)
    elif choice == 2:
        classifier = SVM()
        classifier.fit(train_features, train_labels)
        test_predictions = classifier.predict(test_features)
        acc = classifier.accuracy(test_labels, test_predictions)
        print(f"Accuracy: {acc:.2f}")
        classifier.classification_report(test_labels, test_predictions)
    elif choice == 3:
        classifier = KNN(3)
        classifier.fit(train_features, train_labels)
        test_predictions = classifier.predict(test_features)
        acc = classifier.accuracy(test_labels, test_predictions)
        print(f"Accuracy: {acc:.2f}")
        classifier.classification_report(test_labels, test_predictions)
    else:
        print("Invalid choice. Exiting.")
        return

    input_features = input("Enter features to predict (comma separated): ")
    while input_features:
        features = [float(value) for value in input_features.split(',')]
        if choice == 1:
            prediction = classifier.predict_single(features)
        elif choice == 2:
            prediction = classifier.predict([features])[0]
        elif choice == 3:
            prediction = classifier.predict_single(features)
        print(f"Predicted label: {prediction}")
        input_features = input("Enter features to predict (comma separated): ")


if __name__ == "__main__":
    main()
