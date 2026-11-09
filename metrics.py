from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


class MetricsCalculator:
    def __init__(self, y_true, photo_paths, y_pred):
        self.y_true = y_true
        self.photo_paths = photo_paths
        self.y_pred = y_pred
        self.matrix = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0

    def calculate_metrics(self):
        self.matrix = confusion_matrix(self.y_true, self.y_pred)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision = precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        self.recall = recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        return self

    def errors(self):
        """(predicted, expected, photo path) for every misrecognized photo."""
        return [
            (predicted, expected, path)
            for expected, predicted, path in zip(self.y_true, self.y_pred, self.photo_paths)
            if expected != predicted
        ]
