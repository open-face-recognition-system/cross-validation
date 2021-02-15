from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


class MetricsCalculator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.matrix = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0

    def calculate_metrics(self):
        self.matrix = confusion_matrix(self.y_true, self.y_pred)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision = precision_score(self.y_true, self.y_pred, average='weighted')
        self.recall = recall_score(self.y_true, self.y_pred, average='weighted')

    def print_metrics(self):
        print(self.matrix)
        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
