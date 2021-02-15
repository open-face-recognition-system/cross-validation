import os
from random import sample

from numpy import mean

from recognizer import Recognizer
from trainer import Trainer
from metrics import MetricsCalculator


class Validator:
    def __init__(self):
        self.k = 6
        self.photos_size = '30'
        self.eigenfaces_metrics = []
        self.fisherfaces_metrics = []
        self.lbph_metrics = []
        self.paths = [os.path.join("dataset/", i) for i in os.listdir("dataset/")]

    def validate(self):
        train_test_split = self.create_kfolds()

        for index, paths in enumerate(train_test_split):
            train_test_copy = train_test_split.copy()
            data_to_recognize = train_test_copy[index]
            train_test_copy.pop(index)
            print(f"Treinando grupo {index}")
            self.training_data(train_test_copy, str(index))
            print(f"Reconhecendo grupo {index}")
            self.recognize_data(data_to_recognize, str(index))

        print("Eigenfaces")
        print(f"Accuracy: {mean([metric.accuracy for metric in self.eigenfaces_metrics])}")
        print(f"Precision: {mean([metric.precision for metric in self.eigenfaces_metrics])}")
        print(f"Recall: {mean([metric.recall for metric in self.eigenfaces_metrics])}")

        print("Fisherfaces")
        print(f"Accuracy: {mean([metric.accuracy for metric in self.fisherfaces_metrics])}")
        print(f"Precision: {mean([metric.precision for metric in self.fisherfaces_metrics])}")
        print(f"Recall: {mean([metric.recall for metric in self.fisherfaces_metrics])}")

        print("LBPH")
        print(f"Accuracy: {mean([metric.accuracy for metric in self.lbph_metrics])}")
        print(f"Precision: {mean([metric.precision for metric in self.lbph_metrics])}")
        print(f"Recall: {mean([metric.recall for metric in self.lbph_metrics])}")

    def create_kfolds(self):
        print("Separando os grupos")
        dataset = []

        for student_id_folder in self.paths:
            for photo_type_folder in os.listdir(student_id_folder):
                photo_type_path = os.path.join(f"{student_id_folder}/{photo_type_folder}")
                for photo_path in os.listdir(photo_type_path):
                    dataset.append(f"{photo_type_path}/{photo_path}")

        train_test_split = []
        size = len(dataset)
        num_of_elements = int(size / self.k)

        print(f"Total de fotos: {size}")
        print(f"Total de fotos por grupo: {num_of_elements}")

        for i in range(self.k):
            new_sample = sample(dataset, num_of_elements)
            train_test_split.append(new_sample)
            for row in new_sample:
                dataset.remove(row)
        if len(dataset) != 0:
            for rows in range(len(dataset)):
                train_test_split[rows].append(dataset[rows])
            dataset.clear()
        return train_test_split

    def training_data(self, groups, folder_name):
        classifier_path = f"classifiers/{self.photos_size}/{folder_name}"
        paths_to_recognize = []
        for group in groups:
            for paths_to_training in group:
                paths_to_recognize.append(paths_to_training)

        trainner = Trainer(paths_to_recognize, classifier_path)
        trainner.training()

    def recognize_data(self, group, folder_name):
        classifier_path = f"classifiers/{self.photos_size}/{folder_name}"
        paths_to_recognize = []
        y_true = []
        for photos_path in group:
            paths_to_recognize.append(photos_path)
            student_id = int(os.path.split(photos_path)[-1].split("-")[0])
            y_true.append(student_id)

        recognizer = Recognizer(paths_to_recognize, classifier_path)

        eigenfaces_y_pred = recognizer.eigenfaces()
        eigenfaces_metrics = MetricsCalculator(y_true, eigenfaces_y_pred)
        eigenfaces_metrics.calculate_metrics()
        eigenfaces_metrics.print_metrics()
        self.eigenfaces_metrics.append(eigenfaces_metrics)

        fisherfaces_y_pred = recognizer.fisherfaces()
        fisherfaces_metrics = MetricsCalculator(y_true, fisherfaces_y_pred)
        fisherfaces_metrics.calculate_metrics()
        fisherfaces_metrics.print_metrics()
        self.fisherfaces_metrics.append(fisherfaces_metrics)

        lbph_y_pred = recognizer.lbph()
        lbph_metrics = MetricsCalculator(y_true, lbph_y_pred)
        lbph_metrics.calculate_metrics()
        lbph_metrics.print_metrics()
        self.lbph_metrics.append(lbph_metrics)


if __name__ == '__main__':
    validator = Validator()
    validator.validate()
