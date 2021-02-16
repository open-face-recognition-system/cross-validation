import os
from random import sample

from numpy import mean

from recognizer import Recognizer
from trainer import Trainer
from metrics import MetricsCalculator


class Validator:
    def __init__(self):
        self.photos_size = 10
        self.k = 5
        self.group = 30
        self.eigenfaces_metrics = []
        self.fisherfaces_metrics = []
        self.lbph_metrics = []
        self.paths = os.listdir(f"dataset/group_{self.group}")

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
        main_folder = f"dataset/group_{self.group}"

        dataset = {
            "student_ids": []
        }
        all_paths = []

        for student_id_folder in self.paths:
            student_photos = {
                "id": student_id_folder,
                "photo_types": []
            }
            for photo_type_folder in os.listdir(f"{main_folder}/{student_id_folder}"):
                photo_type_path = os.path.join(f"{main_folder}/{student_id_folder}/{photo_type_folder}")
                photos_path = []
                for photo_path in os.listdir(photo_type_path):
                    all_paths.append(f"{photo_type_path}/{photo_path}")
                    photos_path.append(f"{photo_type_path}/{photo_path}")
                photo_type = {
                    "type": photo_type_folder,
                    "photos": photos_path
                }
                student_photos["photo_types"].append(photo_type)

            dataset["student_ids"].append(student_photos)

        train_test_split = []
        samples_size = len(dataset["student_ids"])
        size = samples_size * self.photos_size
        number_of_photos = int((size / self.k) / samples_size)

        print(f"Total de fotos: {size}")
        print(f"Total de fotos por grupo: {int(size / self.k)}")

        for i in range(self.k):
            train_test_split.append([])
            for student_id in dataset["student_ids"]:
                photo_types = student_id["photo_types"]
                for index, photo_type in enumerate(photo_types):
                    photos = photo_type["photos"]
                    photo_sample = sample(photos, 1)[0]
                    photos.remove(photo_sample)
                    train_test_split[i].append(photo_sample)
                    if number_of_photos == index + 1:
                        break

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
