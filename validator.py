import os
from random import sample

from numpy import mean

from recognizer import Recognizer
from trainer import Trainer
from metrics import MetricsCalculator


class Validator:
    def __init__(self, test_name, group_value, photos_size):
        self.test = test_name
        self.photos_size = photos_size
        self.group = group_value
        self.k = 5
        self.eigenfaces_metrics = []
        self.fisherfaces_metrics = []
        self.lbph_metrics = []
        self.paths = os.listdir(f"dataset/group_{self.group}")

    def validate(self):
        if self.test == "stratified":
            train_test_split = self.create_kfolds_by_photos()
        elif self.test == "all":
            train_test_split = self.create_kfold_by_students()
        else:
            train_test_split = self.create_kfolds_by_photos()

        for index, paths in enumerate(train_test_split):
            train_test_copy = train_test_split.copy()
            data_to_recognize = train_test_copy[index]
            train_test_copy.pop(index)
            print(f"Treinando grupo {index}")
            self.training_data(train_test_copy, str(index))
            print(f"Reconhecendo grupo {index}")
            self.recognize_data(data_to_recognize, str(index))

        f = open("results_final.txt", "a")

        f.write(f"TESTE: {self.test} \n")
        f.write(f"GRUPO: {self.group} | FOTOS POR ALUNO: {self.photos_size} \n")
        f.write("Eigenfaces \n")
        f.write(f"Accuracy: {mean([metric.accuracy for metric in self.eigenfaces_metrics])} \n")
        f.write(f"Precision: {mean([metric.precision for metric in self.eigenfaces_metrics])} \n")
        f.write(f"Recall: {mean([metric.recall for metric in self.eigenfaces_metrics])} \n")

        f.write("Fisherfaces \n")
        f.write(f"Accuracy: {mean([metric.accuracy for metric in self.fisherfaces_metrics])} \n")
        f.write(f"Precision: {mean([metric.precision for metric in self.fisherfaces_metrics])} \n")
        f.write(f"Recall: {mean([metric.recall for metric in self.fisherfaces_metrics])} \n")

        f.write("LBPH \n")
        f.write(f"Accuracy: {mean([metric.accuracy for metric in self.lbph_metrics])} \n")
        f.write(f"Precision: {mean([metric.precision for metric in self.lbph_metrics])} \n")
        f.write(f"Recall: {mean([metric.recall for metric in self.lbph_metrics])} \n")
        f.write(f"================================================================\n")
        f.close()

    def create_kfolds_by_photos(self):
        print("Separando os grupos por fotos")
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

    def create_kfolds_for_mask_group(self):
        print("Separando os grupos por fotos")
        main_folder = f"dataset/group_mask"

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

    def create_kfold_by_students(self):
        print("Separando os grupos por estudante")

        dataset = []
        paths = os.listdir(f"dataset/group_{self.group}")

        for filename in paths:
            dataset.append(f"dataset/group_{self.group}/{filename}")

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
        photos_ids = []
        for photos_path in group:
            paths_to_recognize.append(photos_path)
            student_id = int(os.path.split(photos_path)[-1].split("-")[0])
            photo_id = int(os.path.split(photos_path)[-1].split("-")[1].split(".")[0])
            y_true.append(student_id)
            photos_ids.append(photo_id)

        recognizer = Recognizer(paths_to_recognize, classifier_path)

        eigenfaces_y_pred = recognizer.eigenfaces()
        eigenfaces_metrics = MetricsCalculator(y_true, photos_ids, eigenfaces_y_pred)
        eigenfaces_metrics.calculate_metrics()
        eigenfaces_metrics.print_metrics()
        self.eigenfaces_metrics.append(eigenfaces_metrics)

        fisherfaces_y_pred = recognizer.fisherfaces()
        fisherfaces_metrics = MetricsCalculator(y_true, photos_ids, fisherfaces_y_pred)
        fisherfaces_metrics.calculate_metrics()
        fisherfaces_metrics.print_metrics()
        self.fisherfaces_metrics.append(fisherfaces_metrics)

        lbph_y_pred = recognizer.lbph()
        lbph_metrics = MetricsCalculator(y_true, photos_ids, lbph_y_pred)
        lbph_metrics.calculate_metrics()
        lbph_metrics.print_metrics()
        self.lbph_metrics.append(lbph_metrics)


if __name__ == '__main__':
    groups = [35, 15, 5]
    number_of_photos_size = [30, 15, 10, 5]

    for group in groups:
        for student_photos_size in number_of_photos_size:
            validator = Validator("stratified", group, student_photos_size)
            validator.validate()

    validator = Validator("all", "all", 30)
    validator.validate()

    for student_photos_size in number_of_photos_size:
        validator = Validator("mask", "mask", student_photos_size)
        validator.validate()
