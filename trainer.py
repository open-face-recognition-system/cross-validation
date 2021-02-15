import cv2
import os
import numpy as np


class Trainer:
    def __init__(self, paths_to_training, classifier_path):
        self.width = 220
        self.height = 220
        self.paths_to_training = paths_to_training
        self.classifier_path = classifier_path
        self.eigenface = cv2.face.EigenFaceRecognizer_create()
        self.fisherface = cv2.face.FisherFaceRecognizer_create()
        self.lbph = cv2.face.LBPHFaceRecognizer_create()

    def training(self):
        ids, faces = self.get_paths_with_id()

        self.create_folder_if_not_exists()

        print(f"Treinamento - Eigenfaces - {self.classifier_path}")
        self.eigenface.train(faces, ids)
        self.eigenface.write(f"{self.classifier_path}/eigenfaces_classifier.yml")

        print(f"Treinamento - Fisherfaces - {self.classifier_path}")
        self.fisherface.train(faces, ids)
        self.fisherface.write(f"{self.classifier_path}/fisherfaces_classifier.yml")

        print(f"Treinamento - LBPH - {self.classifier_path}")
        self.lbph.train(faces, ids)
        self.lbph.write(f"{self.classifier_path}/lbph_classifier.yml")

    def create_folder_if_not_exists(self):
        if not os.path.exists(self.classifier_path):
            os.makedirs(self.classifier_path)

    def get_paths_with_id(self):
        student_faces = []
        student_ids = []

        for image_path in self.paths_to_training:
            face_image = cv2.resize(cv2.imread(image_path), (self.width, self.height))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            student_id = int(os.path.split(image_path)[-1].split("-")[0])
            student_ids.append(student_id)
            student_faces.append(face_image)

        return np.array(student_ids), student_faces
