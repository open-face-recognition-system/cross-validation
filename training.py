import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

width, height = 220, 220


def get_image_with_id(paths):
    student_faces = []
    student_ids = []

    for image_path in paths:
        face_image = cv2.resize(cv2.imread(image_path), (width, height))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        student_id = int(os.path.split(image_path)[-1].split("-")[0])
        student_ids.append(student_id)
        student_faces.append(face_image)

    return np.array(student_ids), student_faces


def training(paths, folder_name):
    ids, faces = get_image_with_id(paths)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    eigenface.train(faces, ids)
    eigenface.write(folder_name + "/eigenClassifier.yml")

    fisherface.train(faces, ids)
    fisherface.write(folder_name + "/fisherfaceClassifier.yml")

    lbph.train(faces, ids)
    lbph.write(folder_name + "/lbphClassifier.yml")
