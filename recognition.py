"""Face recognizers under evaluation, with the same preprocessing as vision-server's opencv/recognizer.py."""

import os

import cv2
import numpy as np

FACE_SIZE = (220, 220)

ALGORITHMS = {
    "eigenfaces": cv2.face.EigenFaceRecognizer_create,
    "fisherfaces": cv2.face.FisherFaceRecognizer_create,
    "lbph": cv2.face.LBPHFaceRecognizer_create,
}


def student_id(path):
    """Dataset files are named <student id>-<photo id>.<ext>."""
    return int(os.path.basename(path).split("-")[0])


def load_face(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return cv2.equalizeHist(cv2.resize(image, FACE_SIZE))


def train(algorithm, paths):
    model = ALGORITHMS[algorithm]()
    model.train([load_face(path) for path in paths], np.array([student_id(path) for path in paths], dtype=np.int32))
    return model


def predict(model, paths):
    """Returns the predicted student ids and their distances; lower distance means a closer match."""
    predictions = [model.predict(load_face(path)) for path in paths]
    return [label for label, _ in predictions], [distance for _, distance in predictions]
