import os
import tempfile
import unittest

import cv2
import numpy as np

import recognition


def write_face(folder, student, photo, seed):
    pattern = np.random.default_rng(seed).integers(0, 255, (22, 22), dtype=np.uint8)
    noise = np.random.default_rng(seed * 100 + photo).integers(0, 20, (220, 220), dtype=np.uint8)
    image = cv2.add(cv2.resize(pattern, (220, 220), interpolation=cv2.INTER_NEAREST), noise)
    path = os.path.join(folder, f"{student}-{photo}.png")
    cv2.imwrite(path, image)
    return path


class RecognitionTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.mkdtemp()
        self.train = [write_face(folder, student, photo, student) for student in (11, 22) for photo in range(3)]
        self.test = [write_face(folder, student, 9, student) for student in (11, 22)]

    def test_student_id_comes_from_the_file_name(self):
        self.assertEqual(22, recognition.student_id("dataset/group_5/22/normal/22-1047.jpg"))

    def test_every_algorithm_recognizes_held_out_photos(self):
        for algorithm in recognition.ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                model = recognition.train(algorithm, self.train)
                labels, distances = recognition.predict(model, self.test)
                self.assertEqual([11, 22], labels)
                self.assertEqual(2, len(distances))

    def test_preprocessing_matches_the_server(self):
        dull = os.path.join(tempfile.mkdtemp(), "1-1.png")
        cv2.imwrite(dull, np.random.default_rng(1).integers(100, 120, (300, 400), dtype=np.uint8))

        face = recognition.load_face(dull)

        self.assertEqual((220, 220), face.shape)
        self.assertEqual((0, 255), (face.min(), face.max()))  # equalization stretches a dull image to the full range



class ValidatorTest(unittest.TestCase):
    def test_validates_every_algorithm_on_every_fold(self):
        from validator import Validator

        previous = os.getcwd()
        root = tempfile.mkdtemp()
        for student in (11, 22, 33):
            for pose in ("normal", "smilling", "closedEyes", "rightSide", "leftSide"):
                folder = os.path.join(root, "dataset", "group_test", str(student), pose)
                os.makedirs(folder)
                for photo in range(2):
                    write_face(folder, student, photo + 10 * len(pose), student)
        os.chdir(root)
        try:
            metrics = Validator("stratified", "test", 10, seed=3).validate()
        finally:
            os.chdir(previous)

        self.assertEqual(set(recognition.ALGORITHMS), set(metrics))
        for algorithm, folds in metrics.items():
            with self.subTest(algorithm=algorithm):
                self.assertEqual(5, len(folds))
                self.assertEqual(1.0, min(fold.accuracy for fold in folds))


if __name__ == "__main__":
    unittest.main()
