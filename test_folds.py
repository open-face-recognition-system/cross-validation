import os
import tempfile
import unittest
from collections import Counter

from validator import Validator

PHOTO_TYPES = ["normal", "smilling", "closedEyes", "rightSide", "leftSide"]


def build_dataset(root, group, students=3, photos_per_type=6):
    for student in range(1, students + 1):
        for photo_type in PHOTO_TYPES:
            folder = os.path.join(root, "dataset", f"group_{group}", str(student), photo_type)
            os.makedirs(folder)
            for n in range(photos_per_type):
                open(os.path.join(folder, f"{student}-{photo_type}{n}.jpg"), "w").close()


def student_of(path):
    return int(os.path.basename(path).split("-")[0])


def type_of(path):
    return os.path.basename(os.path.dirname(path))


class StratifiedFoldsTest(unittest.TestCase):
    def setUp(self):
        self.previous = os.getcwd()
        self.root = tempfile.mkdtemp()
        build_dataset(self.root, "test")
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.previous)

    def folds(self, photos_size, seed=7):
        return Validator("stratified", "test", photos_size, seed=seed).create_kfolds_by_photos()

    def test_uses_exactly_the_requested_photos_per_student(self):
        for photos_size in (5, 10, 15, 30):
            with self.subTest(photos_size=photos_size):
                photos = [path for fold in self.folds(photos_size) for path in fold]
                self.assertEqual({1: photos_size, 2: photos_size, 3: photos_size}, Counter(map(student_of, photos)))

    def test_spreads_each_student_photos_across_every_pose(self):
        for photos_size in (5, 10, 15, 30):
            with self.subTest(photos_size=photos_size):
                photos = [path for fold in self.folds(photos_size) for path in fold if student_of(path) == 1]
                per_type = Counter(map(type_of, photos))
                self.assertEqual(set(PHOTO_TYPES), set(per_type))
                self.assertEqual({photos_size // len(PHOTO_TYPES)}, set(per_type.values()))

    def test_every_fold_holds_the_same_share_of_each_student(self):
        for fold in self.folds(10):
            self.assertEqual({1: 2, 2: 2, 3: 2}, Counter(map(student_of, fold)))

    def test_never_repeats_a_photo(self):
        photos = [path for fold in self.folds(30) for path in fold]
        self.assertEqual(len(photos), len(set(photos)))

    def test_same_seed_gives_the_same_folds(self):
        self.assertEqual(self.folds(15, seed=1), self.folds(15, seed=1))


if __name__ == "__main__":
    unittest.main()
