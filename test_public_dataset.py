import os
import tempfile
import unittest
from collections import Counter

import numpy as np

from public_dataset import write_dataset


class PublicDatasetTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.targets = np.repeat(np.arange(4), [40, 30, 35, 12])  # the last person has too few photos
        self.images = rng.random((len(self.targets), 12, 10), dtype=np.float32)

    def files(self, folder):
        base = os.path.join(self.root, "dataset", folder)
        return [os.path.join(path, name) for path, _, names in os.walk(base) for name in names]

    def test_keeps_exactly_thirty_photos_of_each_person_with_enough_photos(self):
        write_dataset(self.root, self.images, self.targets, groups=[2], seed=1)

        people = Counter(os.path.basename(path).split("-")[0] for path in self.files("group_all"))

        self.assertEqual({"1": 30, "2": 30, "3": 30}, people)

    def test_spreads_each_person_across_five_pose_folders(self):
        write_dataset(self.root, self.images, self.targets, groups=[3], seed=1)

        poses = Counter(os.path.basename(os.path.dirname(path)) for path in self.files("group_3/1"))

        self.assertEqual({6}, set(poses.values()))
        self.assertEqual(5, len(poses))

    def test_draws_each_group_from_the_eligible_people(self):
        write_dataset(self.root, self.images, self.targets, groups=[2], seed=1)

        people = set(os.listdir(os.path.join(self.root, "dataset", "group_2")))

        self.assertEqual(2, len(people))
        self.assertTrue(people <= {"1", "2", "3"})

    def test_writes_grayscale_images_with_the_full_range(self):
        import cv2

        write_dataset(self.root, self.images, self.targets, groups=[], seed=1)

        image = cv2.imread(self.files("group_all")[0], cv2.IMREAD_UNCHANGED)
        self.assertEqual((12, 10), image.shape)
        self.assertGreater(image.max(), 200)


if __name__ == "__main__":
    unittest.main()
