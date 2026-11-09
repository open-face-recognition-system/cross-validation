"""Build the evaluation dataset from Labeled Faces in the Wild (LFW), a public research dataset.

    python public_dataset.py --groups 34 15 5 --seed 42

LFW has 34 people with at least 30 photos. Each keeps 30 random photos, spread over five folders so the
validator's layout fits; LFW has no pose labels, so those folders are not real poses. LFW photos come from
news pictures, with far more variation in light, pose and age than classroom photos.
"""

import argparse
import os
from random import Random

import cv2
import numpy as np

POSES = [f"pose{index}" for index in range(1, 6)]


def write_dataset(root, images, targets, groups, seed, photos_per_person=30):
    random = Random(seed)
    people = sorted(person for person, count in enumerate(np.bincount(targets)) if count >= photos_per_person)
    chosen = {person: random.sample(list(np.flatnonzero(targets == person)), photos_per_person) for person in people}

    def write(path, index):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, np.clip(images[index] * 255, 0, 255).astype(np.uint8))

    for person, indexes in chosen.items():
        for index in indexes:
            write(os.path.join(root, "dataset", "group_all", f"{person + 1}-{index}.png"), index)

    for size in groups:
        for person in random.sample(people, min(size, len(people))):
            for position, index in enumerate(chosen[person]):
                pose = POSES[position % len(POSES)]
                write(os.path.join(root, "dataset", f"group_{size}", str(person + 1), pose, f"{person + 1}-{index}.png"), index)


def main():
    from sklearn.datasets import fetch_lfw_people

    parser = argparse.ArgumentParser(description="Build the dataset from LFW")
    parser.add_argument("--groups", type=int, nargs="+", default=[34, 15, 5], help="group sizes, in people")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lfw = fetch_lfw_people(min_faces_per_person=30, resize=1.0, data_home="dataset/.lfw-cache")
    write_dataset(".", lfw.images, lfw.target, args.groups, args.seed)
    print(f"LFW: {len(lfw.target_names)} pessoas com 30 fotos ou mais, grupos {args.groups}")


if __name__ == "__main__":
    main()
