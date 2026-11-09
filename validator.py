import argparse
import os
from datetime import datetime
from random import Random

from numpy import mean

import recognition
from metrics import MetricsCalculator


class Validator:
    def __init__(self, test_name, group_value, photos_size, seed=None, k=5):
        self.random = Random(seed)
        self.test = test_name
        self.photos_size = photos_size
        self.group = group_value
        self.k = k
        self.paths = os.listdir(f"dataset/group_{self.group}")

    def validate(self):
        folds = self.create_kfold_from_flat_group() if self.test == "all" else self.create_kfolds_by_photos()
        metrics = {algorithm: [] for algorithm in recognition.ALGORITHMS}

        for index, test_paths in enumerate(folds):
            train_paths = [path for other, fold in enumerate(folds) if other != index for path in fold]
            expected = [recognition.student_id(path) for path in test_paths]
            print(f"Fold {index + 1}/{self.k}: {len(train_paths)} fotos de treino, {len(test_paths)} de teste")

            for algorithm in recognition.ALGORITHMS:
                predicted, _ = recognition.predict(recognition.train(algorithm, train_paths), test_paths)
                metrics[algorithm].append(MetricsCalculator(expected, test_paths, predicted).calculate_metrics())

        return metrics

    def create_kfolds_by_photos(self):
        """Pick photos_size photos per student, spread evenly across poses, and deal them over the k folds."""
        main_folder = f"dataset/group_{self.group}"
        folds = [[] for _ in range(self.k)]

        for student in sorted(self.paths):
            student_folder = f"{main_folder}/{student}"
            poses = sorted(os.listdir(student_folder))
            per_pose, extra = divmod(self.photos_size, len(poses))
            bonus = set(self.random.sample(poses, extra))

            chosen = []
            for pose in poses:
                photos = sorted(os.listdir(f"{student_folder}/{pose}"))
                amount = per_pose + (1 if pose in bonus else 0)
                chosen += [f"{student_folder}/{pose}/{photo}" for photo in self.random.sample(photos, amount)]

            self.random.shuffle(chosen)
            for index, path in enumerate(chosen):
                folds[index % self.k].append(path)

        return folds

    def create_kfold_from_flat_group(self):
        """group_all keeps every photo in one folder; shuffle them and deal them over the k folds."""
        photos = sorted(f"dataset/group_{self.group}/{name}" for name in self.paths)
        self.random.shuffle(photos)
        return [photos[index::self.k] for index in range(self.k)]


def report(test, group, photos_size, metrics):
    lines = [f"TESTE: {test}", f"GRUPO: {group} | FOTOS POR ALUNO: {photos_size}"]
    for algorithm, folds in metrics.items():
        lines += [
            algorithm,
            f"Accuracy: {mean([fold.accuracy for fold in folds]):.4f}",
            f"Precision: {mean([fold.precision for fold in folds]):.4f}",
            f"Recall: {mean([fold.recall for fold in folds]):.4f}",
        ]
    return "\n".join(lines + ["=" * 64]) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Cross-validate Vision's face recognizers")
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling photos and folds")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    output = f"results/{datetime.now():%Y-%m-%d-%H%M}-seed-{args.seed}.txt"

    scenarios = [("stratified", group, size) for group in (35, 15, 5) for size in (30, 15, 10, 5)]
    scenarios += [("all", "all", 30)] + [("mask", "mask", size) for size in (30, 15, 10, 5)]

    with open(output, "w") as results:
        results.write(f"SEED: {args.seed}\n")
        for test, group, photos_size in scenarios:
            print(f"== {test} | grupo {group} | {photos_size} fotos por aluno")
            metrics = Validator(test, group, photos_size, seed=args.seed).validate()
            results.write(report(test, group, photos_size, metrics))
            results.flush()

    print(f"Resultados em {output}")


if __name__ == "__main__":
    main()
