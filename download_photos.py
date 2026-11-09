"""Download the photos of every student with a complete set (30 photos) into the dataset layout the validator reads.

    DATABASE_URL=postgresql://vision:vision@localhost:5432/vision \
    VISION_FILES_URL=http://localhost:3333/files \
    python download_photos.py --groups 35 15 5 --seed 42

Writes dataset/group_<N>/<student>/<pose>/<student>-<photo>.jpg for each group size, picking students at random,
and dataset/group_all/<student>-<photo>.jpg with every photo. Photos are biometric data: keep them out of git.
"""

import argparse
import os
import shutil
import urllib.request
from random import Random

import psycopg

COMPLETE_STUDENTS = """
    SELECT student_id FROM photos GROUP BY student_id HAVING COUNT(id) = 30 ORDER BY student_id
"""
STUDENT_PHOTOS = 'SELECT id, path, "photoType" FROM photos WHERE student_id = %s ORDER BY id'


def download(files_url, path, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with urllib.request.urlopen(f"{files_url.rstrip('/')}/{path}") as response, open(destination, "wb") as file:
        shutil.copyfileobj(response, file)


def main():
    parser = argparse.ArgumentParser(description="Download Vision photos for cross-validation")
    parser.add_argument("--groups", type=int, nargs="+", default=[35, 15, 5], help="group sizes, in students")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files_url = os.environ["VISION_FILES_URL"]
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        students = [row[0] for row in connection.execute(COMPLETE_STUDENTS)]
        photos = {student: connection.execute(STUDENT_PHOTOS, (student,)).fetchall() for student in students}

    print(f"{len(students)} alunos com 30 fotos")
    for student, rows in photos.items():
        for photo_id, path, _ in rows:
            download(files_url, path, f"dataset/group_all/{student}-{photo_id}.jpg")

    random = Random(args.seed)
    for size in args.groups:
        if size > len(students):
            print(f"Grupo de {size} ignorado: só existem {len(students)} alunos completos")
            continue
        for student in random.sample(students, size):
            for photo_id, _, pose in photos[student]:
                destination = f"dataset/group_{size}/{student}/{pose}/{student}-{photo_id}.jpg"
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copyfile(f"dataset/group_all/{student}-{photo_id}.jpg", destination)
        print(f"group_{size} pronto")


if __name__ == "__main__":
    main()
