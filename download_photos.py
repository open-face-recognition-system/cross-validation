import requests
import psycopg2
import os
from config import config


def download_photos(student_id, photo_id, photo_path, photo_type):
    main_path = "dataset/group_35"
    url = "https://ofrs-files.nyc3.digitaloceanspaces.com/photos/" + photo_path
    filename = student_id + "-" + photo_id
    response = requests.get(url, allow_redirects=True)

    if not os.path.exists(f"{main_path}/{student_id}"):
        os.makedirs(f"{main_path}/{student_id}")

    if not os.path.exists(f"{main_path}/{student_id}/{photo_type}"):
        os.makedirs(f"{main_path}/{student_id}/{photo_type}")
    open(f"{main_path}/{student_id}/{photo_type}/{filename}.jpg", 'wb').write(response.content)


def download_localhost_photos(student_id, photo_id, photo_path, photo_type):
    main_path = "dataset"
    url = "http://localhost/files/" + photo_path
    filename = student_id + "-" + photo_id
    response = requests.get(url, allow_redirects=True)

    if not os.path.exists(f"{main_path}/{student_id}"):
        os.makedirs(f"{main_path}/{student_id}")

    if not os.path.exists(f"{main_path}/{student_id}/{photo_type}"):
        os.makedirs(f"{main_path}/{student_id}/{photo_type}")
    open(f"{main_path}/{student_id}/{photo_type}/{filename}.jpg", 'wb').write(response.content)


def download_all_photos(student_id, photo_id, photo_path):
    main_path = "dataset/group_all/"
    url = "https://ofrs-files.nyc3.digitaloceanspaces.com/photos/" + photo_path
    filename = student_id + "-" + photo_id
    response = requests.get(url, allow_redirects=True)

    open(f"{main_path}/{filename}.jpg", 'wb').write(response.content)


def find_all_student_photos(cur, student_id):
    cur.execute(
        'SELECT photos.id, photos.path, photos."photoType" FROM photos WHERE student_id = ' + student_id)
    photos = cur.fetchall()
    print("Download group_all photos from student: " + str(student_id))
    for photo in photos:
        download_photos(student_id, str(photo[0]), photo[1], photo[2])
        download_all_photos(student_id, str(photo[0]), photo[1])


def find_all_students(cur):
    cur.execute(
        "SELECT students.id FROM users INNER JOIN students ON users.id = students.user_id LEFT OUTER JOIN photos on "
        "students.id = photos.student_id GROUP BY students.id HAVING COUNT(photos.id) = 30")
    students = cur.fetchall()

    for student in students:
        find_all_student_photos(cur, str(student[0]))


def connect():
    conn = None
    try:
        params = config()

        print('Connecting to the Vision database...')
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        find_all_students(cur)

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


if __name__ == '__main__':
    connect()
