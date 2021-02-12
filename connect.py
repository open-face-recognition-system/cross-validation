import requests
import psycopg2
from config import config


def downloadPhotos(student_id, photo_id, photo_path):
    url = "https://ofrs-files.nyc3.digitaloceanspaces.com/photos" + photo_path
    filename = student_id + "-" + photo_id
    r = requests.get(url, allow_redirects=True)
    open("dataset/tmp/"+filename + ".png", 'wb').write(r.content)


def findAllStudentPhotos(cur, student_id):
    cur.execute(
        'SELECT photos.id, photos.path FROM photos WHERE student_id = ' + student_id)
    photos = cur.fetchall()
    for photo in photos:
        downloadPhotos(student_id, str(photo[0]), photo[1])


def findAllStudents(cur):
    cur.execute(
        "SELECT students.id FROM students")
    students = cur.fetchall()

    for student in students:
        findAllStudentPhotos(cur, str(student[0]))


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        params = config()

        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        findAllStudents(cur)

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


if __name__ == '__main__':
    connect()
