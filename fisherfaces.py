import cv2

width, height = 220, 220


def recognize(paths, folder_name):
    ids = []
    recognizer = cv2.face.FisherFaceRecognizer_create()

    recognizer.read(folder_name + "/fisherfaceClassifier.yml")
    for image_path in paths:
        gray_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(gray_image, (width, height))
        student_id, confidence = recognizer.predict(face_image)
        ids.append(student_id)

    return ids
