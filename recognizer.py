import cv2


class Recognizer:
    def __init__(self, paths_to_recognize, classifier_path):
        self.width = 220
        self.height = 220
        self.paths_to_recognize = paths_to_recognize
        self.classifier_path = classifier_path
        self.recognizer = None

    def eigenfaces(self):
        print(f"Reconhecimento facial - Eigenfaces - {self.classifier_path}")
        ids = []
        self.recognizer = cv2.face.EigenFaceRecognizer_create()
        self.recognizer.read(f"{self.classifier_path}/eigenfaces_classifier.yml")
        for image_path in self.paths_to_recognize:
            face_image = self.path_to_image(image_path)
            student_id, confidence = self.recognizer.predict(face_image)
            ids.append(student_id)
        return ids

    def fisherfaces(self):
        print(f"Reconhecimento facial - Fisherfaces - {self.classifier_path}")
        ids = []
        self.recognizer = cv2.face.FisherFaceRecognizer_create()
        self.recognizer.read(f"{self.classifier_path}/fisherfaces_classifier.yml")
        for image_path in self.paths_to_recognize:
            face_image = self.path_to_image(image_path)
            student_id, confidence = self.recognizer.predict(face_image)
            ids.append(student_id)
        return ids

    def lbph(self):
        print(f"Reconhecimento facial - LBPH - {self.classifier_path}")
        ids = []
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(f"{self.classifier_path}/lbph_classifier.yml")
        for image_path in self.paths_to_recognize:
            face_image = self.path_to_image(image_path)
            student_id, confidence = self.recognizer.predict(face_image)
            ids.append(student_id)
        return ids

    def path_to_image(self, image_path):
        gray_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(gray_image, (self.width, self.height))
        return face_image
