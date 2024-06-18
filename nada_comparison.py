import dlib
import cv2
import numpy as np

def get_facial_points(image_path=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    if image_path:
        img = cv2.imread(image_path)
    else:
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        cap.release()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    points = []
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))

    return points

if __name__ == "__main__":
    points = get_facial_points("path_to_image.jpg")
    print(points)
