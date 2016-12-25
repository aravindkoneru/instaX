"""Playing around with opencv and webcam."""
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')


def find_face(gray, original):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("found {:d} faces".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return original


def main():
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = find_face(gray, frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
