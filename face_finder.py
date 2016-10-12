"""Playing around with opencv."""
from __future__ import print_function
import cv2


def display_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def main():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread('faces.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    display_image("original", gray)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    display_image("output", image)

main()
