import cv2
import numpy as np

from collections import deque


def eq(a, b):
    x1, y1 = a
    x2, y2 = b
    if (x1-x2)**2 + (y1-y2)**2 < 30:
        return True
    else:
        return False

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)

    NUM = 4
    SIZE = 200
    cord = deque(maxlen=NUM)

    while (True):
        ret, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        img = frame
        result = np.zeros((480, 640, 3), dtype=np.uint8)
        for (x, y, w, h) in faces:
            if not any(eq((x, y), (item[0], item[1])) for item in cord) or len(cord) < NUM:
                cord.append((x, y, w, h))

        if len(cord) == NUM:
            square = cord[0]
            x, y, w, h = square
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (SIZE, SIZE))
            result[:SIZE, :SIZE] = face

            square = cord[1]
            x, y, w, h = square
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (SIZE, SIZE))
            result[-SIZE:, :SIZE, :] = face
            
            square = cord[2]
            x, y, w, h = square
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (SIZE, SIZE))
            result[:SIZE, -SIZE:, :] = face

            square = cord[3]
            x, y, w, h = square
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (SIZE, SIZE))
            result[-SIZE:, -SIZE:, :] = face

        cv2.imshow('frame2', result)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
