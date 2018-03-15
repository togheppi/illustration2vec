import cv2
import os.path


def detect(filename, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if len(faces) >= 1:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        face_img = image[y:y + h, x:x + w]
        # cv2.imshow("Face", face_img)
        # cv2.waitKey(0)
    else:
        x = 0
        y = 0
        w = 0
        h = 0
        face_img = None

    return face_img, (x, y, w, h)
