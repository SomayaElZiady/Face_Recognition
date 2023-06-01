import pathlib
import cv2
from imutils import face_utils

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))  # find faces in image data

# Real-time Detection
camera = cv2.VideoCapture(0)

while True:
    _, fram = camera.read()
    frame = cv2.flip(fram, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert image into gray scale
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,  # the higher the number, the less faces you are going to find (high criteria)
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 255), 2)  # start, end, BGR , thickness

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

# Detection with image path
def detect(impath):
    frame = cv2.imread(impath)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (512,512))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,  # the higher the number, the less faces you are going to find (high criteria)
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,width,height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 255), 2)  # start, end, BGR , thickness

    cv2.imshow('face', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# if __name__ == '__main__':
     detect('D:/Bachelor/faces/steve jobs.jpg')

impath='D:/Bachelor/lfw/Aaron_Eckhart_0001.jpg'
detect(impath)
