import cv2
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt

# Get the HoG face detection model.
hog_face_detector = dlib.get_frontal_face_detector()

# def detectDlib(impath):
#     frame = cv2.imread(impath)
#     frame = cv2.resize(frame, (512,512))
#     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
#     face_detect = dlib.get_frontal_face_detector()
#     rects = face_detect(gray, 1)
#     for (i, rect) in enumerate(rects):
#         (x, y, w, h) = face_utils.rect_to_bb(rect) # rectangle to bounding box
#
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     # plt.imshow(frame)
#     # plt.show()
#     cv2.imshow('face', frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     detectDlib('faces/dark.jpeg')
#     detectDlib('faces/angry.jpg')
#     detectDlib('faces/occ.jpg')
#     detectDlib('faces/occ2.jpg')
#
      #detectDlib('faces/Elon.jpg')
      # detectDlib('D:/Bachelor/faces/steve jobs.jpg')
      # detectDlib('D:/Bachelor/faces/multiple.jpeg')
      # detectDlib('D:/Bachelor/lfw/Aaron_Eckhart_0001.jpg')

#     detectDlib('faces/henedy.jpg')

camera = cv2.VideoCapture(0)

while True:
    _, fram = camera.read()
    frame = cv2.flip(fram, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray, 1)
    for (i, rect) in enumerate(rects):
        (x, y, width, height) = face_utils.rect_to_bb(rect) # rectangle to bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
