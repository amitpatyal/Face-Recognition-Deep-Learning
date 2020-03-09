import pandas as pd
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
modelFaceRecognizer = cv2.face.LBPHFaceRecognizer_create()

def cvtColorgray_image(image):
    return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

def videoCaptureConnection():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(3, 640)  # set Width
    capture.set(4, 480)  # set Height
    return capture

def videoCaptureReleaseConnection(capture):
    capture.release()
    cv2.destroyAllWindows()

def faceCascadeClassifierDetectMultiScale(image, capture_image):
    minimum_width = int(0.1 * capture_image.get(3))
    minimum_height = int(0.1 * capture_image.get(4))
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(minimum_width, minimum_height)
    )
    return faces

def getCSVFileData():
    DataFrame = pd.read_csv('face_id_name.csv', index_col='id')
    return DataFrame.head(10000)