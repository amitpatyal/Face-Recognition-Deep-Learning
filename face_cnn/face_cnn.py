import os
import cv2
import dlib
import openface
import face_recognition
import pickle
import numpy as np
from helper_function import getCSVFileData, videoCaptureConnection, videoCaptureReleaseConnection


def getFaceDetectdFromDataSet():
    predictor_model = "shape_predictor_68_face_landmarks/shape_predictor_68_face_landmarks.dat"
    names = np.array(getCSVFileData())
    path = 'public_dataset'
    faceDetector = dlib.get_frontal_face_detector()
    faceAlignDlib = openface.AlignDlib(predictor_model)
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    #imageWindow = dlib.image_window()
    for imagePathFolder in imagePaths:
        imagePathFolder = [os.path.join(imagePathFolder, f) for f in os.listdir(imagePathFolder)]
        face_count = 0
        for imagePath in imagePathFolder:
            faceId = int(os.path.split(imagePath)[-1].split('_')[1])
            faceName = str(*names[int(faceId)].flatten())
            faceImage = cv2.imread(imagePath)
            detectedFaces = faceDetector(faceImage, 1)
            face_count += 1
            for i, facePosition in enumerate(detectedFaces):
                #alignedFace = faceAlignDlib.align(534, faceImage, facePosition, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                alignedFace = faceAlignDlib.align(534, faceImage, facePosition, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                face_dr = os.path.join('train_align_dataset', faceName)
                if not os.path.isdir(face_dr):
                    os.mkdir(face_dr)
                cv2.imwrite(face_dr + "/user_" + str(faceId) + '_' + str(face_count) + ".jpg", alignedFace)

    print('Colleting Samples Complete!!!')


def getTrainFaceModel():
    names = np.array(getCSVFileData())
    faceCounts = 0
    faceEncodings = []
    faceNames = []
    path = 'public_dataset'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    for imagePathF in imagePaths:
        faceCounts += 1
        imagePathF = [os.path.join(imagePathF, f) for f in os.listdir(imagePathF)]
        for imagePath in imagePathF:
            image = cv2.imread(imagePath)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fid = int(os.path.split(imagePath)[-1].split('_')[1])
            boxes = face_recognition.face_locations(imageRGB, model="cnn")
            encodedFaces = face_recognition.face_encodings(imageRGB, boxes)
            #print(*names[int(fid)].flatten(), ' Face Name')
            for encodedFace in encodedFaces:
                faceEncodings.append(encodedFace)
                faceNames.append(*names[int(fid)].flatten())
    data = {"encodings": faceEncodings, "names": faceNames}
    writeFace = open("publicFaceData.pkl", "wb")
    writeFace.write(pickle.dumps(data))
    writeFace.close()
    print("\n [INFO] {0} faces trained. Exiting Program".format(np.unique(faceCounts)))

def getFaceModelPrediction():
    faceData = pickle.loads(open('publicFaceData.pkl', 'rb').read())
    faceImage = videoCaptureConnection()
    while True:
        names = []
        ret, image = faceImage.read()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(imageRGB, model="cnn")
        encodedFaces = face_recognition.face_encodings(imageRGB, boxes)
        for encodedFace in encodedFaces:
            name = "Unknown"
            faceMatched = face_recognition.compare_faces(faceData['encodings'], encodedFace)
            if True in faceMatched:
                matchedIdxs = [i for (i, b) in enumerate(faceMatched) if b]
                counts = {}
                for i in matchedIdxs:
                    name = faceData["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)
            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, name, (left, y), cv2.FONT_ITALIC, 0.75, (225, 255, 0), 2)
        cv2.imshow("Face Recognition", image)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break
    videoCaptureReleaseConnection(faceImage)

if __name__ == '__main__':
    getFaceDetectdFromDataSet()
    getTrainFaceModel()
    getFaceModelPrediction()
