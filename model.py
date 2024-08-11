import cv2

def getFaceCascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getRecognitionModel():
    return cv2.face.LBPHFaceRecognizer_create()

#later
def getDNNFaceDetector():
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net