import cv2
import numpy as np
from PIL import Image
import os

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading model yml file")
    exit(0)

path = "dataset"

imagePaths = [os.path.join(path, f) for f in os.listdir(path)]


# imagePath = imagePaths[0]

def getImageAndLabels():
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        print("-----ID----->>> " + os.path.split(imagePath)[1])
        for image in os.listdir(imagePath):
            # print(image)
            image = os.path.join(imagePath, image)
            pil_img = Image.open(image).convert('L')
            img_numpy = np.array(pil_img, "uint8")

            faces = face_cascade.detectMultiScale(img_numpy)
            id = int(os.path.split(imagePath)[1])

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

    # print(faceSamples)
    # print(ids)
    return faceSamples, ids


recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, ids = getImageAndLabels()
recognizer.train(faces, np.array(ids))
if not os.path.exists("model"):
    os.makedirs("model")
recognizer.write("model/trainer2.yml")
print("{0} faces trained.".format(len(np.unique(ids))))

