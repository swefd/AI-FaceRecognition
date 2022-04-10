import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/trainer.yml")

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading xml file")
    exit(0)

names = ["SAD SASA", "IRA"]

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y: y+h, x:x + w])
        label = ""
        if confidence < 50:
            label = "{0} {1}%".format(names[id], round(100 - confidence))
        else:
            label = "UNKNOWN"
        cv2.putText(img, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Recognize", img)
    cv2.waitKey(1)
