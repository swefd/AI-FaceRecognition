import cv2

cap = cv2.VideoCapture(0)

names = {}

file = open("config/name_list.txt", 'r')
lines = file.readlines()
file.close()

for line in lines:
    key, value = line.strip().split("=")
    names[key] = value

face_cascade_name = 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_name)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/trainer2.yml")

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y: y + h, x:x + w])

        label = ""

        if confidence < 50:
            label = "Name: {0} Confidence {1}%".format(names[str(id)], round(100 - confidence))
        else:
            label = "UNKNOWN"

        cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Recognize", img)
    if cv2.waitKey(1) == 27:
        break
