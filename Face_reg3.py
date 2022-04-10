import cv2
import os

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Cascade file not found")

cap = cv2.VideoCapture(0)

face_id = input("\n Input ID ==>")

if not os.path.exists("dataset/" + face_id):
    os.makedirs("dataset/" + face_id, exist_ok=False)
else:
    print("ID already exist")
    exit(0)

count = 0

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imwrite("dataset/" + str(face_id) + "/" + str(count) + ".jpeg",
                    gray[y:y + h, x:x + w])
        print(count)
        count += 1
    cv2.imshow("Face reg", image)
    if cv2.waitKey(10) == 27:
        break
    elif count >= 100:
        break
