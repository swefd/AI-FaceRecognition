import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)

face_id = 0

# if not os.path.exists("C:/Users/maxta/Desktop/MAX/Programing/PythonProject/RoboCode/My-AI-FaceRecognition/dataset/" + str(face_id)):
#     os.makedirs("C:/Users/maxta/Desktop/MAX/Programing/PythonProject/RoboCode/My-AI-FaceRecognition/dataset/" + str(face_id), exist_ok = False)
# else:
#     print("ID alredy exist")

count = 0
while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        cv2.imwrite("dataset/" + str(face_id) + '/' + str(count) + ".jpeg", roi_gray)
        count += 1
        print(count)

        cv2.putText(image, "Count" + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Cam", image)

    if cv2.waitKey(1) == 27:
        break
    elif count >= 200:
        print("OK")
        break
