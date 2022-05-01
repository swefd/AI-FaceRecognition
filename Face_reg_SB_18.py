import cv2
import os

face_id = int(input("Input ID ===> "))

if not os.path.exists("dataset/" + str(face_id)):
    os.makedirs("dataset/" + str(face_id), exist_ok=False)
else:
    print("ID already exist")
    exit(0)
