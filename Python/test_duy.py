import pickle
import sys
import time

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

import database
from PIL import Image

path = '.\ImageAttendance'
images = []
classNames = []
stime = 0
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList




encodeListKnown = findEncodings(images)
print('Encoding Complete')

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# if not cap.isOpened():
#     sys.exit('Video source not found...')


def process(img):
    name = "abc"
    global stime
    # while True:
    imgS = cv2.resize(img, (0, 0), None, fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    etime = datetime.now().strftime('%S')

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            database.addAttendanceTime(name)
        else:
            name = 'Unknown'
        #print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    # if cv2.waitKey(1) == ord('q'):
    #     break
    # while(True):
    #     cv2.imshow('Webcam', img)
    #     cv2.waitKey(1)
    #     time.sleep(2)
    #     break

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

    return name



# image_path = 'D:\\BTL_IoT\\Python\\image\\captured_face.jpg'
# image = Image.open(image_path)
#
# # Convert the image to a numpy array
# image_np = np.array(image)
#
# # Process the image using the existing function
# process(image_np)