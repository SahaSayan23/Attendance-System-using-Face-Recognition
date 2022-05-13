import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
personNames = []
imageList = os.listdir(path)
print(imageList)

for image in imageList:
    current_img = cv2.imread(f'{path}/{image}')
    images.append(current_img)
    personNames.append(os.path.splitext(image)[0])
print(personNames)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeKnown = faceEncodings(images)
print("All Encodings Complete!!!")

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            timeStr = time_now.strftime('%H:%M:%S')
            dateStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name}, {timeStr}, {dateStr}')

cap = cv2.VideoCapture(0)
while True:
    done, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    current_face = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, current_face)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, current_face):
        matches = face_recognition.compare_faces(encodeKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            attendance(name)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()

