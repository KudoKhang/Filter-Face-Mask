import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
nose_image = cv2.imread("pig.png")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def addNose(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        nose_center = (landmarks.part(30).x, landmarks.part(30).y)
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        nose_width = int(hypot(right_nose[0] - left_nose[0], right_nose[1] - left_nose[1]) + 5)
        nose_height = int(nose_width * 0.777 + 5)

        # Nose position
        top_left = (int(nose_center[0] - nose_width / 2),
                             int(nose_center[1] - nose_height / 2))
        bottom_right = (int(nose_center[0] + nose_width / 2 + 10), 
                             int(nose_center[1] + nose_height / 2))

        
        # ADding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGRA2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]:top_left[1] + nose_height, top_left[0]:top_left[0] + nose_width]  # col,row
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask) # mask only area == nose_pig
        final = cv2.add(nose_area_no_nose, nose_pig)
        frame[top_left[1]:top_left[1] + nose_height, top_left[0]:top_left[0] + nose_width] = final
        
        cv2.imshow("Nose Area", nose_area)
        cv2.imshow("Nose Pig", nose_mask)
        # cv2.imshow("final", final)


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    addNose(frame)

    cv2.imshow("video",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break