import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
mask_img = cv2.imread('mask1.png')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img)
    for face in faces:
        landmarks = predictor(img_gray, face)
        lm_center = (landmarks.part(31).x, landmarks.part(31).y)
        lm_bot = (landmarks.part(9).x, landmarks.part(9).y)
        d = int(hypot(lm_center[0] - lm_bot[0], lm_center[1] - lm_bot[1]))

        pt1 = (lm_center[0] - d, lm_center[1] -d)
        pt2 = (lm_center[0] + d, lm_center[1] + d)

        print(lm_center, lm_bot, d, pt1, pt2)

        center_x, center_y = lm_center[0], lm_center[1]
        
        roi1 = img[pt1[0] : pt1[0] + 2*d, pt1[1] :pt1[1] + 2*d ]
        roi2 = roi1.copy()

        roi3 = roi1.copy()
        roi3 = cv2.GaussianBlur(roi3,(99,99),cv2.BORDER_TRANSPARENT)
     
        roi2= cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

        cv2.circle(roi2, (d , d ), d , (255,255,255), -1)
        _, mask = cv2.threshold(roi2, 254, 255, cv2.THRESH_BINARY)
        mask_int = cv2.bitwise_not(mask)

        m = cv2.bitwise_and(roi3, roi3, mask=mask)
        m_inv = cv2.bitwise_and(roi1, roi1, mask=mask_int)
        dst = cv2.add(m, m_inv)
        img[pt1[0] : pt1[0] + 2*d, pt1[1] :pt1[1] + 2*d ] = dst

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
    


    