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
        mask_center = (landmarks.part(67).x, landmarks.part(67).y)
        bottom_mask = (landmarks.part(9).x, landmarks.part(9).y)
        left_mask = (landmarks.part(1).x, landmarks.part(1).y)
        right_mask = (landmarks.part(16).x, landmarks.part(16).y)
        mask_width = int(hypot(right_mask[0] - left_mask[0], right_mask[1] - left_mask[1]))
        mask_height = int(mask_width * 0.71)

        print(mask_width, mask_height)
        # Mask Position
        top_left = left_mask
        bottom_right = (right_mask[0],bottom_mask[1] )
        
        # top_left = (0,0)
        # bottom_right = (170,120)

        print(left_mask, bottom_right)

        # Ading the new mask
        # cv2.rectangle(img, top_left, bottom_right, (255,255,0), 2)
        mask = cv2.resize(mask_img, (mask_width, mask_height))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        _, mask_mask = cv2.threshold(mask_gray, 25,255, cv2.THRESH_BINARY_INV)
        mask_area = img[top_left[1] - 10 :top_left[1] + mask_height - 10,top_left[0]:top_left[0] + mask_width]
        mask_area_no_mask = cv2.bitwise_and(mask_area, mask_area, mask = mask_mask)
        final = cv2.add(mask_area_no_mask, mask)
        img[top_left[1] -10 :top_left[1] + mask_height -10 ,top_left[0]:top_left[0] + mask_width] = final
       

    cv2.imshow("result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break