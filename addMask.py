import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
mask_img = cv2.imread('medical-mask.png')
w, h , _ = mask_img.shape
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(frame_gray, face)

        bottom_mask = (landmarks.part(9).x, landmarks.part(9).y)
        top_left = (landmarks.part(1).x, landmarks.part(1).y)
        top_right = (landmarks.part(16).x, landmarks.part(16).y)

        mask_width = int(hypot(top_right[0] - top_left[0], top_right[1] - top_left[1]))
        mask_height = int(mask_width * w/h)


        mask = cv2.resize(mask_img, (mask_width, mask_height))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        _, mask_mask = cv2.threshold(mask_gray, 25,255, cv2.THRESH_BINARY_INV)

        weight = 10
        area = frame[top_left[1] - weight : top_left[1] + mask_height - weight, top_left[0] : top_left[0] + mask_width]
        area_no_mask = cv2.bitwise_and(area, area, mask = mask_mask)
        final = cv2.add(area_no_mask, mask)
        frame[top_left[1] - weight : top_left[1] + mask_height - weight, top_left[0] : top_left[0] + mask_width] = final


    cv2.imshow("mask_mask", mask_mask)
    cv2.imshow("area", area)
    cv2.imshow("area_no_mask", area_no_mask)
    cv2.imshow("final", final)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
