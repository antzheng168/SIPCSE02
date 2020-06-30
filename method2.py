# -*- coding: utf-8 -*-
"""
Created on Thu May 25 04:07:14 2020
CSE 30 Spring 2020 Program 4 starter code
@author: Fahim
"""

import numpy as np
import cv2

#Download the required files/sample videos from the Google Drive
cascade1 = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cascade2 = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cascade3 = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

cap = cv2.VideoCapture("sample.webm")
#cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    body1 = cascade1.detectMultiScale(gray, 1.3, 5)
    body2 = cascade2.detectMultiScale(gray, 1.3, 5)
    body3 = cascade3.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in body1:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in body2:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


    for (x, y, w, h) in body3:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    image = cv2.resize(img, (640,480))
    out.write(image)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()