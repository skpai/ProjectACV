#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:15:13 2021

@author: spillai
"""

import cv2
import mediapipe as mp
import numpy as np

font                   = cv2.FONT_HERSHEY_SIMPLEX
pos = (25,100)
fontScale              = 2
fontColor              = (0,255,0)
lineType               = 2

cap = cv2.VideoCapture(0)

def cv2flip(img, key=0):
    return cv2.flip(img, key)

sepiakernel=np.array([[0.272,0.534,0.131],
                      [0.0349, 0.686,0.168],
                      [0.393,0.769,0.189]])
#with numpy
def imagemirror(img):
    width = img.shape[1]
    img2=img.copy()
    img2[:, :width // 2,:]=img[:, width // 2:,:][:,::-1,:]
    return img2
def grayfilter(img):
    grayimg= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grayimg, cv2.COLOR_GRAY2RGB)

def invfilter(img):
    return cv2.bitwise_not(img)

def gaussianblur(img):
    return cv2.GaussianBlur(img, (5,5),0)
def cannyfilter(img):
    cannyimg= cv2.Canny(img, 100,100)
    return cv2.cvtColor(cannyimg, cv2.COLOR_GRAY2RGB)

def sepiafilter(img):
    return cv2.filter2D(img,-1, sepiakernel)

def addtext(image, title, pos):
    cv2.putText(image,title,
    pos,
    font,
    fontScale,
    fontColor,
    lineType)
    return image

while cap.isOpened():
    success, image = cap.read()
    img1=image
    row1=np.concatenate((grayfilter(img1), gaussianblur(img1), invfilter(img1)), axis=1)
    row2=np.concatenate((cannyfilter(img1), img1, sepiafilter(img1)), axis=1)
    row3=np.concatenate((imagemirror(img1), cv2flip(img1, key=1), cv2flip(img1, key=0)), axis=1)
    combined=np.concatenate((row1, row2, row3), axis=0)
    h,w,_=image.shape

    for i in range(3):
        for j in range(3):
            addtext(combined,f"filter {i} {j}", (25+i*w,100+j*h))

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    combined.flags.writeable = False

    cv2.imshow('GRID', combined)
    #cv2.destroyAllWindows()
    # if cv2.waitKey(5) & 0xFF == 27:
    #   break
    while(True):
        k = cv2.waitKey(33)
        if k == -1:  # if no key was pressed, -1 is returned
            continue
        else:
            break

cap.release()
cv2.destroyWindow('GRID')
