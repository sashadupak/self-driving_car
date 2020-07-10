#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
import os
import math
from DeepPyCar import *

video = 'video/IMG_1540.mov'

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print("no video")
    exit()
print("Start")

hsv = None
pixel = (20,60,80)
lower = upper = None
height, width = 480, 640

k = 0.85  # higher k - smoothier control
low_pass = width/2
collect = 1
pause=False

def pick_color():
    h_u = s_u = v_u = 0
    h_l = s_l = v_l = 255
    x, y = int(width/2), height-10

    for i in range(-10, 10):
        for j in range(-10, 10):
            if (0<=y+j<=height) and (0<=x+i<=width):
                pixel = hsv[y+j,x+i]
                h, s, v = pixel[0], pixel[1], pixel[2]
                if h>=h_u: h_u=h
                if h<=h_l: h_l=h
                if s>=s_u: s_u=s
                if s<=s_l: s_l=s
                if v>=v_u: v_u=v
                if v<=v_l: v_l=v
    upper =  np.array([h_u + 10, s_u + 10, v_u + 40])
    lower =  np.array([h_l - 10, s_l - 10, v_l - 40])
    return lower, upper

while True:
    if pause==False:
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (640, 480))
        image = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if lower is None:
        lower, upper = pick_color()
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(mask, 300, 400)

    mask_edges = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)
    cv2.fillPoly(mask_edges, polygon, 255)
    edges_cropped = cv2.bitwise_and(edges, mask_edges)

    contours, h = cv2.findContours(edges_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv2.arcLength(x, False), reverse=True)
    #cv2.drawContours(image, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, h, 1)
    center = 0
    n = 0
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 300:
            continue
        n += 1
        if n > 2:
            break
        cnt = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, False), True)
        #cv2.drawContours(image, [cnt], -1, (0, 0, 255), 3)
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        x1 = int((-y*vx/vy)+x)
        x2 = int(((height-y)*vx/vy)+x)
        y1 = 0
        y2 = height-1
        y = int(height/2)
        x = int((x2-x1)*(y-y1)/(y2-y1)+x1)
        cv2.line(image,(x,y),(x2,y2),(0,255,0),2)
        center += x
    if n >= 2:
        center /= 2
        collect = 1
    else:
        collect += 1
        center = width/2 + math.copysign(collect, x-x2)
    low_pass = (1-k)*center + k*low_pass
    cv2.line(image,(int(width/2),height-1),(int(low_pass),int(height/2)),(0,0,255),2)

    cv2.imshow("mask", mask)
    cv2.imshow("edges", edges)
    cv2.imshow("lane lines", image)

    key = cv2.waitKey(33)
    if (key == ord("q")) or (key == 27):
        break
    if key == ord("m"):
        if pause == True:
            pause = False
        else:
            pause = True
    if (key == ord("p")):
        cv2.imwrite('test.png', frame)

cap.release()
cv2.destroyAllWindows()
print("End")