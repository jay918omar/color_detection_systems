# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:43:27 2020

@author: JAY KISHAN OMAR
"""


""" THIS CODE CAN DETECT FOUR COLORS (RED, BLUE, GREEN, YELLOW)
AT A TIME AND CAN ALSO TRACK THEM WITH INDICATION USING A RECTANGLE 
BOX WHERE THESE COLORS ARE PRESENT IN THE SCREEN """


import cv2
import numpy as np
import pandas as pd


video_capture = cv2.VideoCapture(0)

while(True):
    
    _, imageFrame = video_capture.read()
    
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    
    
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    
    kernel = np.ones((5,5), "uint8")
    
    
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask)
    
    
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask)
    
    
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask)
    
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask = yellow_mask)
    
    
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, 
                                        (x, y),
                                        (x+w, y+h),
                                        (0, 0, 255), 2)
            
            cv2.putText(imageFrame, "RED_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
            
    contours, hierarchy = cv2.findContours(blue_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, 
                                        (x, y),
                                        (x+w, y+h),
                                        (255, 0, 0), 2)
            
            cv2.putText(imageFrame, "BLUE_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 0, 0))
    
    contours, hierarchy = cv2.findContours(green_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, 
                                        (x, y),
                                        (x+w, y+h),
                                        (0, 255, 0), 2)
            
            cv2.putText(imageFrame, "GREEN_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0))
    
    contours, hierarchy = cv2.findContours(yellow_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, 
                                        (x, y),
                                        (x+w, y+h),
                                        (255, 255, 0), 2)
            
            cv2.putText(imageFrame, "YELLOW_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 0))


    cv2.imshow("COLOR DETECTION IN REAL TIME", imageFrame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()
        break
    
    
    
    
    