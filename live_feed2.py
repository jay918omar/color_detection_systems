# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:24:31 2020

@author: JAY KISHAN OMAR
"""

""" THIS CODE CAN DETECT ONLY ONE COLOR AT A TIME IN ONE SCREEN 
WITH BLACK COLOR IN BACKGROUND AND CAN ALSO INDICATE USING RECTANGLE WHERE THE COLOR IS PRESENT IN
THE SCREEN. WE CAN ALSO COMPARE THE OUTPUT FROM THE REAL FRAME"""

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
    
    
    #EVERY COLOR EXCEPT WHITE
    col_lower = np.array([0, 42, 0], np.uint8)
    col_upper = np.array([179, 255, 255], np.uint8)
    col_mask = cv2.inRange(hsvFrame, col_lower, col_upper)
    col_mask = cv2.dilate(col_mask, kernel)
    res_col = cv2.bitwise_and(imageFrame, imageFrame, mask = col_mask)
    
    #to tracking red_color
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            res_red = cv2.rectangle(res_red, 
                                        (x, y),
                                        (x+w, y+h),
                                        (0, 0, 255), 2)
            
            cv2.putText(res_red, "RED_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
    #to track bluw color        
    contours, hierarchy = cv2.findContours(blue_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            res_blue = cv2.rectangle(res_blue, 
                                        (x, y),
                                        (x+w, y+h),
                                        (255, 0, 0), 2)
            
            cv2.putText(res_blue, "BLUE_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 0, 0))
    
    #to track green color
    contours, hierarchy = cv2.findContours(green_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            res_green = cv2.rectangle(res_green, 
                                        (x, y),
                                        (x+w, y+h),
                                        (0, 255, 0), 2)
            
            cv2.putText(res_green, "GREEN_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0))
    
    
    #to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            res_yellow = cv2.rectangle(res_yellow, 
                                        (x, y),
                                        (x+w, y+h),
                                        (255, 255, 0), 2)
            
            cv2.putText(res_yellow, "YELLOW_COLOR",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 0))


    #cv2.imshow("DETECT RED COLOR IN REAL TIME", res_red)
    cv2.imshow("DETECT BLUE COLOR IN REAL TIME", res_blue)
    #cv2.imshow("DETECT GREEN COLOR IN REAL TIME", res_green)
    #cv2.imshow("DETECT YELLOW COLOR IN REAL TIME", res_yellow)
    #cv2.imshow("DETECTING EXCEPT WHITE COLOR IN REAL TIME", res_col)
    
    cv2.imshow("ORIGINAL FRAME", imageFrame)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()
        break