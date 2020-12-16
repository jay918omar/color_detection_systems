# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:56:24 2020

@author: JAY KISHAN OMAR
"""


""" THIS CODE CAN DETECt WHAT ARE THE COLORS PRESENT IN ANY IMAGE
 FROM THE LOCAL FILES. BY THE WAY, IT CAN DETECT ABOUT 860 COLORS
IN THR IMAGE ON DOUBLE CLICKING THE IMAGE """
 
 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import pandas as pd


clicked = False
R = G = B = xpos = ypos = 0
text = ""

indexing = ["Colors", "colors_name", "hex", "R", "G", "B"]
color_csv = pd.read_csv('colors.csv', names = indexing, header = None)
all_the_colors = []


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))



def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_image2(image_path):
    image = cv2.imread(image_path)
    return image


# You have to put the path of your image as argument of get_image function.
#You have to put the path of your image as argument of get_image2 function.
picture = get_image(r"C:\Users\TARUN OMAR\Desktop\colorpic.jpg")
picture2 = get_image2(r"C:\Users\TARUN OMAR\Desktop\colorpic.jpg")


def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r, xpos, ypos
        global B,G,R
        global clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = picture2[y,x]
        B = int(b)
        G = int(g)
        R = int(r)  


def get_colors(image, number_of_colors, show_chart, show_other_chart):
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    print("The size of the modified_image is {}".format(modified_image.shape))


    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)


    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    print(hex_colors)
    minm = 1000
    for i in range(number_of_colors):
        for j in range(len(color_csv)):
            d = abs(ordered_colors[i][0]-int(color_csv.loc[j,"R"]))+abs(ordered_colors[i][1]-int(color_csv.loc[j,"G"]))+abs(ordered_colors[i][2]-int(color_csv.loc[j,"B"]))
            if(d<=minm and len(all_the_colors)<number_of_colors):
                minm = d
                color_name = color_csv.loc[j,"colors_name"]
                all_the_colors.append(color_name)
                
    print(all_the_colors)
    if (show_chart):
        plt.figure(figsize = (8,6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        
    if (show_other_chart):
        plt.figure(figsize = (8,6))
        plt.pie(counts.values(), labels = all_the_colors, colors = hex_colors)
    return rgb_colors

def get_color_on_click(R, G, B):
    minimum = 10000
    for i in range(len(color_csv)):
        d = abs(R-int(color_csv.loc[i,"R"]))+abs(G-int(color_csv.loc[i,"G"]))+abs(B-int(color_csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            color_name = color_csv.loc[i,"colors_name"]
        
    return color_name


get_colors(picture, 10, True, True)


cv2.namedWindow('picture')
cv2.setMouseCallback('picture',draw_function)
                   

while (1):
    cv2.imshow("picture", picture2)
    if (clicked): 
       cv2.rectangle(picture2,(20,20), (750,60), (B,G,R), -1)
       text = get_color_on_click(R, G, B)+' R='+ str(R) +' G='+ str(G)+' B='+ str(B)
       print(R, G, B)
       cv2.putText(picture2, text, (50,50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)
       
       if (R+G+B >= 600):
           cv2.putText(picture2, text, (50,50), 2, 0.8, (0,0,0), 2, cv2.LINE_AA)
           
           
       clicked = False
       if (clicked == False):
           text = ""
       
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
       
       





