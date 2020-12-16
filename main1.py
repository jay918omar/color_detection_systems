# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:57:45 2020

@author: JAY KISHAN OMAR
"""
""" THIS CODE WILL SEARCH FOR THE IMAGES FROM AN IMAGE DIRECTORY OF
A SPECIFIC COLOR GIVEN BY THE USER AND IT WILL TAKE TIME BUT THE CODE I
IS AVAILABLE ON GITHUB, YOU CAN TRY ON YOUR SYSTEM"""


"""SO THERE ARE TOTAL FIVE TYPES OF COLOR DETECTION SYSTEM THAT I HAVE
DEVELOPED DURING THIS TASK"""

"""THANK YOU"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import pandas as pd



def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Put the path of your image directory in your local device 
image_directory = r"C:\pictures\cars"
COLORS =  {
    'GREEN': [0, 128, 0],
    'BLUE': [0, 0, 128],
    'YELLOW': [255, 255, 0]}

images = []
for file in os.listdir(image_directory):
    if not file.startswith('.'):
        images.append(get_image(os.path.join(image_directory, file)))
        

plt.figure(figsize = (20, 10))        
for i in range(len(images)):
    plt.subplot(1, len(images), i+1)
    plt.imshow(images[i])
    
    
def get_colors(image, number_of_colors, show_chart):
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


    if (show_chart):
        plt.figure(figsize = (8,6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        
    return rgb_colors
    
    
def match_image_by_color(image, color, threshold = 60, number_of_colors = 10):
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))
    
    select_image = False
    
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff<threshold):
            select_image = True
    
    return select_image



def show_selected_images(images, color, threshold, colors_to_match):
    index = 1
    
    for i in range(len(images)):
        selected = match_image_by_color(images[i], 
                                        color, 
                                        threshold, 
                                        colors_to_match)
        
        if (selected):
            plt.subplot(1, 5 ,index)
            plt.imshow(images[i])
            index = index+1
            
            
plt.figure(figsize = (20,10))
show_selected_images(images, COLORS['BLUE'], 60, 10)