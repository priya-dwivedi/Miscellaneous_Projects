# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:19:52 2016

@author: s6324900
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

paths = 'C:/Users/s6324900/Desktop/Deep learning/Pytesser'
filename = "priya.jpg"
#filename = "nishu.JPG"
os.chdir(paths)
image_file = os.path.join(paths, filename)

im= Image.open(image_file)
#img1 = im.point(lambda p: p * 1.0)
img1 = im.convert('L')
img1.save("grey2.jpg")
print img1.format, img1.size, img1.mode

img1 = Image.open(os.path.join(paths,"grey2.jpg"))
pixdata = img1.load()
threshold = (80)
white = (255)
(xdim, ydim) = img1.size
#If color not equal to black, please set to white
for y in range(ydim-1, 0, -1):
    for x in range(xdim):
        if pixdata[x, y] >= threshold:
            pixdata[x, y] = white
img1.save("filtered2.jpg")

filename2 = "filtered2.jpg"
img = cv2.imread(filename2,0)
edges = cv2.Canny(img,100,20)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

http://stackoverflow.com/questions/7116113/normalize-histogram-brightness-and-contrast-of-a-set-of-images-using-python-im

import operator
im= Image.open(image_file)
def equalize(im):
    h = im.convert("L").histogram()
    lut = []
    for b in range(0, len(h), 256):
        # step size
        step = reduce(operator.add, h[b:b+256]) / 255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]
    # map image through lookup table
    return im.point(lut*im.layers)
    
im.save("check.jpg")

## Check brightness of nishu.jppg

img = cv2.imread('test.jpg') #load rgb image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv

h, s, v = cv2.split(hsv)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2,cmap = 'gray')
plt.title('processed Image'), plt.xticks([]), plt.yticks([])
plt.show()
