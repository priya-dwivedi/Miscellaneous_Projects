# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:19:41 2016

@author: s6324900
"""
##Start with all the imports
from PIL import Image
import pytesseract
import requests
from pytesseract import *
import os
import cv2
import numpy as np

## TODO - Define directory and image file name 
paths = 'C:/Users/s6324900/Desktop/Deep learning/Pytesser'
filename = "priya.jpg"
#filename = "nishu.JPG"
os.chdir(paths)
image_file = os.path.join(paths, filename)

# Convert h,s,v of new image to the same as template

img = cv2.imread(image_file) #load rgb image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv

h, s, v = cv2.split(hsv) # Split individual components

avg_h = np.mean(h)
avg_s = np.mean(s)
avg_v = np.mean(v)
#print(avg_h, avg_s, avg_v)

h_benchmark = 57.0621
s_benchmark = 46.1797
v_benchmark = 193.9882

fact_h = h_benchmark/avg_h
fact_s = s_benchmark/avg_s
fact_v = v_benchmark/avg_v

#print(fact_h, fact_s, fact_v)

# Scale up h,s,v matrix to match template image

h1 = np.multiply(fact_h, h)
h1 = h1.astype(np.float)

s1 = np.multiply(fact_s, s)
s1 = s1.astype(np.float)

v1 = np.multiply(fact_v, v)
v1 = v1.astype(np.float)

h = h.astype(np.float)
s = s.astype(np.float)
v = v.astype(np.float)

final_hsv = cv2.merge((h, s, v))
final_hsv = final_hsv.astype(np.uint8)

img_cv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("processed.jpg", img_cv)

#convert to grayscale
im= Image.open(os.path.join(paths,"processed.jpg"))
img1 = im.convert('L')
img1.save("grey.jpg")

# Crop NAME 
width, height = img1.size
left = int(width*(0.34))
top = int(height*(0.24))
right = int(width*(0.65))
bottom = int(height*(0.35))
img2 = img1.crop((left, top, right, bottom)).save('name.jpg')

# Clean background noise on NAME and do OCR
img2 = Image.open(os.path.join(paths,"name.jpg"))
pixdata = img2.load()
threshold = (100)
white = (255)
black = (0)
(xdim, ydim) = img2.size
# If color is lower than threshold, please set to white
# Convert everything else to black
for y in range(ydim-1, 0, -1):
    for x in range(xdim):
        if pixdata[x, y] >= threshold:
            pixdata[x, y] = white
        else:
            pixdata[x, y] = black

img2.save("name-filtered.jpg")
img2.split()
text_name = pytesseract.image_to_string(img2)
#print(text)

# Crop ADDRESS
width, height = img1.size
left = int(width*(0.38))
top = int(height*(0.35))
right = int(width*(0.90))
bottom = int(height*(0.46))
img3 = img1.crop((left, top, right, bottom)).save('address.jpg')

# Clean background noise on ADDRESS and do OCR
img3 = Image.open(os.path.join(paths,"address.jpg"))
pixdata = img3.load()
threshold = (90)
white = (255)
black = (0)
(xdim, ydim) = img3.size
# If color is lower than threshold, please set to white
# Convert everything else to black
for y in range(ydim-1, 0, -1):
    for x in range(xdim):
        if pixdata[x, y] >= threshold:
            pixdata[x, y] = white
        else:
            pixdata[x, y] = black

img3.save("address-filtered.jpg")
img3.split()
text_address = pytesseract.image_to_string(img3)

# Crop DOB
width, height = img1.size
left = int(width*(0.14))
top = int(height*(0.90))
right = int(width*(0.35))
bottom = int(height*(1.00))
img4 = img1.crop((left, top, right, bottom)).save('dob.jpg')

# Clean background noise on DOB and do OCR
img4 = Image.open(os.path.join(paths,"dob.jpg"))
pixdata = img4.load()
threshold = (70)
white = (255)
black = (0)
(xdim, ydim) = img4.size
# If color is lower than threshold, please set to white
# Convert everything else to black
for y in range(ydim-1, 0, -1):
    for x in range(xdim):
        if pixdata[x, y] >= threshold:
            pixdata[x, y] = white
        else:
            pixdata[x, y] = black

img4.save("dob-filtered.jpg")
img4.split()
text_dob = pytesseract.image_to_string(img4)

print("Name is:", text_name)
print("Address is:", text_address)
print("DOB is:", text_dob)
