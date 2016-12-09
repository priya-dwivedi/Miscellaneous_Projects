# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:27:42 2016

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


img = cv2.imread(image_file) #load rgb image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv

h, s, v = cv2.split(hsv)

avg_h = np.mean(h)
print(avg_h)
avg_s = np.mean(s)
print(avg_s)
avg_v = np.mean(v)
print(avg_v)

# Avg_h of nishu - 57.0621
# Avg_s of nishu - 46.1797
# Avg_v of nishu - 193.9882
h_benchmark = 57.0621
s_benchmark = 46.1797
v_benchmark = 193.9882

fact_h = h_benchmark/avg_h
fact_s = s_benchmark/avg_s
fact_v = v_benchmark/avg_v

print(fact_h, fact_s, fact_v)

h1 = np.multiply(fact_h, h)
h1 = h1.astype(np.float)

s1 = np.multiply(fact_s, s)
s1 = s1.astype(np.float)

v1 = np.multiply(fact_v, v)
v1 = v1.astype(np.float)

final_hsv = cv2.merge((h1, s1, v1))
final_hsv = final_hsv.astype(np.uint8)

img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("image_processed.jpg", img2)


plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2,cmap = 'gray')
plt.title('processed Image'), plt.xticks([]), plt.yticks([])
plt.show()