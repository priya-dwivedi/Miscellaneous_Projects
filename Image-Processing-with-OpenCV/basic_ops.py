# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:18:50 2016

@author: s6324900
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\s6324900\Desktop\Opencv image processing\Sample")
paths = "C:\Users\s6324900\Desktop\Opencv image processing\Sample"
filename = os.path.join(paths, "cat.3.jpg")
# Load an color image in grayscale
img = cv2.imread(filename,0)

#Display an image - First argument is the window name which is a string.
# You can create as many windows

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Get size of image
height, width = img.shape[:2]
print(height, width)
print(img.shape)
print(img.size, img.dtype)
## access specific values
print(img[50,100])
print(img)

## Plot an entire row of values
import matplotlib.pyplot as plt
val = img[50,:]

## Crop am image
cropped = img[50:200, :200]
cv2.imshow('image1',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()


## Look at color channels
file1 = "C:\Users\s6324900\Desktop\Opencv image processing\priyanka.jpg"
img1= cv2.imread(file1)
print(img1.shape, img1.size, img1.dtype)
## Display the red channel - 2
# b -0, g-1, r-2 
red = img1[:,:,2]
print(red)
cv2.imshow('image',red)
cv2.waitKey(0)

## scale up the brightness
img2 = (img1*(1.75)).astype(int)
print(img2)
# Orig image
cv2.imshow('image',img1)
cv2.waitKey(0)

cv2.imshow('image',img2)
cv2.waitKey(0)

## Blend my pic with a cat image
file1 = "C:\Users\s6324900\Desktop\Opencv image processing\priyanka.jpg"
file2 = "C:\Users\s6324900\Desktop\Opencv image processing\cat.3.jpg"
# Load both as grayscale
img1= cv2.imread(file1,0)
print(img1.shape, img1.size, img1.dtype)
img2 = cv2.imread(file2,0)
print(img2.shape, img2.size, img2.dtype)
# Resize the second image so both have same size
img3 = cv2.resize(img2, (400,400))
print(img3.shape, img3.size, img3.dtype)
# Display both images
cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.imshow('image',img3)
cv2.waitKey(0)

#Blend two images now
img4 = cv2.addWeighted(img1, 0.5, img3, 0.5, 0)
cv2.imshow('image',img4)
cv2.waitKey(0)

# Add guassian noise to an image
def add_guassian_noise(image):
    row,col,ch= image.shape
    mean = 0
    sigma = 2
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch).astype(np.uint8)
    noisy = cv2.add(image, gauss, dtype = -1)
    return noisy

# Load both as grayscale
img1= cv2.imread(file1)
print(img1.shape, img1.size, img1.dtype)
img1_n = add_guassian_noise(img1)

cv2.imshow('image',img1)
cv2.waitKey(0)

cv2.imshow('image',img1_n)
cv2.waitKey(0)

## Gaussian Blurring 
img1= cv2.imread(file1)
## For guassian blurring specify kernel size and sigma. sigma x is
# assumed same as sigma Y
blur = cv2.GaussianBlur(img1,(15,15),10)

cv2.imshow('image',img1)
cv2.waitKey(0)


cv2.imshow('image',blur)
cv2.waitKey(0)

## Lets try to smooth a noisy image
cv2.imshow('image',img1_n)
cv2.waitKey(0)
blur = cv2.GaussianBlur(img1_n,(15,15),10)
cv2.imshow('image',blur)
cv2.waitKey(0)

#Image filtering -http://docs.opencv.org/trunk/d4/d13/tutorial_py_filtering.html
## Add salt and pepper noise

img1= cv2.imread(file1)
cv2.imshow('image',img1)
cv2.waitKey(0)

def add_sp_noise(image):
      row,col,ch= image.shape
      s_vs_p = 0.5
      amount = 0.09
      out = image
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out

img1_sp = add_sp_noise(img1)
cv2.imshow('image',img1_sp)
cv2.waitKey(0)

# Median blurring. No is the kernel size. should be positive odd    
median = cv2.medianBlur(img1_sp,5)
cv2.imshow('image',median)
cv2.waitKey(0)

## Template matching
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
file2 = "C:\Users\s6324900\Desktop\Opencv image processing\people.jpg"
img_ppl= cv2.imread(file2, 0)
print(img_ppl.shape, img_ppl.size, img_ppl.dtype)

template = img_ppl[150:300, 120:260]
cv2.imshow('image',template)
cv2.waitKey(0)


w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] 
for meth in methods:
    img = img_ppl.copy()
    method = eval(meth)

     # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
   # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, (0,0,255), 2)
    
    plt.figure(figsize=(15,15))
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()

##Usual chosen method is 'cv2.TM_CCOEFF_NORMED'

## Canny Edge Detection
#http://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
file1 = "C:\Users\s6324900\Desktop\Opencv image processing\priyanka.jpg"
file2 = "C:\Users\s6324900\Desktop\Opencv image processing\cat.3.jpg"
#img1= cv2.imread(file1,0)
img1= cv2.imread(file2,0)
cv2.imshow('image',img1)
cv2.waitKey(0)
# Second and third arguments are our minVal and maxVal respectively.
edges = cv2.Canny(img1,100,200)

plt.figure(figsize=(8,8))
plt.subplot(121),plt.imshow(img1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
