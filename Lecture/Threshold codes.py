# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:00:43 2021

@author: Saad  Ishtiaq
"""


import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import cv2 as cv



# Binary
img=cv.imread("original.png",0)

thresh=190

r,c=img.shape
i=0
j=0
new_im=np.zeros((r,c),dtype=np.uint8)
for i in range(r):
    for j in range(c):
        if(img[i,j]>=thresh):
            new_im[i,j]=255
        else:
            new_im[i,j]=0
plt.imshow(new_im,cmap="gray")

plt.show()
plt.close()



# Inverted Binary
img=cv.imread("original.png",0)

thresh=190
r,c=img.shape
i=0
j=0
new_im=np.zeros((r,c),dtype=np.uint8)
for i in range(r):
    for j in range(c):
        if(img[i,j]>=thresh):
            new_im[i,j]=0
        else:
            new_im[i,j]=255
plt.imshow(new_im,cmap="gray")

plt.show()
plt.close()


# Truncated
img=cv.imread("original.png",0)

thresh=190
r,c=img.shape
i=0
j=0
new_im=np.zeros((r,c),dtype=np.uint8)
for i in range(r):
    for j in range(c):
        if(img[i,j]>=thresh):
            new_im[i,j]=thresh
        else:
            new_im[i,j]=img[i,j]
plt.imshow(new_im,cmap="gray")

plt.show()
plt.close()



# To Zero
img=cv.imread("original.png",0)

thresh=190
r,c=img.shape
i=0
j=0
new_im=np.zeros((r,c),dtype=np.uint8)
for i in range(r):
    for j in range(c):
        if(img[i,j]>=thresh):
            new_im[i,j]=img[i,j]
        else:
            new_im[i,j]=0
plt.imshow(new_im,cmap="gray")

plt.show()
plt.close()


# To Zero Inverted
img=cv.imread("original.png",0)

thresh=190
r,c=img.shape
i=0
j=0
new_im=np.zeros((r,c),dtype=np.uint8)
for i in range(r):
    for j in range(c):
        if(img[i,j]>=thresh):
            new_im[i,j]=0
        else:
            new_im[i,j]=img[i,j]
plt.imshow(new_im,cmap="gray")

plt.show()
plt.close()


# #By Functions
# thresh=190
# img=cv.imread("original.png",0)
# #one line code
# r,img_b= cv.threshold(img,thresh,255,cv.THRESH_TOZERO_INV)
# plt.imshow(img_b,cmap="gray")
# plt.show()
# plt.close()


