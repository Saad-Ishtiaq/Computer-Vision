# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:27:56 2021

@author: Saad  Ishtiaq
"""

import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import cv2 as cv


img=cv.imread("original.png",0)

plt.hist(img.ravel(),bins=100)
plt.show()
plt.close()



r,c=img.shape
i=0
j=0
thresh=157
new_im=np.zeros((r,c),dtype=np.uint8)

for i in range(r):
    for j in range(c):
        if(img[i,j]<(thresh)):
                new_im[i,j]=1
        else:
                new_im[i,j]=0
                
plt.imshow(img,cmap="gray")
plt.show()
plt.close()                




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

        