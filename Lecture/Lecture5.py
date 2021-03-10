# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:23:30 2021

@author: Saad  Ishtiaq
"""

import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import cv2 as cv



#ravel
img=cv.imread("original.png",0)


img_b=np.zeros(img.shape,dtype=np.uint8)
plt.imshow(img,cmap="gray")
plt.show()
plt.close()




plt.hist(img.ravel(),bins=100)
plt.show()
plt.close()


thresh=190
#one line code
r,img_b= cv.threshold(img,thresh,255,cv.THRESH_BINARY)
"""
#manual execution
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
"""
plt.imshow(img_b,cmap="gray")
plt.show()
plt.close()







# Creating histogram with numpy function
np.histogram(img, bins = 10) 
hist, bins = np.histogram(img, bins =10)  
  
# printing histogram 
print() 
print (hist)  
print (bins)  
print() 
plt.imshow(img_b,cmap="gray")
plt.show()
plt.close()



