# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:42:09 2021

@author: Saad  Ishtiaq
"""



import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import cv2 as cv


img=cv.imread("Quiz1.jpeg",0)

backtorgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
r,c,l=backtorgb.shape
i=0
j=0
k=0
thresh=18
new_im=np.zeros((r,c,l),dtype=np.uint8)
for i in range(r):
    for j in range(c):
        for k in range(l):   
            if(backtorgb[i,j,k]<thresh):
                new_im[i,j,k]=backtorgb[i,j,k]
            if(backtorgb[i,j,k]>=thresh and backtorgb[i,j,k]<=thresh+40):
                new_im[i,j,0]=255
            if(backtorgb[i,j,k]>thresh+30):
                new_im[i,j,k]=backtorgb[i,j,k]
plt.imshow(new_im)
plt.show()
plt.close()


