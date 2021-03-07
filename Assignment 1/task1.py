# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 01:48:24 2021

@author: Saad  Ishtiaq
"""



import numpy as np
import imageio as io
import matplotlib.pyplot as plt


print("This is task:01 done by Saad Ishtiaq")
img = io.imread('Saad.png')

rows, cols ,depth = img.shape
newImage= np.empty((rows,cols))

for i in range(rows):
    for j in range (cols):
        red=0.30*(img[i,j,0])
        green=0.59*(img[i,j,1])
        blue=0.11*(img[i,j,2])
        newImage[i,j]=red+green+blue


plt.imshow(newImage,cmap="gray")
plt.show()
plt.close()

