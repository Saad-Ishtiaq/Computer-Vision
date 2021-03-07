# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 03:04:06 2021

@author: Saad  Ishtiaq
"""

import numpy as np
import imageio as io
import matplotlib.pyplot as plt

print("This is task: 02 done by Saad Ishtiaq")

img1 = io.imread('Saad.png')
img2 = io.imread('Saad.png')


rows,cols, depth = img1.shape
rows1,cols2,depth2 = img2.shape

newImage= np.full((rows,cols+cols2+5),255,dtype = np.uint8)

for i in range(rows):
    for j in range(cols):
        newImage[i,j]= img1[i,j,0]

for x in range(rows1):
    for y in range(cols2):
        newImage[x,y+cols+5]= img2[x,y,0]
        

plt.imshow(newImage, cmap="gray")
plt.show()
plt.close()



