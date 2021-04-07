# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:02:34 2021

@author: Saad  Ishtiaq
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



#Gaussian Noise


img=cv2.imread('Saad.png',0)

#Generate Gaussian Noise
gauss_noise= np.random.normal(0,1,img.shape)
#Changed float to int
gauss_noise= gauss_noise.astype('uint8')
# img= img.astype("float64")
plt.imshow(img,cmap="gray")
plt.show()
plt.close()


noise_img=cv2.add(img,gauss_noise)
# Histogram BeforeNoice
plt.hist(img.ravel(),bins=100)
plt.show()
plt.close()

# Histogram afterNoice
plt.hist(noise_img.ravel(),bins=100)
plt.show()
plt.close()

plt.imshow(noise_img,cmap='gray')
plt.show()
plt.close()









#Salt and Pepper Noise

import random
impulseNoise= np.zeros(img.shape,dtype=np.uint8)
r,c=img.shape

prob=0.08
thresh=1-prob

for i in range(r):
    for j in range(c):
        rand=random.random()
        if rand<prob :
            impulseNoise[i][j]=0
        elif rand>thresh:
            impulseNoise[i][j]=255
        else:
            impulseNoise[i][j]=img[i][j]


# Histogram BeforeNoice
plt.hist(img.ravel(),bins=100)
plt.show()
plt.close()
#Histogram
plt.hist(impulseNoise.ravel(),bins=100)
plt.show()
plt.close()


plt.imshow(impulseNoise,cmap='gray')
plt.show()
plt.close()


# MedianFiltering  Built-in Function
medianImage=cv2.medianBlur(impulseNoise,3)
plt.imshow(medianImage,cmap='gray')
plt.show()
plt.close()




# UsamaAnotherTry of Median Filter------Successful
kerelArray=np.zeros((9),dtype=np.float)
rows,columns=impulseNoise.shape
new_img=np.zeros((rows+2,columns+2),dtype=np.uint8)
for j in range(columns+1):
    new_img[0][j]=0
    new_img[rows+1][j]=0

for i in range(rows+1):
    new_img[i][0]=0
    new_img[i][columns+1]=0
    
for i in range(rows):
    for j in range(columns):
        new_img[i+1][j+1]=impulseNoise[i][j]
        kerelArray[0]=new_img[i-1][j-1]
        kerelArray[1]=new_img[i-1][j]
        kerelArray[2]=new_img[i-1][j+1]
        kerelArray[3]=new_img[i][j-1]
        kerelArray[4]=new_img[i][j]
        kerelArray[5]=new_img[i][j+1]
        kerelArray[6]=new_img[i+1][j-1]
        kerelArray[7]=new_img[i+1][j]
        kerelArray[8]=new_img[i+1][j+1]
        kerelArray=np.sort(kerelArray)
        new_img[i][j]=kerelArray[4]
plt.imshow(new_img,cmap="gray")
plt.show()
plt.close()





# Weighted Average filter
kerelArray=np.zeros((3,3),dtype=np.float)
kerelArray[0][0]=1
kerelArray[0][1]=2
kerelArray[0][2]=1
kerelArray[1][0]=2
kerelArray[1][1]=4
kerelArray[1][2]=2
kerelArray[2][0]=1
kerelArray[2][1]=2
kerelArray[2][2]=1

rows,columns=impulseNoise.shape
new_img=np.zeros((rows,columns),dtype=np.float)
for i in range(1,rows-1):
    for j in range(1,columns-1):
        new_img[i][j]=(1/16)*(kerelArray[0][0]*impulseNoise[i-1][j-1]+kerelArray[0][1]*impulseNoise[i-1][j]+kerelArray[0][2]*impulseNoise[i-1][j+1]+kerelArray[1][0]*impulseNoise[i][j-1]+kerelArray[1][1]*impulseNoise[i][j]+kerelArray[1][2]*impulseNoise[i][j+1]+kerelArray[2][0]*impulseNoise[i+1][j-1]+kerelArray[2][1]*impulseNoise[i+1][j]+kerelArray[2][2]*impulseNoise[i+1][j+1])
plt.imshow(new_img,cmap="gray")
plt.show()
plt.close()





# Usama Average filter
kerelArray=np.zeros((3,3),dtype=np.float)
kerelArray[0][0]=1
kerelArray[0][1]=1
kerelArray[0][2]=1
kerelArray[1][0]=1
kerelArray[1][1]=1
kerelArray[1][2]=1
kerelArray[2][0]=1
kerelArray[2][1]=1
kerelArray[2][2]=1

rows,columns=impulseNoise.shape
new_img=np.zeros((rows,columns),dtype=np.float)
for i in range(1,rows-1):
    for j in range(1,columns-1):
        new_img[i][j]=(1/9)*(kerelArray[0][0]*impulseNoise[i-1][j-1]+kerelArray[0][1]*impulseNoise[i-1][j]+kerelArray[0][2]*impulseNoise[i-1][j+1]+kerelArray[1][0]*impulseNoise[i][j-1]+kerelArray[1][1]*impulseNoise[i][j]+kerelArray[1][2]*impulseNoise[i][j+1]+kerelArray[2][0]*impulseNoise[i+1][j-1]+kerelArray[2][1]*impulseNoise[i+1][j]+kerelArray[2][2]*impulseNoise[i+1][j+1])
plt.imshow(new_img,cmap="gray")
plt.show()
plt.close()




# Gaussian Filter
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


im=cv.imread('Saad.png',0) 
# Image Reading   
plt.imshow(im,cmap="gray")
plt.show()
plt.close()
kerelArray=np.zeros((3,3),dtype=np.float)
# Sigma Value
sigma=1
# To Calculate Values
sum=0
for i in range(-1,2):
    for j in range(-1,2):
        kerelArray[i+1][j+1]=(1/(2*3.14*pow(sigma,2)))*(pow(2.718,((-(pow(i,2)+pow(j,2)))/(2*pow(sigma,2)))))
        sum=sum+kerelArray[i+1][j+1]
rows,columns=im.shape
new_img=np.zeros((rows,columns),dtype=np.float)
for i in range(1,rows-1):
    for j in range(1,columns-1):
        new_img[i][j]=(1/sum)*(kerelArray[0][0]*im[i-1][j-1]+kerelArray[0][1]*im[i-1][j]+kerelArray[0][2]*im[i-1][j+1]+kerelArray[1][0]*im[i][j-1]+kerelArray[1][1]*im[i][j]+kerelArray[1][2]*im[i][j+1]+kerelArray[2][0]*im[i+1][j-1]+kerelArray[2][1]*im[i+1][j]+kerelArray[2][2]*im[i+1][j+1])
plt.imshow(new_img,cmap="gray")
plt.show()
plt.close()








# Generate Poisson Noise
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('original.png',0)

noisy_img = np.zeros(img.shape,np.uint8)

# 10=value of lembda, img.shape=size
poisson_noise=np.random.poisson(10,img.shape)

# To make both compatible
poisson_noise=poisson_noise.astype(np.uint8)
noisy_img = img + poisson_noise


plt.imshow(img,cmap="gray")
plt.show()
plt.close()


plt.imshow(noisy_img,cmap="gray")
plt.show()
plt.close()






# Generate Speckle Noise
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('original.png',0)
r,c = img.shape
speckle_noise = np.random.normal(0,1,img.shape)
speckle_img =  img + img * speckle_noise

# To add a percentage of speckle noise
# speckle_img =  img + img *(.1* speckle_noise)


speckle_img = speckle_img.astype(np.uint8)      


plt.imshow(img,cmap="gray")
plt.show()
plt.close()

plt.imshow(speckle_img,cmap="gray")
plt.show()
plt.close()






import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.util import random_noise


img = cv2.imread('original.png',0)
gauss = random_noise(img, mode='gaussian')
salt_pepper = random_noise(img, mode='s&p')
poisson =random_noise(img,mode='poisson')
speckle=random_noise(img,mode='speckle')


plt.subplot(231), plt.axis('off'), plt.imshow(img), plt.title('Origin')
plt.subplot(232), plt.axis('off'), plt.imshow(gauss), plt.title('Gaussian')
plt.subplot(233), plt.axis('off'), plt.imshow(salt_pepper), plt.title('Salt & Pepper')
plt.subplot(234), plt.axis('off'), plt.imshow(poisson), plt.title('poisson')
plt.subplot(235), plt.axis('off'), plt.imshow(speckle), plt.title('speckle')

# # Show GrayScale Image
# plt.imshow(gauss,cmap="gray")
# plt.show()
# plt.close()
# plt.imshow(salt_pepper,cmap="gray")
# plt.show()
# plt.close()
# plt.imshow(poisson,cmap="gray")
# plt.show()
# plt.close()
# plt.imshow(speckle,cmap="gray")
# plt.show()
# plt.close()


# To show in seperate window
# cv2.imshow("Images", gauss)
# cv2.waitKey(0)
# cv2.destroyAllWindows


plt.show();
plt.close()











#HistogramEqualization-----looks like contrast streching but different

import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('original.png',0)
#Generate Gaussian Noise


# Histogram Before
plt.imshow(img,cmap="gray")
plt.show()
plt.close()

plt.hist(img.ravel(),bins=100)
plt.show()
plt.close()

new_Image=cv2.equalizeHist(img)
# Histogram after
plt.hist(new_Image.ravel(),bins=100)
plt.show()
plt.close()
plt.imshow(new_Image,cmap='gray')
plt.show()
plt.close()








# Ma'am Method of Contrast streching
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('original.png',0)
r,c=img.shape
i=0
j=0

MP=255
a= img.min()
b= img.max()
R=b-a

new_im=np.zeros((r,c),dtype=np.float)
for i in range(r):
    for j in range(c):
         new_im[i][j]=((img[i][j]-a)/R)*MP
         new_im[i][j]= round(new_im[i][j])


plt.imshow(img,cmap="gray")
plt.show()
plt.close()
# Histogram Before
plt.hist(img.ravel(),bins=100)
plt.show()
plt.close()

plt.imshow(new_im,cmap="gray")
plt.show()
plt.close()
# Histogram After
plt.hist(new_im.ravel(),bins=100)
plt.show()
plt.close()










# Ma'am Method of Contrast/Image Enhancement
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('original.png',0)
r,c=img.shape
i=0
j=0

new_im=np.zeros((r,c),dtype=np.float)
for i in range(r):
    for j in range(c):
         new_im[i][j]=(255-img[i][j])

plt.imshow(img,cmap="gray")
plt.show()
plt.close()
# Histogram Before
plt.hist(img.ravel(),bins=100)
plt.show()
plt.close()

plt.imshow(new_im,cmap="gray")
plt.show()
plt.close()
# Histogram After
plt.hist(new_im.ravel(),bins=100)
plt.show()
plt.close()













