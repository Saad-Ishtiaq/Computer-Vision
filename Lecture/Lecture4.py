import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import cv2 as cv

#Slide 9
print("This is task:01 done by Saad Ishtiaq")
img = io.imread('Saad.png')
()
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

plt.hist(newImage,bins=10)
plt.show()
plt.close()


# if No module named 'cv2'-> open anaconda prompt write pip install opencv-python
#Slide 10
im1=cv.imread('Saad.png',0)
new_im= cv.resize(im1,(100,100))
plt.imshow(new_im,cmap="gray")
plt.show()
plt.close()


#Slide 11
im2=cv.imread('Saad.png',0);
new_img=cv.resize(im2,(200,200))
cv.imshow("new image", im2)
cv.waitKey(0)
cv.destroyAllWindows()






