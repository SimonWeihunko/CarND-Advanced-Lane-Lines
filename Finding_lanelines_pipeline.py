#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

print(os.listdir("test_images/"))

image = mpimg.imread('test_images/test1.jpg')

# grayscaling the image
print('This image is:', type(image), 'with dimensions:', image.shape)
image_gray = grayscale(image)
f1 = plt.figure(1)
plt.imshow(image_gray, cmap = 'gray')
plt.show()