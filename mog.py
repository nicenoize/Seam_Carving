import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import misc, ndimage

def convolution2D(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix
    :return: result of the convolution
    """
    newimg = np.zeros(img.shape)
    # TODO write convolution of arbritrary sized convolution here
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(img)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = img
    for x in range(img.shape[1]):     # Loop over every pixel of the image
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image
            newimg[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    
    return newimg

# load image and convert to floating point
image = mpl.image.imread('vsc-05/Broadway_tower_medium2.jpg')
img = np.asarray(image, dtype="float64")
# convert to grayscale
gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])

# TODO 1. define different kernels
# Identiy
identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
edge_detection = np.array([[1,0,-1], [0,0,0], [-1,0,1]])
sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
box_blur = np.multiply((np.array([[0,0,0],[0,0,0],[0,0,0]])), (1/9))
sobelX = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
sobelY = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
# TODO 2. implement convolution2D function and test with at least 4 different kernels
conv1 = convolution2D(gray, identity)
conv2 = convolution2D(gray, edge_detection)
conv3 = convolution2D(gray, sharpen)
sobelX = convolution2D(gray, sobelX)
sobelY = convolution2D(gray, sobelY)
blur_kernel = np.matrix('0.04 0.04 0.04 0.04 0.04; 0.04 0.04 0.04 0.04 0.04; 0.04 0.04 0.04 0.04 0.04; 0.04 0.04 0.04 0.04 0.04; 0.04 0.04 0.04 0.04 0.04')
# TODO 3. compute magnitude of gradients image
sobel = np.sqrt(np.add((np.square(sobelX)), (np.square(sobelY))))
# TODO 4. save all your results in image files, e.g. scipy.misc.imsave()
x = np.zeros((255, 255))
x = np.zeros((255, 255), dtype=np.uint8)
x[:] = np.arange(255)
scipy.misc.imsave("conv1.jpg", conv1)
scipy.misc.imsave("conv2.jpg", conv2)
scipy.misc.imsave("conv3.jpg", conv3)
scipy.misc.imsave("sobelX.jpg", sobelX)
scipy.misc.imsave("sobelY.jpg", sobelY)
scipy.misc.imsave("sobel.jpg", sobel)



