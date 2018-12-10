import matplotlib as mpl
import numpy as np
import scipy.ndimage
import mog
from matplotlib import image


def calculate_accum_energy(energy):
    """
    Function computes the accumulated energies
    :param energy:
    :return: nd array
    """
    accumE = np.array(energy)

    height = energy.shape[0]
    width = energy.shape[1]
    
    for i in range(height):
        for j in range(width):
            L = energy[i, (j-1) % width]
            R = energy[i, (j+1) % width]
            U = energy[(i-1) % height, j]
            D = energy[(i+1) % height, j]

            dx_sq = np.sum((R - L)**2)
            dy_sq = np.sum((D - U)**2)
            energy[i,j] = np.sqrt(dx_sq + dy_sq)
    accumE = energy
    # TODO compute accumulated energies - use the example from the exercise to debug
    # YOUR CODE HERE

    # print("after", accumE)
    return accumE



def magnitude_of_gradients(rgb):
    """
    Computes the magnitude of the sobel gradients from a grayscale image

    :param rgb: rgb image
    :return: image containing magnitude of gradients
    """

    # TODO: convert rgb to gray scale image
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    sobelX = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    sobelY = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    sobelX = mog.convolution2D(gray, sobelX)
    sobelY = mog.convolution2D(gray, sobelY)
    # TODO: compute magnitude of sobel gradients
    gray = np.sqrt(np.add((np.square(sobelX)), (np.square(sobelY))))
    # and return this image containing these values as energy
    # You can use your own code (exercise 1) or numpy/scipy built-in functions
    # YOUR CODE HERE
    return gray


if __name__ == '__main__':

    img = mpl.image.imread('./vsc-05/bird.jpg') #'Broadway_tower_medium2.jpg') #'bird.jpg')  #
    invSeamMask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    # hier am Anfang einfach mal auf 1 setzen
    number_of_seams_to_remove = 10
    newimg = np.array(img, copy=True)

    for r in range(number_of_seams_to_remove):

        # TODO: compute magnitude_of_gradients
        energy = magnitude_of_gradients(newimg)
        mpl.image.imsave('energy.png', energy)
        print("Energy: ", energy)

        # TODO: just test with some easy example in the beginning
        # energy = np.matrix('40 60 40 10; 53.3 50 25 47.5; 50 40 40 60')
        accumE = calculate_accum_energy(energy)
        print("AE: ", energy)

        # TODO: implement find and remove seam


        # save images in each iteration
        #mpl.image.imsave("carved_path"+str(r)+".png", img)
        #mpl.image.imsave("carved"+str(r)+".png", newimg)


        print(str(r), " image carved:", newimg.shape)

    #mpl.image.imsave("carved_path.png", img)
    #mpl.image.imsave("carved.png", newimg)

