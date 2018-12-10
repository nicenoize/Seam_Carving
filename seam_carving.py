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
    # TODO compute accumulated energies - use the example from the exercise to debug
    # Source: http://www.shawnye1994.com/2018/08/06/Seam-Carving/
    Height, Width = energy.shape
    for h in range(1,Height):
        for w in range(0,Width):
            if w == 0:
                upper_pixels = [energy[h-1,w], energy[h-1, w+1]]
            elif w == Width-1:
                upper_pixels = [energy[h-1, w-1], energy[h-1,w]]
            else:
                upper_pixels = [energy[h-1, w-1], energy[h-1,w],
                                energy[h-1, w+1]]
            min_energy = np.amin(upper_pixels)
            accumE[h,w] += min_energy
    print("AccumE: \n", accumE)
    # print("after", accumE)
    return accumE


def create_seam_mask(accumE):
    """
    Creates and returns boolean matrix containing zeros (False) where to remove the seam
    :param accumE:
    :return:
    """
    # bei accumE von unten nach oben laufen und Minima finden um Pfad zu erstellen
    seamMask = np.ones(accumE.shape, dtype=bool)

    # TODO: find minimum index in accumE matrix 
    minIdx = np.argmin(accumE[(accumE.shape[0])-1,:])
    # TODO: fill the seamMask and invSeamMask (global variable ... bad software design,
    # but just to debug anyway)
    H,W = accumE.shape

    for row in reversed(range(0, accumE.shape[0])):
        # print "seam mask", row, minIdx
        seamMask[row, minIdx] = False
        invSeamMask[row, minIdx] = True
        # TODO: compute minIdx for each row
        # Eine Reihe weiter hoch
        if minIdx == 0:
            upper_index = [0, 2]
        elif minIdx == W-1:
            upper_index = [W-2, W]
        else:
            upper_index = [minIdx-1, minIdx+2]

        temp_index = np.argmin(accumE[row,upper_index[0]:upper_index[1]])
        minIdx = list(range(upper_index[0], upper_index[1]))[temp_index]

            
    print("Seam Mask: \n", seamMask)
    print("invSeam Mask: \n", invSeamMask)



    return seamMask


def seam_carve(image, seamMask):
    """
    Removes a seam from the image depending on the seam mask. Returns an image
     that has one column less than <image>

    :param image:
    :param seamMask:
    :return: smaller image
    """
    shrunkenImage = np.zeros((image.shape[0], image.shape[1]-1, image.shape[2]), dtype=np.uint8)

    for i in range(seamMask.shape[0]):
        shrunkenImage[i, :, 0] = image[i, seamMask[i, :], 0]
        shrunkenImage[i, :, 1] = image[i, seamMask[i, :], 1]
        shrunkenImage[i, :, 2] = image[i, seamMask[i, :], 2]

    return shrunkenImage


def find_and_remove_seam(im, energy):
    """
    Finds and remove the seam containg the minimum energy over the
    original image.

    :param im: original rgb image
    :param energy: image (matrix) containing the energies
    :return: image with one seam removed
    """

    # compute the accumulated energies
    accumEnergies = calculate_accum_energy(energy)

    # create seam mask
    seamMask = create_seam_mask(accumEnergies)

    # use seam mask and remove seam in image
    carved = seam_carve(im, seamMask)
    return carved



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
    print("Magnitude of gradient: \n", gray)
    # and return this image containing these values as energy
    # You can use your own code (exercise 1) or numpy/scipy built-in functions
    # YOUR CODE HERE
    return gray


if __name__ == '__main__':

    img = mpl.image.imread('./vsc-05/bird.jpg') #'Broadway_tower_medium2.jpg') #'bird.jpg')  #
    invSeamMask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    # hier am Anfang einfach mal auf 1 setzen
    number_of_seams_to_remove = 20
    newimg = np.array(img, copy=True)
    energy = magnitude_of_gradients(newimg)


    for r in range(number_of_seams_to_remove):

        # TODO: compute magnitude_of_gradients
        energy = magnitude_of_gradients(newimg)
        # mpl.image.imsave('energy.png', energy)

        # TODO: just test with some easy example in the beginning
        # energy = np.matrix('40 60 40 10; 53.3 50 25 47.5; 50 40 40 60')

        # TODO: implement find and remove seam
        newimg = find_and_remove_seam(newimg, energy)

        # For debugging purposes we keep invSeamMask
        # that contains all seams from the original image
        # for i in range(invSeamMask.shape[0]):
        #     img[i, invSeamMask[i, :], 0] = 255
        #     img[i, invSeamMask[i, :], 1] = 0
        #     img[i, invSeamMask[i, :], 2] = 0

        # save images in each iteration
        mpl.image.imsave("carved_path"+str(r)+".png", img)
        mpl.image.imsave("carved"+str(r)+".png", newimg)


        print(str(r), " image carved:", newimg.shape)

    mpl.image.imsave("carved_path.png", img)
    mpl.image.imsave("carved.png", newimg)


    accumEnergies = (calculate_accum_energy(energy))
    seamMask = create_seam_mask(accumEnergies)

