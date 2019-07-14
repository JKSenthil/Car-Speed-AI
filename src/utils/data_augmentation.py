import cv2
import numpy as np

def rgb2grey(imgs):
    """
    Converts a group of images from color to black and white
    imgs :: np.array with shape (i, img_width, img_height)
    """
    converted_imgs = np.zeros(imgs.shape)
    for i in range(len(imgs)):
        converted_imgs[i,:,:] = cv2.cvtColor(imgs[i,:,:], cv2.COLOR_BGR2GRAY)
    return converted_imgs

# TODO need to implement equalize adapthist next