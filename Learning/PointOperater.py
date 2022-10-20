import numpy as np
import cv2


def intensity_normalization(image):
    """Intensity normalization
    :param image: the image to be normalized
    :type numpy.ndarray
    :return: the normalized image
    :rtype numpy.ndarray
    """
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def histogram_equalization(image):
    """Histogram equalization
    :param image: the image to be equalized
    :type numpy.ndarray
    :return: the equalized image
    :rtype numpy.ndarray
    """
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * 255
    image = image.astype(np.uint8)
    image = cv2.equalizeHist(image)
    image = image.astype(np.float32)
    image = image / 255
    return image


def thresholding(image, threshold,output_max, output_min):
    """Thresholding
    :param image: the image to be thresholded
    :type numpy.ndarray
    :param threshold: the threshold value
    :type float
    :return: the thresholded image
    :rtype numpy.ndarray
    """
    image = image.astype(np.float32)
    image[image > threshold] = output_max
    image[image <= threshold] = output_min
    return

