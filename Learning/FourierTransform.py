import numpy as np
import scipy.fftpack as fft
import cv2


def fourier_transform(image):
    """
    fourier transform of image and return the magnitude and phase of the image in frequency domain
    :param image: the image in spatial domain
    :type numpy.ndarray
    :return: the fourier transform of the image
    :rtype numpy.ndarray
    """
    image = fft.fft2(image)
    image = fft.fftshift(image)
    return image


def inverse_fourier_transform(image):
    """
    inverse fourier transform of image and return the magnitude and phase of the image in frequency domain
    :param image: the image in frequency domain
    :type numpy.ndarray
    :return: the inverse fourier transform of the image
    :rtype numpy.ndarray
    """
    image = fft.ifftshift(image)
    image = fft.ifft2(image)
    image = np.abs(image)
    image = image * 255
    image = image.astype(np.uint8)
    return image
