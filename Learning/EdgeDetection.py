import numpy as np
from scipy.ndimage import gaussian_filter

from Learning.GroupOperater import template_convolution


def basic_edge_detection_operator(image):
    """
    Basic edge detection operator 基本的边缘检测算子
    :param image: input image
    :return: edge image
    """
    image = image.astype(np.float32)
    template = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    output = template_convolution(image, template)
    return output


def prewitt_edge_detection_operator(image):
    """
    Prewitt edge detection operator prewitt边缘检测算子
    :param image: input image
    :return: edge image
    """
    image = image.astype(np.float32)
    template_mx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    template_my = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    output_mx = template_convolution(image, template_mx)
    output_my = template_convolution(image, template_my)
    output = np.sqrt(output_mx ** 2 + output_my ** 2)
    return output

def sobel_edge_detection_operator(image):
    """
    Sobel edge detection operator sobel边缘检测算子
    :param image: input image
    :return: edge image
    """
    image = image.astype(np.float32)
    template_mx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    template_my = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    output_mx = template_convolution(image, template_mx)
    output_my = template_convolution(image, template_my)
    output = np.sqrt(output_mx ** 2 + output_my ** 2)
    return output

def roberts_edge_detection_operator(image):
    """
    Roberts edge detection operator roberts边缘检测算子
    :param image: input image
    :return: edge image
    """
    image = image.astype(np.float32)
    template_mx = np.array([[1, 0], [0, -1]])
    template_my = np.array([[0, 1], [-1, 0]])
    output_mx = template_convolution(image, template_mx)
    output_my = template_convolution(image, template_my)
    output = np.sqrt(output_mx ** 2 + output_my ** 2)
    return output

def canny_edge_detection_operator(image):
    """
    Canny edge detection operator canny边缘检测算子
    :param image: input image
    :return: edge image
    """
    image = image.astype(np.float32)
    image = gaussian_filter(image, 1)
    image = sobel_edge_detection_operator(image)
    return image

def laplacian_edge_detection_operator(image):
    """
    Laplacian edge detection operator 拉普拉斯的边缘检测算子
    :param image: input image
    :return: edge image
    """
    image = image.astype(np.float32)
    template = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    output = template_convolution(image, template)
    return output

def marr_hildreth_edge_detection_operator(image, sigma):
    """
    Marr-Hildreth edge detection operator marr-hildreth边缘检测算子
    :param image: input image
    :param sigma: sigma of the gaussian filter
    :return: edge image
    """
    image = image.astype(np.float32)
    x, y = np.meshgrid(np.arange(-3 * sigma, 3 * sigma + 1), np.arange(-3 * sigma, 3 * sigma + 1))
    template = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * (x ** 2 + y ** 2 - 2 * sigma ** 2) / (sigma ** 4)
    template = template / np.sum(template)
    output = template_convolution(image, template)
    return output






