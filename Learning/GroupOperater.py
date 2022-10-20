import numpy as np


def template_convolution(image, template, stride=1, padding=0):
    """
    Apply a template to the image
    :param image: input image
    :param template: template to be applied
    :param stride: stride of the template
    :param padding: padding of the template
    :return: output image
    """
    image = image.astype(np.float32)
    image = np.pad(image, padding, 'constant')
    output = np.zeros(image.shape)
    for i in range(0, image.shape[0] - template.shape[0] + 1, stride):
        for j in range(0, image.shape[1] - template.shape[1] + 1, stride):
            output[i, j] = np.sum(image[i:i + template.shape[0], j:j + template.shape[1]] * template)
    return output


def gaussion_filter(image, sigma):
    """
    Apply a gaussian filter to the image
    :param image: input image
    :param sigma: sigma of the gaussian filter
    :return: filtered image
    """
    image = image.astype(np.float32)
    x, y = np.meshgrid(np.arange(-3 * sigma, 3 * sigma + 1), np.arange(-3 * sigma, 3 * sigma + 1))
    template = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    template = template / np.sum(template)
    output = template_convolution(image, template)
    return output


