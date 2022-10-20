import numpy as np
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """ Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray  :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
     """
    # Your code here. You'll need to vectorise your implementation to ensure it runs # at a reasonable speed.
    kernel = np.flip(kernel)
    image_colum = image.shape[1]
    image_row = image.shape[0]
    kernel_colum = kernel.shape[1]
    kernel_row = kernel.shape[0]
    kernel_colum_half = kernel_colum // 2
    kernel_row_half = kernel_row // 2
    image_convolve = np.zeros(image.shape)
    if kernel_colum % 2 == 0 or kernel_row % 2 == 0:
        raise ValueError("Kernel size must be odd")
    if len(image.shape) == 3:
        for i in range(image_row):
            for j in range(image_colum):
                for k in range(image.shape[2]):
                    padding_image = np.pad(image[:, :, k], ((kernel_row_half, kernel_row_half), (kernel_colum_half, kernel_colum_half)), 'constant')
                    image_convolve[i, j, k] = np.sum(padding_image[i:i+kernel_row, j:j+kernel_colum] * kernel)

    else:
        for i in range(image_row):
            for j in range(image_colum):
                padding_image = np.pad(image, ((kernel_row_half, kernel_row_half), (kernel_colum_half, kernel_colum_half)), 'constant')
                image_convolve[i, j] = np.sum(padding_image[i:i+kernel_row, j:j+kernel_colum] * kernel)
    return image_convolve

