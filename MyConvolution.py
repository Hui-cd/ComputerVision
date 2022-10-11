import numpy as np
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """ Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray  :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
     """
    # Your code here. You'll need to vectorise your implementation to ensure it runs # at a reasonable speed.
    image_colum = image.shape[1]
    image_row = image.shape[0]
    kernel_colum = kernel.shape[1]
    kernel_row = kernel.shape[0]
    kernel_colum_half = kernel_colum // 2
    kernel_row_half = kernel_row // 2
    image_convolve = np.zeros(image.shape)
    for i in range(kernel_colum_half+1, image_colum-kernel_colum_half):
        for j in range(kernel_row_half+1 , image_row-kernel_row_half):
            sum = 0
            for k in range(kernel_colum):
                for l in range(kernel_row):
                    sum += image[i-kernel_colum_half+k][j-kernel_row_half+l] * kernel[k][l]
            image_convolve[i][j] = sum
    return image_convolve
