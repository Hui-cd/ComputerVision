import  numpy as np
from PIL import Image
from MyHybridImages import myHybridImages
low_image = Image.open('/Users/hui/Project/ComputerVision/data/cat.bmp')
low_image = np.array(low_image)
low_sigma = 1.0
high_image = Image.open('/Users/hui/Project/ComputerVision/data/dog.bmp')
high_image = np.array(high_image)
high_sigma = 1.0

hybrid_image = myHybridImages(low_image, low_sigma, high_image, high_sigma)
hybrid_image = Image.fromarray(hybrid_image.astype(np.uint8))
hybrid_image.save('/Users/hui/Project/ComputerVision/data/hybrid_image.png')
