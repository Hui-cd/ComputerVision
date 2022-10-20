from Learning import EdgeDetection
from Learning.FourierTransform import fourier_transform
from Learning.Utility import load_image, export_image
import Learning

image = load_image("/Users/hui/PycharmProjects/ComputerVision/cw/trump_real.png")
image = EdgeDetection.basic_edge_detection_operator(image)
export_image(image, "/Users/hui/PycharmProjects/ComputerVision/cw/trump_real1_fourier.png")
