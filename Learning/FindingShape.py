import cv2
import numpy as np


def hough_transform(img, threshold=100, line_length=5, line_gap=3):
    # Hough transform
    lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold, minLineLength=line_length, maxLineGap=line_gap)
    return lines