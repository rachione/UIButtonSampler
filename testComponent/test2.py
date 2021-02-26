import cv2
import sys
import numpy as np
sys.path.insert(0, '..')
from imgProcess import *

img = cv2.imread('testImg/s1.jpg', 0)
img = TrackbarDebug().imshow(
    127, 255,
    (lambda min, max: cv2.threshold(img, min, max, cv2.THRESH_BINARY)))

num_labels, labels_im = cv2.connectedComponents(img)


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()


imshow_components(labels_im)