import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, '..')
from imgProcess import *


def removeWords(self, img):
    img = getThresholdOtsu(img)


imgProcess = ImgProcess()
origin = cv2.imread('testImg/s1.jpg')
