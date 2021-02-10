import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, '..')
from imgProcess import TrackbarDebug


def getContours(dilate, origin):
    cnts, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 30 and w > 50 and w < 500 and w * h < 50000:
            areas.append((x, y, w, h))
            cv2.rectangle(origin, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('origin', origin)
    # cv2.imshow('dilate', dilate)
    # cv2.waitKey(0)


def getAverageColor(img, x, y, range):
    offset = -range // 2
    x = x + offset
    y = y + offset
    cropImg = img[y:y + range, x:x + range]

    avgColorPerRow = np.average(cropImg, axis=0)
    avgColor = np.average(avgColorPerRow, axis=0)
    return cropImg, avgColor


def getSegmentColor(img):
    max_dist = 80
    cropImg, avgColor = getAverageColor(img, 185, 584, 30)
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    # Euclidean distance
    sq_dist = (R - avgColor[0]) ** 2 + \
        (G - avgColor[1]) ** 2 + (B - avgColor[2]) ** 2
    mask = sq_dist < (max_dist**2)
    res = np.repeat(mask[..., None], 3, axis=2) * img

    colorImg = np.zeros((100, 100, 3), np.uint8)
    colorImg[:] = avgColor
    cv2.imshow("cropImg", cropImg)
    cv2.imshow("avgColor", colorImg)
    cv2.imshow("res", res)
    cv2.waitKey(0)

    return res


def removeWords(self, img):
    img = getThresholdOtsu(img)


origin = cv2.imread('testImg/s1.jpg')
# img = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
# img = TrackbarDebug().imshow(
#     114, 255,
#     (lambda min, max: cv2.threshold(img, min, max, cv2.THRESH_BINARY)))
# img = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)

img = getSegmentColor(origin)

# blur = cv2.GaussianBlur(img, (3, 3), 3, 0)
# edge = TrackbarDebug().imshow(82, 143,
#                               (lambda min, max: cv2.Canny(img, min, max)))
edge = cv2.Canny(img, 82, 143)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilate = cv2.dilate(edge, kernel)

getContours(dilate, origin)
