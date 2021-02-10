import cv2
import numpy as np
import sys
import pytesseract
from pathlib import Path
sys.path.insert(0, '..')
from imgProcess import *
pytesseract.pytesseract.tesseract_cmd = r"D:\EXE\Tesseract-OCR\tesseract.exe"


def isGoodRect(rect):
    x, y, w, h = rect
    return (h > 15 and w > 5 and w < 150 and h < 150)


def removeContains(rects):
    for rect1 in rects[:]:
        x1, y1, w1, h1 = rect1
        for rect2 in rects[:]:
            x2, y2, w2, h2 = rect2
            if rect1 == rect2:
                continue
            if (x2 + w2) >= (x1 + w1)\
                    and x2 <= x1\
                    and y2 <= y1\
                    and (y2 + h2) >= (y1 + h1):
                rects.remove(rect1)
                break

    return rects


def getBiggerRect(img, rects):
    imgH, imgW, _ = img.shape
    for i in range(len(rects)):
        x, y, w, h = rects[i]
        rects[i] = max(0, x - 10), max(0, y - 10), min(imgW,
                                                       w + 20), min(imgH, h + 20)

    return rects


debug = TrackbarDebug()
img = cv2.imread('testImg/s1.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edge = debug.imshow(
#     280, 500, (lambda min, max: cv2.Canny(gray, min, max)))
edge = cv2.Canny(gray, 280, 500)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilate = cv2.dilate(edge, kernel, iterations=4)

cnts, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
rects = [cv2.boundingRect(cnt) for cnt in cnts]
rects = [rect for rect in rects if isGoodRect(rect)]
rects = removeContains(rects)

# for rect in rects:
#     x, y, w, h = rect
#     cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)

area = np.copy(img)
OTSUtest = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
for rect in rects:
    x, y, w, h = rect
    ROI = gray[y:y + h, x:x + w]
    _, ROI = cv2.threshold(ROI, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    OTSUtest[y:y + h, x:x + w] = ROI
    word = pytesseract.image_to_string(
        ROI, lang='jpn', config='--psm 10 --oem 1')
    word = word.replace(' ', '').replace('\f', '').replace('\n', '')
    if word != '':
        print(word)
        cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("OTSUtest", OTSUtest)
cv2.imshow("dilate", dilate)
cv2.imshow("area", area)
cv2.waitKey(0)
