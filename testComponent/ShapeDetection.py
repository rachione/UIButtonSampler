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
    return (h > 5 and w > 5 and w < 150)


def removeContains(rects):
    for i, rect1 in enumerate(rects):
        x1, y1, w1, h1 = rect1
        for j, rect2 in enumerate(rects):
            if i == j:
                continue
            x2, y2, w2, h2 = rect2
            if (x2 + w2) >= (x1 + w1)\
                    and x2 <= x1\
                    and y2 <= y1\
                    and (y2 + h2) >= (y1 + h1):
                rects.remove(rect1)
                break

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
area = np.copy(img)
cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
rects = [cv2.boundingRect(cnt) for cnt in cnts]
rects = [rect for rect in rects if isGoodRect(rect)]
rects = removeContains(rects)

for rect in rects:
    x, y, w, h = rect
    cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)


# for rect in rects:
#     x, y, w, h = rect
#     ROI = gray[y:y + h, x:x + w]
#     word = pytesseract.image_to_string(ROI, 'jpn+eng')
#     word = word.replace(' ', '').replace('\f', '').replace('\n', '')
#     if word != '':
#         print(word)
#         cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("dilate", dilate)
cv2.imshow("area", area)
cv2.waitKey(0)
