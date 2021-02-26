import cv2
import numpy as np

img = cv2.imread('testImg/s1.jpg')

area = np.copy(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 7, 2)

thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 7, 1)

thresh_inv = cv2.bitwise_not(thresh2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(thresh_inv, kernel, iterations=1)

cnts, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 20 and w > 20 and w < 500:
        cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)

images = [thresh, thresh2, dilate, area]

for i in range(len(images)):
    cv2.imshow('test%d' % i, images[i])
cv2.waitKey(0)