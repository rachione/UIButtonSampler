import cv2
import uuid
import numpy as np
from enum import Enum
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\exe\Tesseract-OCR\tesseract.exe"


class Rect(Enum):
    WORD = 1
    PANEL = 2


class TrackbarDebug:
    def updateMin(self, x):
        self.min = x

    def updateMax(self, x):
        self.max = x

    def resize(self, src, percent):
        w = int(src.shape[1] * percent / 100)
        h = int(src.shape[0] * percent / 100)
        dsize = (w, h)
        return cv2.resize(src, dsize)

    def imshow(self, min, max, callback):
        cv2.namedWindow('image')
        cv2.createTrackbar('Min', 'image', 0, 500, self.updateMin)
        cv2.createTrackbar('Max', 'image', 0, 500, self.updateMax)

        self.min = min
        self.max = max
        cv2.setTrackbarPos('Min', 'image', self.min)
        cv2.setTrackbarPos('Max', 'image', self.max)

        while (True):
            img = callback(self.min, self.max)
            # for cv2.threshold
            if type(img) is tuple:
                img = img[1]
            cv2.imshow('image', self.resize(img, 50))
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()

        return img


class RectModule:
    def __init__(self):
        self.debug = TrackbarDebug()

    def isGoodRange(self, rect, type):
        x, y, w, h = rect
        return {
            Rect.WORD: (h > 15 and w > 5 and w < 150 and h < 150),
            Rect.PANEL: (h > 20 and w > 30 and w < 300 and h < 300),
        }[type]

    def sort(self, rects):
        return sorted(rects, key=lambda rect: rect[2] * rect[3])

    #remove contained and overlapped rects
    def removeBadRects(self, rects):
        rects = self.sort(rects)
        for rect1 in rects[:]:
            x1, y1, w1, h1 = rect1
            for rect2 in rects[:]:
                x2, y2, w2, h2 = rect2
                if rect1 == rect2:
                    continue

                isContain=(x2 + w2) >= (x1 + w1)\
                        and x2 <= x1\
                        and y2 <= y1\
                        and (y2 + h2) >= (y1 + h1)

                isOverlap= not((x1>=(x2 + w2))\
                        or ((x1+w1)<=x2)\
                        or (y1>=(y2+h2))\
                        or ((y1+h1)<=y2))
                if isContain or isOverlap:
                    rects.remove(rect1)
                    break

        return rects

    def getBiggerRect(self, img, rects):
        imgH, imgW, _ = img.shape
        for i in range(len(rects)):
            x, y, w, h = rects[i]
            rects[i] = max(0, x - 3), max(0,y - 3),\
                        min(imgW,w + 6), min(imgH, h + 6)

        return rects

    def getGoodRect(self, cnts, rectType):
        rects = [cv2.boundingRect(cnt) for cnt in cnts]
        rects = [rect for rect in rects if self.isGoodRange(rect, rectType)]
        rects = self.removeBadRects(rects)
        return rects


class ImgProcess:
    def __init__(self):
        self.debug = TrackbarDebug()
        self.rectModule = RectModule()

    def imwrite(self, img):
        name = str(uuid.uuid1())
        path = 'samples/%s.jpg' % name
        cv2.imwrite(path, img)

    def imshow(self, imgs):
        for i, img in enumerate(imgs):
            cv2.imshow('img%d' % i, img)
        cv2.waitKey()

    def convertBGR(self, pilImg):
        img = np.array(pilImg)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def convertHSV(self, pilImg):
        img = np.array(pilImg)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def convertGray(self, pilImg):
        img = np.array(pilImg)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def getAverageColor(self, img, pos, range):
        cnt = (pos.x - range // 2, pos.y - range // 2, range, range)
        cropImg = self.getCropImg(img, cnt)
        avgColorPerRow = np.average(cropImg, axis=0)
        avgColor = np.average(avgColorPerRow, axis=0)

        return avgColor

    def getColorThreshold(self, img, pos):
        max_dist = 20
        avgColor = self.getAverageColor(img, pos, 10)
        R = img[..., 0].astype(np.float32)
        G = img[..., 1].astype(np.float32)
        B = img[..., 2].astype(np.float32)
        # Euclidean distance
        dist = (R - avgColor[0])**2 + (G - avgColor[1])**2 +\
         (B -avgColor[2])**2
        mask = dist < (max_dist**2)
        # mask = [[False,False ...]]
        # mask[..., None] = [[[False],[False]]]
        # repeat,3,axis=2 = [[[False,False,False],[False,False,False]]]
        res = np.repeat(mask[..., None], 3, axis=2) * img
        res = self.getThresholdOtsu(res)

        return res

    def getWordMask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 280, 500)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilate = cv2.dilate(edge, kernel, iterations=4)

        cnts, _ = cv2.findContours(dilate, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
        rects = self.rectModule.getGoodRect(cnts, Rect.WORD)
        mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

        return mask

    def getThresholdOtsu(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, res = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return res

    def getCropImg(self, img, rect):
        x, y, w, h = rect
        return img[y:y + h, x:x + w]

    def getRects(self, dilate, origin):
        # area = np.copy(origin)
        cnts, _ = cv2.findContours(dilate, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
        goodRects = []
        for cnt in cnts:
            peri = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 20 and w > 30 and w < 300 and h < 300:
                goodRects.append((x, y, w, h))

        return goodRects

    def getBestRect(self, rects, pos):
        bestRect = rects[0]
        for rect in rects:
            x, y, w, h = rect
            if pos.x >= x and pos.x <= x + w and pos.y >= y and pos.y <= y + h:
                bestRect = rect
                break

        return bestRect

    def getUIContour(self, origin, pos):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        wordMask = self.wordDetect.getWordMask(origin)
        colorThresh = self.getColorThreshold(origin, pos)
        mergeThresh = cv2.add(wordMask, colorThresh)
        mergeThresh_inv = cv2.bitwise_not(mergeThresh)

        resThresh = cv2.dilate(mergeThresh_inv, kernel, iterations=3)

        # edge = self.debug.imshow(
        #     40, 150, (lambda min, max: cv2.Canny(colorThresh, min, max)))
        # edge = cv2.Canny(resThresh, 40, 150)
        # dilate = cv2.dilate(edge, kernel, iterations=3)
        rects = self.getRects(resThresh, origin)
        rect = self.getBestRect(rects, pos)

        area = np.copy(origin)
        for r in rects:
            x, y, w, h = r
            cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('wordMask', wordMask)
        cv2.imshow('colorThresh', colorThresh)
        cv2.imshow('mergeThresh_inv', mergeThresh_inv)
        cv2.imshow('resThresh', resThresh)
        cv2.imshow('area', area)
        cv2.waitKey(0)

        return rect
