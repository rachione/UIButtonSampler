import cv2
import uuid
import numpy as np
from enum import Enum
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\exe\Tesseract-OCR\tesseract.exe"


class Rect(Enum):
    WORD = 1
    BUTTON = 2


class Math:
    @staticmethod
    def getAngle(p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        cosAngle = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        angle = np.arccos(cosAngle)
        return np.degrees(angle)

    @staticmethod
    def isGoodQuad(cnt, limitAngle):
        cnt = cnt.reshape(-1, 2)
        maxAngle = np.max([
            Math.getAngle(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
            for i in range(4)
        ])
        return maxAngle <= limitAngle


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
            Rect.BUTTON: (h > 20 and w > 30 and w < 300 and h < 300),
        }[type]

    def sort(self, rects):
        return sorted(rects, key=lambda rect: rect[2] * rect[3])

    # remove contained and overlapped rects
    def removeBadRects(self, rects):
        rects = self.sort(rects)
        for rect1 in rects[:]:
            x1, y1, w1, h1 = rect1
            for rect2 in rects[:]:
                x2, y2, w2, h2 = rect2
                if rect1 == rect2:
                    continue

                isContain = (x2 + w2) >= (x1 + w1)\
                    and x2 <= x1\
                    and y2 <= y1\
                    and (y2 + h2) >= (y1 + h1)

                isOverlap = not ((x1 >= (x2 + w2)) or ((x1 + w1) <= x2) or
                                 (y1 >= (y2 + h2)) or ((y1 + h1) <= y2))
                if isContain:
                    rects.remove(rect1)
                    break

        return rects

    def getBiggerRect(self, img, rects):
        imgH, imgW, _ = img.shape
        for i in range(len(rects)):
            x, y, w, h = rects[i]
            rects[i] = max(0, x - 3), max(0, y - 3),\
                min(imgW, w + 6), min(imgH, h + 6)

        return rects

    def getRects(self, cnts):
        return [cv2.boundingRect(cnt) for cnt in cnts]

    def getGoodRects(self, rects):
        rects = self.removeBadRects(rects)
        return rects

    def isRectHasPoint(self, rect, pos):
        x, y, w, h = rect
        posX, posY = pos
        return (posX >= x and posX <= x + w and posY >= y and posY <= y + h)

    def getRectHasPoint(self, rects, pos):
        bestRect = rects[0]
        for rect in rects:
            x, y, w, h = rect
            if pos.x >= x and pos.x <= x + w and pos.y >= y and pos.y <= y + h:
                bestRect = rect
                break

        return bestRect


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
            (B - avgColor[2])**2
        mask = dist < (max_dist**2)
        # mask = [[False,False ...]]
        # mask[..., None] = [[[False],[False]]]
        # repeat,3,axis=2 = [[[False,False,False],[False,False,False]]]
        res = np.repeat(mask[..., None], 3, axis=2) * img
        res = self.getThresholdOtsu(res)

        return res

    def getThresholdOtsu(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, res = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return res

    def getCropImg(self, img, rect):
        x, y, w, h = rect
        return img[y:y + h, x:x + w]

    def getCropImgByPos(self, img, pos):
        rangeX = 600
        rangeY = 600
        pivotX = pos.x - rangeX // 2
        pivotY = pos.y - rangeY // 2
        if pivotX < 0:
            rangeX += pivotX
            pivotX = 0
        if pivotY < 0:
            rangeY += pivotY
            pivotY = 0
        rect = pivotX, pivotY, rangeX, rangeY
        return  (pivotX, pivotY),\
                (pos.x -pivotX,pos.y -pivotY),\
                self.getCropImg(img, rect)

    def findSquares(self, src, pos, type):
        resCnts = []
        cnts, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for oriCnt in cnts:
            rect = cv2.boundingRect(oriCnt)
            if self.rectModule.isGoodRange(rect, type):
                peri = cv2.arcLength(oriCnt, True)
                # make different epsilon for approximation
                for ratio in range(3, 10, 2):
                    cnt = cv2.approxPolyDP(oriCnt, 0.01 * ratio * peri, True)
                    if  len(cnt) == 4 and\
                            cv2.isContourConvex(cnt) and\
                            self.rectModule.isRectHasPoint(rect, pos) and\
                            Math.isGoodQuad(cnt, 100):
                        resCnts.append(cnt)

        return resCnts

    def getVariantImgs(self, src):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(src, -1, kernel)
        blurImg = cv2.GaussianBlur(src, (11, 11), 0)
        return [src, blurImg, sharpen]

    def findSquaresByThreshhold(self, src, pos, type):
        resCnts = []
        for img in self.getVariantImgs(src):
            for channel in cv2.split(img):
                for thrs in range(0, 255, 26):
                    _, bin = cv2.threshold(channel, thrs, 255,
                                           cv2.THRESH_BINARY)
                    resCnts += self.findSquares(bin, pos, type)
        return resCnts

    def findSquaresByCanny(self, src, pos, type):
        resCnts = []
        for img in self.getVariantImgs(src):
            for thresh1 in range(0, 301, 100):
                for thresh2 in range(0, 301, 100):
                    edge = cv2.Canny(img, thresh1, thresh2)
                    dilate = cv2.dilate(edge, None)
                    resCnts += self.findSquares(dilate, pos, type)
            if len(resCnts) != 0:
                break

        return resCnts

    def getUIContour(self, origin, pos):
        pivot, posInCrop, crop = self.getCropImgByPos(origin, pos)
        cnts = self.findSquaresByCanny(crop, posInCrop, Rect.BUTTON)
        if len(cnts) == 0:
            print('no rect')
            return None

        rects = self.rectModule.getRects(cnts)
        # restore position
        rects = [(rect[0] + pivot[0], rect[1] + pivot[1], rect[2], rect[3])
                 for rect in rects]
        rects = self.rectModule.getGoodRects(rects)

        rect = self.rectModule.getRectHasPoint(rects, pos)

        area = np.copy(origin)
        for r in rects:
            x, y, w, h = r
            cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('area', area)
        cv2.waitKey(0)

        return rect
