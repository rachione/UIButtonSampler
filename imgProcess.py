import cv2
import uuid
import numpy as np


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


class ImgProcess:
    def __init__(self):
        self.debug = TrackbarDebug()

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
        max_dist = 60
        avgColor = self.getAverageColor(img, pos, 30)
        R = img[..., 0].astype(np.float32)
        G = img[..., 1].astype(np.float32)
        B = img[..., 2].astype(np.float32)
        # Euclidean distance
        sq_dist = (R - avgColor[0]) ** 2 + \
            (G - avgColor[1]) ** 2 + (B - avgColor[2]) ** 2
        mask = sq_dist < (max_dist**2)
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
            if h > 20 and w > 30 and w < 500 and h < 500:
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

        colorThresh = self.getColorThreshold(origin, pos)
        # edge = self.debug.imshow(
        #     40, 150, (lambda min, max: cv2.Canny(colorThresh, min, max)))
        edge = cv2.Canny(colorThresh, 40, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilate = cv2.dilate(edge, kernel)
        rects = self.getRects(dilate, origin)
        rect = self.getBestRect(rects, pos)

        # area = np.copy(origin)
        # for r in rects:
        #     x, y, w, h = r
        #     cv2.rectangle(area, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('colorThresh', colorThresh)
        # cv2.imshow('dilate', dilate)
        # cv2.imshow('area', area)
        # cv2.waitKey(0)

        return rect
