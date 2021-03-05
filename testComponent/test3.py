import numpy as np
import cv2 as cv


def getAngle(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    cosAngle = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    angle = np.arccos(cosAngle)
    return np.degrees(angle)


def find_contour(bin):
    squares = []
    contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST,
                                           cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
        x, y, w, h = cv.boundingRect(cnt)
        if len(
                cnt
        ) == 4 and h > 20 and w > 30 and w < 300 and h < 300 and cv.isContourConvex(
                cnt):
            squares.append(cnt)
    return squares


def find_squares_entity(bin):
    squares = []
    contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST,
                                           cv.CHAIN_APPROX_SIMPLE)
    for oriCnt in contours:
        cnt_len = cv.arcLength(oriCnt, True)
        x, y, w, h = cv.boundingRect(oriCnt)
        if h > 20 and w > 30 and w < 300 and h < 300:
            for ratio in range(3, 10, 2):
                cnt = cv.approxPolyDP(oriCnt, 0.01 * ratio * cnt_len, True)
                if len(cnt) == 4 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    maxAngle = np.max([
                        getAngle(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                        for i in range(4)
                    ])
                    if maxAngle < 100:
                        squares.append(cnt)

    return squares


def find_squares(src):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv.filter2D(src, -1, kernel)

    blurImg = cv.GaussianBlur(src, (11, 11), 0)
    squares = []
    for img in [src, blurImg, sharpen]:
        for thrs1 in range(0, 301, 100):
            for thrs2 in range(0, 301, 100):

                bin = cv.Canny(img, thrs1, thrs2)
                bin = cv.dilate(bin, None)
                squares += find_squares_entity(bin)
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # for thrs in range(0, 255, 26):
        #     _, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
        #     squares += find_squares_entity(bin)
    return squares


def main():

    img = cv.imread('testImg/s8.jpg')
    squares = find_squares(img)
    cv.drawContours(img, squares, -1, (0, 255, 0), 3)
    cv.imshow('squares', img)
    ch = cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
