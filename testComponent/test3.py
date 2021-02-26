import numpy as np
import cv2 as cv


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares_entity(bin):
    squares = []
    contours, _hierarchy = cv.findContours(
        bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
        if len(cnt) == 4 and h > 20 and w > 30 and w < 300 and h < 300 and cv.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max(
                [angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            if max_cos < 0.3:
                squares.append(cnt)

    return squares


def find_squares(img):
    # img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs1 in range(0, 300, 100):
            for thrs2 in range(100, 300, 100):

                bin = cv.Canny(gray, thrs1, thrs2)
                squares += find_squares_entity(bin)
                bin = cv.dilate(bin, None)
                squares += find_squares_entity(bin)
    return squares


def main():

    img = cv.imread('testImg/s1.jpg')
    squares = find_squares(img)
    cv.drawContours(img, squares, -1, (0, 255, 0), 3)
    cv.imshow('squares', img)
    ch = cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
