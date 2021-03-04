import numpy as np


def getAngle(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    cosAngle = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    angle = np.arccos(cosAngle)
    return np.degrees(angle)


cnt = np.array([[[531, 422]], [[534, 494]], [[780, 492]], [[778, 420]]])

cnt = cnt.reshape((-1, 2))
max_cos = np.max(
    [getAngle(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
print(max_cos)
