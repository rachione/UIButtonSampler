import os
import time
import pyautogui
from imgProcess import *
from threading import Thread
from ctypes import *
from ctypes.wintypes import *
user32 = windll.user32

VK = {
    'VK_LBUTTON': 0x01,
    'VK_RBUTTON': 0x02,
    'VK_SHIFT': 0x10,
    'VK_CONTROL': 0x11,
    'VK_KEYQ': 0x51,
}


class Sampler:

    def __init__(self):
        self.imgProcess = ImgProcess()
        self.count = 0

    def getCursorPos(self):
        ppoint = pointer(POINT())
        user32.GetCursorPos(ppoint)
        return ppoint[0]

    def getScreenShot(self):
        img = pyautogui.screenshot()
        return self.imgProcess.convertBGR(img)

    def addSample(self):

        origin = self.getScreenShot()
        pos = self.getCursorPos()
        rect = self.imgProcess.getUIContour(origin, pos)
        sample = self.imgProcess.getCropImg(origin, rect)
        self.imgProcess.imwrite(sample)
        self.count += 1
        print("Add sample %d" % self.count)


class KeyRecorder:

    def __init__(self):
        self.key = VK['VK_LBUTTON']
        self.sampler = Sampler()

    def record(self):
        self.sampler.addSample()

    def keyDownJob(self, keys, callback):
        isKeyDown = False
        while True:
            allPressed = all(
                [user32.GetAsyncKeyState(key) & 0x8000 for key in keys])
            if allPressed and isKeyDown == False:
                isKeyDown = True
                callback()
            if not allPressed and isKeyDown == True:
                isKeyDown = False
            time.sleep(0.01)

    def exit(self):
        # child process call termination
        os._exit(0)

    def start(self):
        # ctrl+click = add samples
        keys = [VK['VK_CONTROL'], VK['VK_LBUTTON']]
        Thread(target=self.keyDownJob, args=(keys, self.record)).start()
        # ctrl+q = exit application
        keys = [VK['VK_CONTROL'], VK['VK_KEYQ']]
        Thread(target=self.keyDownJob, args=(keys, self.exit)).start()


if __name__ == '__main__':
    kr = KeyRecorder()
    kr.start()
