# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2023/2/19 19:39
# ------------------------------------------------
import socket
import numpy as np
import logging
import json
from floor import Floor_plan
import cv2 as cv

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.DEBUG)


class SocketCppServer:
    def __init__(self):
        self.maxRetryTime = 3
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("localhost", 1234))
        self.server.listen(0)
        self.connection = None
        self.address = None
        logging.info("__server started, wait client...")
        self.connection, self.address = self.server.accept()
        logging.info("client connected: %s %s", self.connection, self.address)

    def resetConnect(self):
        self.connection.close()
        logging.info("__server started, wait client...")
        self.connection, self.address = self.server.accept()
        logging.info("client connected: %s %s", self.connection, self.address)

    def sendMessage(self, bufferStr: str):
        bufferBytes = bufferStr.encode("ascii")
        return self.sendBytes(bufferBytes)

    def sendBytes(self, bufferBytes: bytes):
        length = len(bufferBytes)
        sendSize = 0
        while sendSize < length:
            res = self.connection.send(bufferBytes[sendSize:length])
            logging.info("send buffer: %d", res)
            sendSize += res
        return sendSize

    def getBuffer(self, size) -> bytes:
        bufferOfAll = bytes()
        getSize = 0
        while getSize < size:
            bufferByte = self.connection.recv(size)
            bufferOfAll += bufferByte
            logging.info("get buffer: %d", len(bufferByte))
            getSize += len(bufferByte)
        return bufferOfAll

    def getMessage(self) -> str:
        bufferByte = self.connection.recv(1024)
        logging.info("get message: %d", len(bufferByte))
        bufferStr = bufferByte.decode("ascii", "strict")
        return bufferStr


class Frontend:
    def __init__(self):
        self.__server = SocketCppServer()
        self.floorCalc = None
        self.image = None
        self.height = 0
        self.width = 0
        self.form = 0
        self.depth = 0

    def __getTaskParam(self):
        logging.info("start wait task param ")
        taskParamStr = self.__server.getMessage()
        if taskParamStr == "":
            logging.error("failed get message from cpp client ")
            return False
        elif taskParamStr[:3] == "end":
            logging.info("client closed, start reset socket and wait new client ")
            self.__server.resetConnect()
            return self.__getTaskParam()
        else:
            logging.info("success get message from cpp client: %s ", taskParamStr)
            taskParamJS = json.loads(taskParamStr)
            self.height = taskParamJS["height"]
            self.width = taskParamJS["width"]
            self.depth = taskParamJS["depth"]
            self.form = taskParamJS["format"]
            self.__server.sendMessage("success\0")
            return True

    def __getImage(self):
        # 返回获取的图像, cv2的格式, 再该函数中全部封装好
        imageBytes = self.__server.getBuffer(self.width*self.height*self.form)
        if len(imageBytes) == 0:
            logging.error("failed get _image from cpp client")
        imageArrays = np.frombuffer(imageBytes, np.uint8)
        imageArrays.resize(self.height, self.width, self.form)
        logging.info("success get _image from cpp client")
        self.image = imageArrays
        # cv.imwrite("get.png", imageArrays)
        self.__server.sendMessage("success\0")
        pass

    def __sendImage(self, _image: np.ndarray):
        # 发送图像, 包括提前发送图像格式, 再发送图像本身
        w, h, c = _image.shape
        paramJS = dict()
        paramJS["width"] = w
        paramJS["height"] = h
        paramJS["format"] = c
        paramStr = json.dumps(paramJS)
        paramStr += '\0'
        res = self.__server.sendMessage(paramStr)
        if res:
            logging.info("success send image message from cpp client")
        res = self.__server.getMessage()
        logging.info("success get success message from cpp client: %s", res)
        if res != "":
            logging.info("cpp client get message")
            imageBytes = _image.tobytes()
            imageArrays = np.frombuffer(imageBytes, np.uint8)
            imageArrays.resize(w, h, c)
            cv.imwrite("get.png", imageArrays)
            res = self.__server.sendBytes(imageBytes)
            if res:
                logging.info("success send image from cpp client")
            res = self.__server.getMessage()
            if len(res) > 0:
                logging.info("success get success message from cpp client: %s", res)

    def __sendRetJS(self, retStr):
        retStr += '\0'
        self.__server.sendMessage(retStr)
        res = self.__server.getMessage()
        if len(res) > 0:
            logging.info("success get success message from cpp client: %s", res)

    def taskRun(self):
        while True:
            if self.__getTaskParam():
                self.__getImage()
            self.floorCalc = Floor_plan(self.image)
            retStr, grid_img, bubble_img = self.floorCalc.getResult()
            self.__sendRetJS(retStr)
            self.__sendImage(grid_img)
            self.__sendImage(bubble_img)
            logging.info("task end")


if __name__ == '__main__':
    pass
    frontend = Frontend()
    frontend.taskRun()
