from socketCpp import SocketCppServer
import numpy as np
import logging
import json
import cv2 as cv
from subTask import SubTask
from subTask import SubTaskParam
import concurrent.futures


def subTaskThread(subtaskParam: SubTaskParam, image: np.ndarray, ID: int):
    subTask = SubTask(subtaskParam, image, ID)
    logging.info("subtask %d created", ID)
    logging.info("subtask %d running", ID)
    subTask.Run()
    return ID, subTask.retStr, subTask.grid_img, subTask.bubble_img


class Frontend:
    def __init__(self):
        self.__server = SocketCppServer()
        self.subTaskNum = 0

    def __getTaskParam(self):
        # 获取一批任务的参数
        self.subTaskNum = 0
        logging.info("start wait task param ")
        taskParamStr = self.__server.getMessage()
        if taskParamStr == "":
            logging.error("failed get message from cpp client ")
            return None
        elif taskParamStr[:3] == "end":
            logging.info("client closed, start reset socket and wait new client ")
            self.__server.resetConnect()
            return self.__getTaskParam()
        else:
            logging.info("success get TaskParam from cpp client: %s ", taskParamStr)
            taskParamJS = json.loads(taskParamStr)
            # 获取任务数量
            if "subTaskNum" in taskParamJS:
                self.subTaskNum = taskParamJS["subTaskNum"]
            self.__server.sendMessage("success\0")

    def __getSubTaskParam(self):
        logging.info("start wait subtask param ")
        subTaskParamStr = self.__server.getMessage()
        if subTaskParamStr == "":
            logging.error("failed get message from cpp client ")
            return None
        else:
            logging.info("success get subtask param from cpp client: %s ", subTaskParamStr)
            taskParamJS = json.loads(subTaskParamStr)
            subTaskParam = SubTaskParam()
            subTaskParam.height = taskParamJS["height"]
            subTaskParam.width = taskParamJS["width"]
            subTaskParam.depth = taskParamJS["depth"]
            subTaskParam.form = taskParamJS["format"]
            self.__server.sendMessage("success\0")
            return subTaskParam

    def __getSubTaskImage(self, subTaskParam: SubTaskParam):
        # 返回获取的图像, cv2的格式, 再该函数中全部封装好
        imageBytes = self.__server.getBuffer(subTaskParam.width * subTaskParam.height * subTaskParam.form)
        if len(imageBytes) == 0:
            logging.error("failed get _image from cpp client")
        imageArrays = np.frombuffer(imageBytes, np.uint8)
        imageArrays.resize(subTaskParam.height, subTaskParam.width, subTaskParam.form)
        logging.info("success get _image from cpp client")
        image = imageArrays
        self.__server.sendMessage("success\0")
        return image
        pass

    def __sendSubTaskImage(self, _image: np.ndarray):
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
            self.__getTaskParam()
            # Create a thread pool with 5 threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.subTaskNum) as executor:
                futures = list()
                # Submit tasks to the thread pool
                for i in range(self.subTaskNum):
                    # 循环获取任务参数与任务图像数据, 创建线程执行任务
                    subTaskParam = self.__getSubTaskParam()
                    image = self.__getSubTaskImage(subTaskParam)
                    futures.append(executor.submit(subTaskThread, subTaskParam, image, i))
                # Wait for all tasks to complete and get the results
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
                # 根据任务ID进行排序
                results = sorted(results, key=lambda res: res[0])
                for result in results:
                    # 排序好的任务结果顺序发送给c++端
                    ID, retStr, grid_img, bubble_img = result
                    self.__sendRetJS(retStr)
                    self.__sendSubTaskImage(grid_img)
                    self.__sendSubTaskImage(bubble_img)


if __name__ == '__main__':
    pass
    frontend = Frontend()
    frontend.taskRun()
