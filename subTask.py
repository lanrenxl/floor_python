import numpy as np
import logging
from floor import Floor_plan


# 子任务参数
class SubTaskParam:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.form = 0
        self.depth = 0


# 子任务结构体
class SubTask:
    def __init__(self, taskParam: SubTaskParam, _image: np.ndarray, _taskID: int):
        self.floorCalc = None
        self.image = _image.copy()
        self.height = taskParam.height
        self.width = taskParam.width
        self.form = taskParam.form
        self.depth = taskParam.depth
        self.retStr = ""
        self.grid_img = None
        self.bubble_img = None
        self.taskID = _taskID

    # 调用floorPlan类运行算法计算结果
    def Run(self):
        logging.info("subtask %d start", self.taskID)
        self.floorCalc = Floor_plan(self.image)
        self.retStr, self.grid_img, self.bubble_img = self.floorCalc.getResult()
        logging.info("subtask %d end", self.taskID)

    def getRetStr(self):
        return self.retStr

    def getGridImg(self):
        return self.grid_img

    def getBubbleImg(self):
        return self.bubble_img
