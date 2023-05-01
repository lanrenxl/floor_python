# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2023/4/13 20:03
# ------------------------------------------------
from time import time
import datetime
import pyodbc


# 没有权限判断
# 数据库连接
# 数据库增删改查
class DB_operator:
    # 连接数据库
    def __init__(self):
        self.my_connect = pyodbc.connect('DRIVER={SQL Server};SERVER=127.0.0.1;DATABASE=floor;UID=web;PWD=1234')
        if self.my_connect:
            print("连接数据库成功!\n")
        self.my_cursor = self.my_connect.cursor()
        pass

    def save_result(self, bubble_img_path:str, grid_img_path:str):
        pass

    def getResult(self, ):
        pass


if __name__ == '__main__':
    db = DB_operator()
    pass