# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2023/2/25 20:11
# ------------------------------------------------
import logging


# 设置打印日志的级别，level级别以上的日志会打印出
# level=logging.DEBUG 、INFO 、WARNING、ERROR、CRITICAL
def log_testing():
    # 此处进行Logging.basicConfig() 设置，后面设置无效
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.DEBUG)
    logging.debug('debug，用来打印一些调试信息，级别最低')
    logging.info('info，用来打印一些正常的操作信息')
    logging.warning('waring，用来用来打印警告信息')
    logging.error('error，一般用来打印一些错误信息')
    logging.critical('critical，用来打印一些致命的错误信息，等级最高')


import numpy as np
from scipy.spatial import ConvexHull

points = np.random.rand(30, 2)  # 30 random points in 2-D
hull = ConvexHull(points)

print(hull.vertices)

# log_testing()
