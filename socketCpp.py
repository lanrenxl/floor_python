# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2023/2/19 19:39
# ------------------------------------------------
import socket
import logging

# 初始化日志格式
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(filename)s:%(funcName)s',
                    level=logging.DEBUG)


# 与C++端的底层通信模块
class SocketCppServer:
    # 初始化函数, 初始化服务器并监听1234端口
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

    # 重置socket连接
    # 在C++端发送end时中断连接并等待新的连接
    def resetConnect(self):
        self.connection.close()
        logging.info("__server started, wait client...")
        self.connection, self.address = self.server.accept()
        logging.info("client connected: %s %s", self.connection, self.address)

    # 发送字符串信息
    # 用于发送json类信息
    def sendMessage(self, bufferStr: str):
        bufferBytes = bufferStr.encode("ascii")
        return self.sendBytes(bufferBytes)

    # 发送字节流信息
    # 用于发送图像数据
    def sendBytes(self, bufferBytes: bytes):
        length = len(bufferBytes)
        sendSize = 0
        while sendSize < length:
            res = self.connection.send(bufferBytes[sendSize:length])
            logging.info("send buffer: %d", res)
            sendSize += res
        return sendSize

    # 获取字节流信息
    # 用于接受图像数据
    def getBuffer(self, size) -> bytes:
        bufferOfAll = bytes()
        getSize = 0
        while getSize < size:
            bufferByte = self.connection.recv(size)
            bufferOfAll += bufferByte
            logging.info("get buffer: %d", len(bufferByte))
            getSize += len(bufferByte)
        return bufferOfAll

    # 获取字符串信息
    # 用于接收json任务类信息
    def getMessage(self) -> str:
        bufferByte = self.connection.recv(1024)
        logging.info("get message: %d", len(bufferByte))
        bufferStr = bufferByte.decode("ascii", "strict")
        return bufferStr

