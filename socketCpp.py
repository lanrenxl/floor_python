# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2023/2/19 19:39
# ------------------------------------------------
import socket
import logging

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

