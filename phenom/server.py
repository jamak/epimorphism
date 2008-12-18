import threading
import socket


class Server(threading.Thread):
    def __init__(self, cmdcenter):

        threading.Thread.__init__(self)
        self.cmdcenter = cmdcenter

        self.com = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.com.bind(('', 8563))
        self.com.listen(1)

    def __del__(self):
        print "close server"
        self.com.close()

    def run(self):
        channel, details = self.com.accept()
        print "connected to: ", details
        while(True):
            cmd = channel.recv ( 100 )
            print "executing: ", cmd
            res = self.cmdcenter.cmd(cmd, True)
            channel.send (str(res))
        channel.close()
