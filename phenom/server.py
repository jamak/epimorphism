import threading
import socket

from common.runner import *

from common.log import *
set_log("SERVER")

class Server(threading.Thread):
    ''' The Server object is responsible for creating and maintaining
        connects to outside clients, and relaying messages to the
        Cmdcenter '''

    def __init__(self, cmdcenter):

        self.cmdcenter = cmdcenter

        # create & start socket
        self.com = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.com.bind(('', 8563))
        self.com.listen(1)

        # init thread
        threading.Thread.__init__(self)


    def __del__(self):
        debug("Closeing server")
        self.com.close()


    def handle_connection(self, channel):
        ''' Receive & parse data from connection '''

        # get packets
        while(not self.cmdcenter.context.exit):

            # receive data
            cmd = channel.recv ( 100 )

            # execute command
            info("executing: %s" % cmd)
            res = self.cmdcenter.cmd(cmd, True)

            # send response
            channel.send (str(res))

        # close channel
        channel.close()


    def run(self):
        ''' Async wait for a connection '''

        # accept connections
        while(not self.cmdcenter.context.exit):

            # accept connection
            channel, details = self.com.accept()
            info("connected to: %s" % details)

            # spawn thread to handle connection
            async(self.handle_connection(channel))
