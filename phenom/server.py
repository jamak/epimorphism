import threading
import socket

from common.runner import *

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
        print "close server"
        self.com.close()


    def handle_connection(self, channel):

        # get packets
        while(not cmdcenter.context.exit):

            # receive data
            cmd = channel.recv ( 100 )

            # execute command
            print "executing: ", cmd
            res = self.cmdcenter.cmd(cmd, True)

            # send response
            channel.send (str(res))

        # close channel
        channel.close()


    def run(self):

        # accept connections
        while(not cmdcenter.context.exit):

            # accept connection
            channel, details = self.com.accept()
            print "connected to: ", details

            # spawn thread to handle connection
            run_as_thread(self.handle_connection(channel))
