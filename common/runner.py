import threading


class Runner(threading.Thread):
    ''' A runner object spawns a new thread to call a function '''

    def __init__(self, func):
        self.func = func

        # init thread
        threading.Thread.__init__(self)


    def run(self):

        # call func
        self.func()


def run_as_thread(func):

    # create and start thread
    Runner(func).start()

