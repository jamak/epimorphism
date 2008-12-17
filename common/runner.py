import threading

class Runner(threading.Thread):
    def __init__(self, func):
        self.func = func
        threading.Thread.__init__(self)

    def run(self):
        self.func()

def run_as_thread(func):
    Runner(func).start()
