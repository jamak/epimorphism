import time

from aer.datapath import *


class Animator(object):

    paths = []


    def __init__(self):
        self.time = time.clock


    def animate_var(self, id, setter, type, speed, data, exclude=True):
        if(exclude and any([path[0] == id for path in self.paths])):
            return False
        self.paths.append((id, setter, lambda t: eval(type)(t, data), self.time(), speed / 1000.0))
        return True


    def do(self):

        t = self.time()

        # execute paths
        for path in self.paths[::-1]:
            (res, status) = path[2]((t - path[3]) / path[4])
            path[1](res)
            if(not status):
                self.paths.remove(path)


