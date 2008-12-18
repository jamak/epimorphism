import time

from aer.datapath import *


class Animator(object):

    def __init__(self):
        self.time = time.clock
        self.paths = []


    def radial_2d(self, obj, i, spd, z0, z1):
        self.animate_var((obj, i), "radial_2d", spd, {"s" : z0, "e" : z1, 'loop' : False})


    def linear_1d(self, obj, i, spd, x0, x1):
        self.animate_var((obj, i), "linear_1d", spd, {"s" : x0, "e" : x1, 'loop' : False})


    def animate_var(self, var, type, speed, data, exclude=True):
        if(exclude and any([path[0] == var for path in self.paths])):
            return False
        self.paths.append((var, lambda t: eval(type)(t, data), self.time(), speed))
        return True


    def execute_paths(self):

        t = self.time()

        # execute paths
        for path in self.paths[::-1]:
            (res, status) = path[1]((t - path[2]) / path[3])
            path[0][0][path[0][1]] = res
            if(not status):
                self.paths.remove(path)


