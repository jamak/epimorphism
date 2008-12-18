import time

from aer.datapath import *


class Animator(object):

    def __init__(self):
        self.time = time.clock
        self.paths = []


    def radial_2d(self, i, spd, z0, z1):
        self.animate_var("zn%d" % i, self.zn_set_i(i), "radial_2d", spd, {"s" : z0, "e" : z1, 'loop' : False})


    def linear_1d(self, i, spd, x0, x1):
        self.animate_var("par%d" % i, self.par_set_i(i), "linear_1d", spd, {"s" : x0, "e" : x1, 'loop' : False})


    def animate_var(self, id, setter, type, speed, data, exclude=True):
        if(exclude and any([path[0] == id for path in self.paths])):
            return False
        self.paths.append((id, setter, lambda t: eval(type)(t, data), self.time(), speed / 1000.0))
        return True


    def execute_paths(self):

        t = self.time()

        # execute paths
        for path in self.paths[::-1]:
            (res, status) = path[2]((t - path[3]) / path[4])
            path[1](res)
            if(not status):
                self.paths.remove(path)


