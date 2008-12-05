import time

from aer.datapath import *



class Animator:

    paths = []


    def __init__(self):
        pass


    def animate_var(self, id, setter, type, speed, data, exclude=True):
        if(exclude and any([path[0] == id for path in self.paths])):
            return False
        self.paths.append((id, setter, lambda t: eval(type)(t, data), time.clock(), speed / 1000.0))
        return True        



    def animate_t(self, new_t):

        self.engine.compile_kernel()
        

    def do(self):

        # get time
        t = time.clock()

        # execute paths
        for path in self.paths[::-1]:
            (res, status) = path[2]((t - path[3]) / path[4]) 
            path[1](res)
            if(not status):
                self.paths.remove(path)
            

