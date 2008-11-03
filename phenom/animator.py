import time

from aer.datapath import *



class Animator:


    recompile = False
    paths = []


    def __init__(self, state):
        self.state = state


    def animate_var(self, var, type, speed, data, exclude=True):
        if(exclude and any([path[0] == var for path in self.paths])):
            return False
        self.paths.append((var, lambda t: eval(type)(t, data), time.clock(), speed / 1000.0))
        return True        


    def animate_t(self, new_t):
        self.state.T = new_t
        self.recompile = True
        

    def do(self):

        # get time
        t = time.clock()

        # execute paths
        for path in self.paths:
            (res, status) = path[1]((t - path[2]) / path[3]) 
            exec(path[0] + " = " + str(res))
            if(not status):
                self.paths.remove(path)
                print 5
                
        if(self.recompile):
            self.recompile = False
            return set(["recompile"])

        return {}
