import time

from aer.datapath import *



class Animator:

    paths = []

    def __init__(self, state):
        self.state = state


    def animate_var(self, var, type, speed, data):
        self.paths.append((var, lambda t: eval(type)(t, data), time.clock(), speed / 1000.0))
        

    def do(self):
        t = time.clock()
        for path in self.paths:
            try:
                exec(path[0] + " = " + str(path[1]((t - path[2]) / path[3])))
            except:
                self.paths.remove(path)
                print 5
                
