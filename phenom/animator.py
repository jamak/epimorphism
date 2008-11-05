import time

from aer.datapath import *



class Animator:


    recompile = False
    paths = []

    d_paths = []


    def __init__(self, state):
        self.state = state


    def animate_var(self, var, type, speed, data, exclude=True):
        if(exclude and any([path[0] == var for path in self.paths])):
            return False
        self.paths.append((var, lambda t: eval(type)(t, data), time.clock(), speed / 1000.0))
        return True        


    def delta_var(self, var, dim, delta, speed):
        new_path = True
        for path in self.d_paths:
            if(path[0] == var):
                new_path = False
                path[2] += delta
                path[4] = (eval(var) < path[2]) and 1 or -1
        if(new_path):
            self.d_paths.append([var, (lambda cur, d_t, v : eval((dim == 1) and "d_linear_1d" or "d_radial_2d")(cur, d_t, speed, v)), 
                                eval(var), eval(var) + delta, (delta < 0) and -1 or 1, speed, time.clock()])


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

        for path in self.d_paths:
            cur = eval(path[0])            
            res = path[1](eval(path[0]), t - path[6], path[4])
            if((cur <= path[3] <= res) or (cur >= path[3] >= res)):
                res = path[3]
                self.d_paths.remove(path)
            exec(path[0] + " = " + str(res))
            path[6] = t            
                
        if(self.recompile):
            self.recompile = False
            return set(["recompile"])

        return {}
