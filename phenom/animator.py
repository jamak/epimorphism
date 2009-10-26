from aeon.datapath import *

from common.log import *
set_log("ANIMATOR")

class Animator(object):
    ''' The Animator class is a module for Cmdcenter that is
        responsible for automation of data. '''


    def __init__(self):
        self.paths = []


    def radial_2d(self, obj, idx, spd, z0, z1):
        ''' Helper function for creating radial_2d paths. '''
        debug("Radial 2d: %s %s %s %s %s", obj, idx, spd, str(z0), str(z1))

        self.animate_var("radial_2d", obj, idx, spd, {"s" : z0, "e" : z1, 'loop' : False})


    def linear_1d(self, obj, idx, spd, x0, x1):
        ''' Helper function for creating linear_1d paths. '''
        debug("Linear 1d: %s %s %s %s %s", obj, idx, spd, x0, x1)

        self.animate_var("linear_1d", obj, idx, spd, {"s" : x0, "e" : x1, 'loop' : False})


    def animate_var(self, type, obj, idx, speed, data, exclude=True):
        ''' Adds a path to the animator. '''

        # if !exclude, don't add another path if one exists
        if(exclude and any([(path["obj"] == obj and path["idx"] == idx) for path in self.paths])):
            return False

        # add path

        self.paths.append({"obj": obj, "idx":idx, "start": self.time(), "speed": speed, "func":(lambda t: eval(type)(t, data))})

        return True


    def execute_paths(self):

        # get time
        t = self.time()

        # execute paths
        for path in self.paths[::-1]:

            # execute path
            (res, status) = path["func"]((t - path["start"]) / path["speed"])

            # set result
            path["obj"][path["idx"]] = res

            # if necessary, remove path
            if(not status):
                self.paths.remove(path)


