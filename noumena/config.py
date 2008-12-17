import noumena

from noumena.migration import *

import os.path



class State(object):

    def __init__(self, **vars):

        # update dict with migrated vars
        self.__dict__.update(migrate(vars))

        # set par defaults
        for i in xrange(len(self.par_names)):
            self.par[i] = float(self.par_defaults[self.par_names[i]])


class Profile(object):

    def __init__(self, **vars):

        self.__dict__.update(vars)


class Context(object):

    def __init__(self, **vars):

        self.__dict__.update(vars)


class ConfigManager(object):

    extension_names = {"est" : "state", "prf" : "profile", "ctx" : "context"}

    def load_dict(self, name, **additional_vars):

        extension = name.split(".")[1]
        file = open("config/" + self.extension_names[extension] + "/" + name)
        vars = eval(file.read().replace("\n", ""))
        vars.update(additional_vars)
        file.close()
        return eval(self.extension_names[extension].capitalize())(**vars)


    def outp_dict(self, name, obj):

        file = open(name, "w")
        file.write(repr(obj.__dict__).replace(",", ",\n"))
        file.close()


    def save_state(self, state, image, name=""):

        state.VERSION = noumena.VERSION

        if(name == ""):
            i = 0
            while(os.path.exists("config/state/state_" + str(i) + ".est")) : i += 1
            name = "state_" + str(i)

        if(image):
            image.save("image/image_" + str(i) + ".png")

        self.outp_dict("config/state/" + name + ".est", state)

